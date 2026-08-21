"""Injectable HTTP transport, plus a Fake that makes the whole client testable
with no API key and no network.

No API key exists on this machine, so the model-calling path has to be provably
exercisable today. Everything that touches HTTP goes through :class:`HttpTransport`;
:class:`UrllibTransport` is the real one (standard library only) and
:class:`FakeTransport` is a first-class, shipped implementation -- not a test
fixture -- so downstream projects can run their own cost accounting end to end
before a credential ever appears.

The Fake records every request it is handed. That is what lets a test assert
the *negative* control that matters here: with no key set, the live path must
raise before the transport is ever called. A test that only checks for an
exception cannot tell "refused to send" from "sent and then failed".
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout: float = 60.0

    def json_body(self) -> Any:
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


class TransportError(RuntimeError):
    """Raised when a transport cannot complete a request at all."""


class HttpTransport(Protocol):
    def send(self, request: HttpRequest) -> HttpResponse:  # pragma: no cover - protocol
        ...


class UrllibTransport:
    """The real transport. Standard library only."""

    user_agent = "openrouter-model-router/0.2 (+https://github.com/dyuhaus/openrouter-model-router)"

    def send(self, request: HttpRequest) -> HttpResponse:
        headers = {"User-Agent": self.user_agent, **request.headers}
        req = urllib.request.Request(
            request.url,
            data=request.body,
            headers=headers,
            method=request.method,
        )
        try:
            with urllib.request.urlopen(req, timeout=request.timeout) as response:
                return HttpResponse(
                    status=getattr(response, "status", 200) or 200,
                    body=response.read(),
                    headers={k.lower(): v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            body = exc.read() if hasattr(exc, "read") else b""
            return HttpResponse(
                status=exc.code,
                body=body,
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(f"{request.method} {request.url} failed: {exc}") from exc


class FakeTransport:
    """A scripted transport for tests and for running without a key.

    Give it either a list of :class:`HttpResponse` objects returned in order, or
    a handler ``(HttpRequest) -> HttpResponse``. Every request is appended to
    :attr:`requests` so callers can assert on what was (or was not) sent.
    """

    def __init__(
        self,
        responses: list[HttpResponse] | None = None,
        handler: Callable[[HttpRequest], HttpResponse] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.handler = handler
        self.requests: list[HttpRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def last_request(self) -> HttpRequest | None:
        return self.requests[-1] if self.requests else None

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)
        if not self.responses:
            raise TransportError("FakeTransport has no scripted response left for this request")
        return self.responses.pop(0)

    @staticmethod
    def json_response(payload: Any, status: int = 200) -> HttpResponse:
        return HttpResponse(
            status=status,
            body=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
