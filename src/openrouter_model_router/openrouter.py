"""Minimal OpenRouter client that reads usage and cost back off every response."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .catalog import OPENROUTER_BASE_URL
from .transport import HttpRequest, HttpTransport, TransportError, UrllibTransport
from .usage import TokenUsage, parse_usage

MISSING_KEY_MESSAGE = (
    "OPENROUTER_API_KEY is not set. The live OpenRouter path refuses to run without a "
    "credential; no request was sent. Inject a FakeTransport (openrouter_model_router."
    "transport.FakeTransport) to exercise this path without a key."
)


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter API request fails."""


@dataclass(frozen=True)
class ChatResult:
    """A chat completion plus what it actually cost."""

    model: str
    content: str
    usage: TokenUsage
    response: dict[str, Any]
    latency_ms: float
    finish_reason: str | None = None

    @property
    def reported_cost_usd(self) -> float | None:
        return self.usage.reported_cost_usd

    @property
    def prompt_tokens(self) -> int | None:
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self) -> int | None:
        return self.usage.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "content": self.content,
            "finish_reason": self.finish_reason,
            "latency_ms": round(self.latency_ms, 3),
            "usage": self.usage.to_dict(),
        }


class OpenRouterClient:
    """Small OpenAI-compatible client for OpenRouter chat completions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        referer: str | None = None,
        app_title: str | None = None,
        timeout: float = 60.0,
        transport: HttpTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.app_title = app_title
        self.timeout = timeout
        self.transport: HttpTransport = transport or UrllibTransport()

    @classmethod
    def from_env(cls, transport: HttpTransport | None = None) -> "OpenRouterClient":
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            referer=os.environ.get("OPENROUTER_HTTP_REFERER"),
            app_title=os.environ.get("OPENROUTER_APP_TITLE"),
            transport=transport,
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def require_api_key(self) -> None:
        """Fail loudly, before any request is built, when no credential exists."""

        if not self.has_api_key:
            raise OpenRouterError(MISSING_KEY_MESSAGE)

    def chat_completion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Return the raw OpenRouter response payload."""

        return self.chat(model, messages, **kwargs).response

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        """Call the model and return content *and* the usage reported with it."""

        self.require_api_key()
        body = {"model": model, "messages": messages, **kwargs}
        request = HttpRequest(
            method="POST",
            url=f"{self.base_url}/chat/completions",
            headers=self._headers(),
            body=json.dumps(body).encode("utf-8"),
            timeout=self.timeout,
        )

        started = time.monotonic()
        try:
            response = self.transport.send(request)
        except TransportError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0

        if response.status >= 400:
            detail = response.body.decode("utf-8", errors="replace")
            raise OpenRouterError(f"OpenRouter HTTP {response.status}: {detail}")

        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OpenRouterError(f"OpenRouter returned a non-JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise OpenRouterError(f"OpenRouter returned {type(payload).__name__}, expected a JSON object")

        content, finish_reason = _first_choice(payload)
        return ChatResult(
            model=str(payload.get("model") or model),
            content=content,
            usage=parse_usage(payload),
            response=payload,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers


def _first_choice(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choice = choices[0]
    if not isinstance(choice, dict):
        return "", None
    finish_reason = choice.get("finish_reason")
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content, finish_reason
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "".join(parts), finish_reason
    text = choice.get("text")
    if isinstance(text, str):
        return text, finish_reason
    return "", finish_reason
