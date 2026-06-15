"""Minimal OpenRouter client using only the Python standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .catalog import OPENROUTER_BASE_URL


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter API request fails."""


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
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.app_title = app_title
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            referer=os.environ.get("OPENROUTER_HTTP_REFERER"),
            app_title=os.environ.get("OPENROUTER_APP_TITLE"),
        )

    def chat_completion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is required for chat completions")
        body = {"model": model, "messages": messages, **kwargs}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

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
