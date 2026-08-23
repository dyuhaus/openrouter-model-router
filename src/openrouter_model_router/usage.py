"""Reading token usage and reported cost back off a chat-completion response.

Field names are NOT assumed. OpenRouter's documented shape (verified against
https://openrouter.ai/docs/use-cases/usage-accounting and the API reference on
2026-08-21) is::

    "usage": {
      "prompt_tokens": 194,
      "completion_tokens": 88,
      "total_tokens": 282,
      "cost": 0.00047,
      "is_byok": false,
      "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0,
                                "audio_tokens": 0},
      "completion_tokens_details": {"reasoning_tokens": 0},
      "cost_details": {"upstream_inference_cost": 0.0,
                       "upstream_inference_prompt_cost": 0.0,
                       "upstream_inference_completions_cost": 0.0}
    }

Two facts from that doc drive the defensive parsing here:

1. ``usage`` is typed OPTIONAL (``usage?: ResponseUsage``) and ``cost`` is
   optional inside it. A response with no usage block is a legal response, so
   the parser must return "unknown", never zero. Zero is a measurement; ``None``
   is the absence of one, and conflating them is how a cost model silently
   reports $0.00.
2. For streaming responses usage arrives only in the LAST SSE message, so a
   caller that stops reading early legitimately has no usage at all.

Other OpenAI-compatible gateways and OpenRouter's own ``/generation`` endpoint
spell the same quantities differently (``input_tokens``/``output_tokens``,
``tokens_prompt``/``tokens_completion``, ``total_cost``), so each quantity is
resolved through an alias list and the winning key is recorded in
``source_fields`` for the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# Ordered by preference. First key present and coercible wins.
PROMPT_TOKEN_ALIASES: tuple[str, ...] = (
    "prompt_tokens",
    "input_tokens",
    "promptTokens",
    "inputTokens",
    "tokens_prompt",
    "native_tokens_prompt",
)
COMPLETION_TOKEN_ALIASES: tuple[str, ...] = (
    "completion_tokens",
    "output_tokens",
    "completionTokens",
    "outputTokens",
    "tokens_completion",
    "native_tokens_completion",
)
TOTAL_TOKEN_ALIASES: tuple[str, ...] = ("total_tokens", "totalTokens", "tokens_total")
COST_ALIASES: tuple[str, ...] = ("cost", "total_cost", "cost_usd", "totalCost")

# Where a usage block may live on a response payload.
USAGE_CONTAINER_ALIASES: tuple[str, ...] = ("usage", "token_usage", "usageMetadata")


@dataclass(frozen=True)
class TokenUsage:
    """Reported token counts and cost, or an explicit record of their absence."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reported_cost_usd: float | None = None
    is_byok: bool | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    upstream_inference_cost_usd: float | None = None
    source_fields: dict[str, str] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    present: bool = False

    @property
    def has_token_counts(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def has_reported_cost(self) -> bool:
        return self.reported_cost_usd is not None

    def resolved_total_tokens(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.has_token_counts:
            return int(self.prompt_tokens or 0) + int(self.completion_tokens or 0)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reported_cost_usd": self.reported_cost_usd,
            "is_byok": self.is_byok,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "upstream_inference_cost_usd": self.upstream_inference_cost_usd,
            "source_fields": dict(self.source_fields),
            "missing_fields": list(self.missing_fields),
        }


def parse_usage(response: Any) -> TokenUsage:
    """Extract a :class:`TokenUsage` from a chat-completion payload.

    Never raises on a surprising shape. Anything that cannot be read comes back
    as ``None`` and is named in ``missing_fields`` so a caller can tell
    "the provider said zero" apart from "the provider said nothing".
    """

    block = _find_usage_block(response)
    if block is None:
        return TokenUsage(
            present=False,
            missing_fields=("usage", "prompt_tokens", "completion_tokens", "cost"),
        )

    source_fields: dict[str, str] = {}
    missing: list[str] = []

    prompt = _pick_int(block, PROMPT_TOKEN_ALIASES, "prompt_tokens", source_fields, missing)
    completion = _pick_int(block, COMPLETION_TOKEN_ALIASES, "completion_tokens", source_fields, missing)
    total = _pick_int(block, TOTAL_TOKEN_ALIASES, "total_tokens", source_fields, missing)
    cost = _pick_float(block, COST_ALIASES, "cost", source_fields, missing)

    prompt_details = _as_mapping(block.get("prompt_tokens_details")) or _as_mapping(
        block.get("promptTokensDetails")
    )
    completion_details = _as_mapping(block.get("completion_tokens_details")) or _as_mapping(
        block.get("completionTokensDetails")
    )
    cost_details = _as_mapping(block.get("cost_details")) or _as_mapping(block.get("costDetails"))

    cached = _coerce_int(prompt_details.get("cached_tokens")) if prompt_details else None
    reasoning = _coerce_int(completion_details.get("reasoning_tokens")) if completion_details else None
    upstream = _coerce_float(cost_details.get("upstream_inference_cost")) if cost_details else None

    is_byok = block.get("is_byok")
    if not isinstance(is_byok, bool):
        is_byok = None

    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reported_cost_usd=cost,
        is_byok=is_byok,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        upstream_inference_cost_usd=upstream,
        source_fields=source_fields,
        missing_fields=tuple(missing),
        present=True,
    )


def _find_usage_block(response: Any) -> Mapping[str, Any] | None:
    mapping = _as_mapping(response)
    if mapping is None:
        return None
    for key in USAGE_CONTAINER_ALIASES:
        block = _as_mapping(mapping.get(key))
        if block is not None:
            return block
    # Some gateways return the generation-stats object itself, un-nested.
    if any(k in mapping for k in PROMPT_TOKEN_ALIASES + COMPLETION_TOKEN_ALIASES):
        return mapping
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _pick_int(
    block: Mapping[str, Any],
    aliases: Iterable[str],
    label: str,
    source_fields: dict[str, str],
    missing: list[str],
) -> int | None:
    for alias in aliases:
        if alias in block:
            value = _coerce_int(block.get(alias))
            if value is not None:
                source_fields[label] = alias
                return value
    missing.append(label)
    return None


def _pick_float(
    block: Mapping[str, Any],
    aliases: Iterable[str],
    label: str,
    source_fields: dict[str, str],
    missing: list[str],
) -> float | None:
    for alias in aliases:
        if alias in block:
            value = _coerce_float(block.get(alias))
            if value is not None:
                source_fields[label] = alias
                return value
    missing.append(label)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
