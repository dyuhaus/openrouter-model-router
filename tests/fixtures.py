"""Shared fixtures: a recorded-shape OpenRouter response and a test catalog.

The usage block below matches the shape documented at
https://openrouter.ai/docs/use-cases/usage-accounting (read 2026-08-21), which
states that full usage detail is now returned automatically on every response
and that `usage: {include: true}` / `stream_options` are deprecated no-ops.
"""

from __future__ import annotations

from openrouter_model_router import ModelCatalog, ModelInfo

REAL_SHAPE_RESPONSE = {
    "id": "gen-1787311563-abc123",
    "object": "chat.completion",
    "created": 1787311563,
    "model": "openai/gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Espresso is 25-30 seconds."},
        }
    ],
    "usage": {
        "prompt_tokens": 194,
        "completion_tokens": 88,
        "total_tokens": 282,
        "cost": 0.00007830,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
        "cost_details": {
            "upstream_inference_cost": 0.0,
            "upstream_inference_prompt_cost": 0.0000291,
            "upstream_inference_completions_cost": 0.0000528,
        },
    },
}

#: Some OpenAI-compatible gateways spell the same numbers differently.
ALIAS_SHAPE_RESPONSE = {
    "model": "some/gateway-model",
    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"input_tokens": 100, "output_tokens": 50, "total_cost": 0.0025},
}

#: A legal response with no usage block at all (documented as optional, and the
#: normal case when a streaming reader stops before the final SSE message).
NO_USAGE_RESPONSE = {
    "model": "some/quiet-model",
    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
}


def test_catalog() -> ModelCatalog:
    return ModelCatalog(
        [
            ModelInfo(
                id="openai/gpt-4.1-mini",
                context_length=1_000_000,
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
                capabilities=("text", "cheap", "fast", "json_mode", "long_context", "tool_use"),
                quality_score=0.6,
            ),
            ModelInfo(
                id="meta/unpriced-router",
                context_length=1_000_000,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                pricing_known=False,
                capabilities=("text", "long_context"),
                quality_score=0.5,
            ),
        ]
    )
