"""Model catalog loading, saving, refresh, and capability inference."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .transport import HttpRequest, HttpTransport, TransportError, UrllibTransport
from .types import ModelInfo

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def default_catalog_path() -> Path:
    override = os.environ.get("OPENROUTER_MODEL_ROUTER_CATALOG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "openrouter-model-router" / "catalog.json"


class CatalogRefreshError(RuntimeError):
    """Raised when the OpenRouter catalog cannot be fetched or parsed."""


class ModelCatalog:
    """A mutable collection of routable model metadata."""

    def __init__(self, models: Iterable[ModelInfo] | None = None, updated_at: str | None = None) -> None:
        self.models: dict[str, ModelInfo] = {}
        self.updated_at = updated_at or _utc_now()
        for model in models or []:
            self.upsert(model)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        return iter(self.models.values())

    def get(self, model_id: str) -> ModelInfo | None:
        return self.models.get(model_id)

    def upsert(self, model: ModelInfo) -> None:
        if not model.id:
            raise ValueError("model id is required")
        self.models[model.id] = model
        self.updated_at = _utc_now()

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "updated_at": self.updated_at,
            "models": [m.to_dict(include_raw=include_raw) for m in sorted(self.models.values(), key=lambda x: x.id)],
        }

    def save(self, path: str | os.PathLike[str] | None = None, include_raw: bool = False) -> Path:
        target = Path(path).expanduser() if path else default_catalog_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(include_raw=include_raw), indent=2, sort_keys=True) + "\n")
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, bootstrap: bool = True) -> "ModelCatalog":
        target = Path(path).expanduser() if path else default_catalog_path()
        if not target.exists():
            return cls.bootstrap() if bootstrap else cls()
        data = json.loads(target.read_text())
        models = [ModelInfo.from_dict(item) for item in data.get("models", [])]
        return cls(models=models, updated_at=data.get("updated_at"))

    @classmethod
    def bootstrap(cls) -> "ModelCatalog":
        """Return a minimal deterministic fallback catalog.

        These entries are intentionally generic. Run `refresh` to replace them
        with current OpenRouter metadata before relying on optimized routing.
        """

        now = _utc_now()
        return cls(
            models=[
                ModelInfo(
                    id="openrouter/auto",
                    name="OpenRouter Auto",
                    provider="openrouter",
                    context_length=128_000,
                    input_cost_per_million=0.0,
                    output_cost_per_million=0.0,
                    # NOT free -- unpriced. openrouter/auto's real price depends
                    # on whichever model it routes to, so any estimate built on
                    # this fallback is $0.00 because nothing was measured.
                    pricing_known=False,
                    capabilities=("text", "tool_use", "json_mode"),
                    quality_score=0.55,
                    speed_score=0.55,
                    reliability_score=0.65,
                    updated_at=now,
                    source="bootstrap",
                )
            ],
            updated_at=now,
        )

    @classmethod
    def from_openrouter_payload(cls, payload: dict[str, Any]) -> "ModelCatalog":
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise CatalogRefreshError("OpenRouter response did not contain a data list")
        return cls(model_from_openrouter(item) for item in rows if isinstance(item, dict))

    @classmethod
    def refresh_from_openrouter(
        cls,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: str | None = None,
        timeout: float = 20.0,
        transport: HttpTransport | None = None,
    ) -> "ModelCatalog":
        """Fetch ``GET /models`` and build a catalog from it.

        This endpoint is PUBLIC: verified 2026-08-21 returning HTTP 200 with 420
        models and no credential. ``api_key`` is optional and only forwarded when
        present, so a machine with no key can still populate real prices.
        """

        url = f"{base_url.rstrip('/')}/models"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        sender = transport or UrllibTransport()
        try:
            response = sender.send(HttpRequest(method="GET", url=url, headers=headers, timeout=timeout))
        except TransportError as exc:
            raise CatalogRefreshError(f"failed to refresh OpenRouter model catalog: {exc}") from exc

        if response.status >= 400:
            detail = response.body.decode("utf-8", errors="replace")[:500]
            raise CatalogRefreshError(
                f"failed to refresh OpenRouter model catalog: HTTP {response.status}: {detail}"
            )
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CatalogRefreshError(f"failed to refresh OpenRouter model catalog: {exc}") from exc
        return cls.from_openrouter_payload(payload)

    def pricing_coverage(self) -> dict[str, Any]:
        """How much of this catalog can actually be costed.

        A catalog that loads is not a catalog that prices. This is the number to
        print after a refresh: models whose price is unknown will estimate $0.00
        and quietly understate spend.
        """

        models = list(self.models.values())
        priced = [m for m in models if m.pricing_known and (m.input_cost_per_million or m.output_cost_per_million)]
        free = [m for m in models if m.is_free]
        unknown = [m for m in models if not m.pricing_known]
        return {
            "total": len(models),
            "priced": len(priced),
            "free": len(free),
            "pricing_unknown": len(unknown),
            "pricing_unknown_ids": sorted(m.id for m in unknown)[:25],
        }

    def merge(self, other: "ModelCatalog", preserve_observations: bool = True) -> tuple[int, int]:
        """Merge another catalog into this one.

        Returns `(added, updated)`.
        """

        added = 0
        updated = 0
        for incoming in other:
            existing = self.get(incoming.id)
            if existing is None:
                added += 1
                self.upsert(incoming)
                continue
            if preserve_observations:
                incoming.stats = existing.stats
                incoming.quality_score = _weighted_refresh(existing.quality_score, incoming.quality_score)
                incoming.speed_score = _weighted_refresh(existing.speed_score, incoming.speed_score)
                incoming.reliability_score = _weighted_refresh(existing.reliability_score, incoming.reliability_score)
            updated += 1
            self.upsert(incoming)
        return added, updated

    def record_outcome(
        self,
        model_id: str,
        *,
        success: bool,
        latency_ms: float | None = None,
        quality_score: float | None = None,
    ) -> ModelInfo:
        model = self.get(model_id)
        if model is None:
            raise KeyError(f"unknown model: {model_id}")

        stats = dict(model.stats)
        runs = int(stats.get("runs") or 0) + 1
        successes = int(stats.get("successes") or 0) + (1 if success else 0)
        stats["runs"] = runs
        stats["successes"] = successes
        stats["success_rate"] = successes / runs

        reliability = _ema(model.reliability_score, 1.0 if success else 0.0)
        speed = model.speed_score
        if latency_ms is not None and latency_ms > 0:
            stats["avg_latency_ms"] = _running_average(float(stats.get("avg_latency_ms") or 0.0), latency_ms, runs)
            speed_observation = 1.0 / (1.0 + (latency_ms / 10_000.0))
            speed = _ema(model.speed_score, speed_observation)

        quality = model.quality_score
        if quality_score is not None:
            quality = _ema(model.quality_score, max(0.0, min(1.0, quality_score)))
            stats["avg_quality_score"] = _running_average(float(stats.get("avg_quality_score") or 0.0), quality_score, runs)

        updated = ModelInfo(
            **{
                **model.to_dict(include_raw=True),
                "quality_score": quality,
                "speed_score": speed,
                "reliability_score": reliability,
                "stats": stats,
                "updated_at": _utc_now(),
            }
        )
        self.upsert(updated)
        return updated


def model_from_openrouter(raw: dict[str, Any]) -> ModelInfo:
    model_id = str(raw.get("id") or "").strip()
    if not model_id:
        raise CatalogRefreshError("OpenRouter model record is missing id")

    pricing = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
    prompt_cost = _price_per_million(pricing.get("prompt"))
    completion_cost = _price_per_million(pricing.get("completion"))
    pricing_known = _pricing_is_known(raw.get("pricing"))
    context_length = int(
        raw.get("context_length")
        or (raw.get("top_provider") or {}).get("context_length")
        or 0
    )
    capabilities = infer_capabilities(raw, prompt_cost, completion_cost, context_length)
    quality = infer_quality_score(model_id, capabilities)
    speed = infer_speed_score(model_id, capabilities)
    provider = model_id.split("/", 1)[0] if "/" in model_id else ""

    return ModelInfo(
        id=model_id,
        name=str(raw.get("name") or model_id),
        provider=provider,
        context_length=context_length,
        input_cost_per_million=prompt_cost,
        output_cost_per_million=completion_cost,
        pricing_known=pricing_known,
        capabilities=tuple(sorted(capabilities)),
        quality_score=quality,
        speed_score=speed,
        reliability_score=0.65,
        updated_at=_utc_now(),
        source="openrouter:/models",
        raw=raw,
    )


def infer_capabilities(
    raw: dict[str, Any],
    prompt_cost: float,
    completion_cost: float,
    context_length: int,
) -> set[str]:
    model_id = str(raw.get("id") or "").lower()
    name = str(raw.get("name") or "").lower()
    description = str(raw.get("description") or "").lower()
    combined = f"{model_id} {name} {description}"
    architecture = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
    supported_parameters = {str(p).lower() for p in raw.get("supported_parameters") or []}

    capabilities = {"text"}
    modalities = {
        str(m).lower()
        for m in (
            list(architecture.get("input_modalities") or [])
            + list(architecture.get("output_modalities") or [])
        )
    }
    modality = str(architecture.get("modality") or "").lower()

    if "image" in modalities or "vision" in combined or "image" in modality:
        capabilities.add("vision")
    if "audio" in modalities or "audio" in modality:
        capabilities.add("audio")
    if {"tools", "tool_choice", "parallel_tool_calls"} & supported_parameters:
        capabilities.add("tool_use")
    if {"response_format", "structured_outputs"} & supported_parameters:
        capabilities.add("json_mode")
    if context_length >= 128_000:
        capabilities.add("long_context")
    if _pricing_is_known(raw.get("pricing")) and prompt_cost + completion_cost <= 1.0:
        capabilities.add("cheap")

    if any(token in combined for token in ("code", "coder", "codestral", "devstral", "programming")):
        capabilities.add("coding")
    if any(token in combined for token in ("reason", "thinking", "r1", "o1", "o3", "math", "logic")):
        capabilities.add("reasoning")
    if any(token in combined for token in ("flash", "mini", "haiku", "lite", "turbo", "fast")):
        capabilities.add("fast")

    return capabilities


def infer_quality_score(model_id: str, capabilities: set[str]) -> float:
    text = model_id.lower()
    score = 0.52
    if "reasoning" in capabilities:
        score += 0.14
    if "coding" in capabilities:
        score += 0.08
    if "long_context" in capabilities:
        score += 0.04
    if any(token in text for token in ("opus", "pro", "max", "gpt-5", "sonnet", "ultra")):
        score += 0.20
    if any(token in text for token in ("mini", "nano", "flash", "haiku", "lite", "small")):
        score -= 0.07
    if ":free" in text:
        score -= 0.18
    return max(0.05, min(0.98, score))


def infer_speed_score(model_id: str, capabilities: set[str]) -> float:
    text = model_id.lower()
    score = 0.55
    if "fast" in capabilities:
        score += 0.22
    if any(token in text for token in ("mini", "nano", "flash", "haiku", "lite", "turbo")):
        score += 0.12
    if any(token in text for token in ("opus", "pro", "max", "ultra", "thinking")):
        score -= 0.15
    return max(0.05, min(0.98, score))


def _pricing_is_known(pricing: Any) -> bool:
    """True only when the provider published a real, non-sentinel price.

    OpenRouter publishes "-1" for prompt/completion on its meta-routers
    (openrouter/auto and friends) because the real price depends on whichever
    model the router picks. `_price_per_million` clamps that to 0.0, which is
    indistinguishable from free -- and a free-looking model wins every
    cost-weighted comparison while contributing $0.00 to the spend estimate.
    Verified 2026-08-21: 5 of 420 live models carry the -1 sentinel.
    """

    if not isinstance(pricing, dict):
        return False
    prompt = pricing.get("prompt")
    completion = pricing.get("completion")
    if prompt is None and completion is None:
        return False
    for value in (prompt, completion):
        if value is None:
            continue
        try:
            if float(value) < 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _price_per_million(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0) * 1_000_000)
    except (TypeError, ValueError):
        return 0.0


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ema(previous: float, observation: float, alpha: float = 0.25) -> float:
    return max(0.0, min(1.0, (previous * (1 - alpha)) + (observation * alpha)))


def _running_average(previous: float, observation: float, count: int) -> float:
    if count <= 1:
        return float(observation)
    return previous + ((float(observation) - previous) / count)


def _weighted_refresh(existing: float, incoming: float) -> float:
    return max(0.0, min(1.0, (existing * 0.75) + (incoming * 0.25)))
