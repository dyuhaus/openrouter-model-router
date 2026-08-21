"""Model catalog loading, saving, refresh, and capability inference."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .transport import HttpRequest, HttpTransport, TransportError, UrllibTransport
from .types import ModelInfo

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: How old a catalog may get before its prices stop being evidence. OpenRouter's
#: model list and prices move on a scale of days, so a week is the outer bound of
#: "this was true when it was written down".
DEFAULT_MAX_AGE_DAYS = 7.0

STALENESS_FRESH = "fresh"
STALENESS_STALE = "stale"
STALENESS_NEVER_FETCHED = "never_fetched"
STALENESS_UNPARSEABLE = "unparseable_timestamp"
STALENESS_CLOCK_SKEW = "clock_skew"
STALENESS_EMPTY = "empty_catalog"
STALENESS_UNVERIFIABLE = "unverifiable_legacy_timestamp"

#: Tolerated clock skew before a future fetch timestamp is treated as broken
#: rather than as extremely fresh. A catalog stamped next year would otherwise
#: read as fresh forever.
CLOCK_SKEW_TOLERANCE_SECONDS = 86_400.0


def default_catalog_path() -> Path:
    override = os.environ.get("OPENROUTER_MODEL_ROUTER_CATALOG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "openrouter-model-router" / "catalog.json"


class CatalogRefreshError(RuntimeError):
    """Raised when the OpenRouter catalog cannot be fetched or parsed."""


class CatalogLoadError(RuntimeError):
    """Raised when an on-disk catalog exists but cannot be read.

    Distinct from "no catalog" on purpose. A truncated or hand-edited catalog
    must not fall through to the unpriced bootstrap, because that failure looks
    identical to a healthy run right up until every cost estimate is $0.00.
    """


class ModelCatalog:
    """A mutable collection of routable model metadata."""

    def __init__(
        self,
        models: Iterable[ModelInfo] | None = None,
        updated_at: str | None = None,
        fetched_at: str | None = None,
        fetched_at_is_derived: bool = False,
    ) -> None:
        self.models: dict[str, ModelInfo] = {}
        #: When this catalog's contents last came from upstream. A property of
        #: the FETCH, not of the load: nothing in this class may set it except a
        #: refresh, or a load reading it back off disk. ``None`` means "never
        #: fetched", which is not the same as "fetched a long time ago" and is
        #: never treated as fresh.
        self.fetched_at = fetched_at
        #: True when fetched_at was inferred from a schema_version 1 catalog's
        #: updated_at rather than recorded at a fetch. That value was restamped
        #: by every load and every local edit, so it is an UPPER BOUND on the
        #: real fetch time: such a catalog can only ever look fresher than it is,
        #: and must not be able to report fresh.
        self.fetched_at_is_derived = bool(fetched_at_is_derived)
        for model in models or []:
            self.upsert(model)
        # AFTER the loop, on purpose. upsert() stamps updated_at, so assigning
        # first let every construction - including load() - overwrite the
        # caller's timestamp with "now". That is how a 2019 catalog reported as
        # fetched today, and how one load-and-save cycle destroyed the real date.
        self.updated_at = updated_at or _utc_now()

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
        # Local modification time only. fetched_at is deliberately untouched:
        # recording an outcome or hand-adding a model does not make the prices
        # any newer than the fetch that produced them.
        self.updated_at = _utc_now()

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "updated_at": self.updated_at,
            "fetched_at": self.fetched_at,
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
        try:
            data = json.loads(target.read_text())
            if not isinstance(data, dict):
                raise CatalogLoadError(f"{target}: expected a JSON object, got {type(data).__name__}")
            models = [ModelInfo.from_dict(item) for item in data.get("models", [])]
        except CatalogLoadError:
            raise
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            raise CatalogLoadError(
                f"{target} exists but could not be read ({type(exc).__name__}: {exc}). "
                "Refusing to fall back to the unpriced bootstrap catalog - "
                "run `openrouter-model-router refresh` to rewrite it."
            ) from exc
        # schema_version 1 catalogs have no fetched_at key at all. Fall back to
        # their only timestamp rather than to None, so an old v1 catalog is still
        # judged stale rather than excused as "never fetched, cannot say".
        #
        # Keyed on the key being ABSENT, not on it being falsy. A v2 catalog that
        # records `"fetched_at": null` is saying "never fetched", and reading
        # updated_at in its place would restore the exact laundering this fix
        # removes: updated_at is restamped by any local edit, so an unfetched
        # catalog would come back from disk looking brand new.
        is_legacy = "fetched_at" not in data
        fetched_at = data.get("updated_at") if is_legacy else data["fetched_at"]
        return cls(
            models=models,
            updated_at=data.get("updated_at"),
            fetched_at=fetched_at,
            fetched_at_is_derived=is_legacy and fetched_at is not None,
        )

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
            # fetched_at stays None: the bootstrap entry was never fetched from
            # anywhere, so it must never satisfy a freshness check.
            fetched_at=None,
        )

    @classmethod
    def from_openrouter_payload(cls, payload: dict[str, Any], fetched_at: str | None = None) -> "ModelCatalog":
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise CatalogRefreshError("OpenRouter response did not contain a data list")
        return cls(
            (model_from_openrouter(item) for item in rows if isinstance(item, dict)),
            fetched_at=fetched_at or _utc_now(),
        )

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
        # Stamped here, at the fetch, which is the only moment that knows when
        # these prices were true.
        return cls.from_openrouter_payload(payload, fetched_at=_utc_now())

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

    def age_seconds(self, now: float | None = None) -> float | None:
        """Seconds since the last upstream fetch, or None if that is unknowable."""

        fetched = _parse_utc(self.fetched_at)
        if fetched is None:
            return None
        return (time.time() if now is None else now) - fetched

    def staleness(self, max_age_days: float = DEFAULT_MAX_AGE_DAYS, now: float | None = None) -> dict[str, Any]:
        """Is this catalog still evidence?

        Only ``fresh`` passes. Every other answer - never fetched, unparseable
        timestamp, a future timestamp, an empty catalog, or simply old - is a
        failure, because none of them establish that the prices in here are
        current. "I cannot tell how old this is" must never be scored as new.
        """

        max_age_seconds = float(max_age_days) * 86_400.0
        age = self.age_seconds(now=now)
        result: dict[str, Any] = {
            "fetched_at": self.fetched_at,
            "updated_at": self.updated_at,
            "max_age_days": float(max_age_days),
            "models": len(self.models),
            "age_seconds": None if age is None else round(age, 3),
            "age_days": None if age is None else round(age / 86_400.0, 4),
            "fetched_at_is_derived": self.fetched_at_is_derived,
        }

        if not self.models:
            # A catalog holding nothing can price nothing. Calling that "fresh"
            # is a gate passing over zero items.
            status, reason = STALENESS_EMPTY, "catalog contains 0 models - it can price nothing"
        elif self.fetched_at is None:
            status = STALENESS_NEVER_FETCHED
            reason = "no fetch timestamp: this catalog was never refreshed from OpenRouter"
        elif age is None:
            status = STALENESS_UNPARSEABLE
            reason = f"fetch timestamp {self.fetched_at!r} could not be parsed, so its age is unknown"
        elif age < -CLOCK_SKEW_TOLERANCE_SECONDS:
            status = STALENESS_CLOCK_SKEW
            reason = (
                f"fetch timestamp {self.fetched_at} is {abs(age) / 86_400.0:.1f} days in the future - "
                "a clock is wrong and the age cannot be trusted"
            )
        elif age > max_age_seconds:
            # Sound even for a derived timestamp: updated_at is always at or
            # after the real fetch, so "older than the limit" can only understate.
            status = STALENESS_STALE
            reason = (
                f"catalog was fetched {age / 86_400.0:.1f} days ago (limit {max_age_days:g}) - "
                "prices may have changed; run `openrouter-model-router refresh`"
            )
        elif self.fetched_at_is_derived:
            # Inside the limit, but the only timestamp available was restamped on
            # every load by the version that wrote this file. It is an upper
            # bound, so "recent" here is not evidence of anything. One refresh
            # replaces it with a real fetch time.
            status = STALENESS_UNVERIFIABLE
            reason = (
                f"this is a schema_version 1 catalog with no fetch timestamp; its updated_at "
                f"({self.updated_at}) was restamped on every load, so an apparent age of "
                f"{age / 86_400.0:.1f} days is an upper bound and proves nothing - "
                "run `openrouter-model-router refresh` once to record a real fetch time"
            )
        else:
            status = STALENESS_FRESH
            reason = f"catalog fetched {max(0.0, age) / 86_400.0:.1f} days ago (limit {max_age_days:g})"

        result["status"] = status
        result["fresh"] = status == STALENESS_FRESH
        result["stale"] = status != STALENESS_FRESH
        result["reason"] = reason
        return result

    def is_stale(self, max_age_days: float = DEFAULT_MAX_AGE_DAYS, now: float | None = None) -> bool:
        return bool(self.staleness(max_age_days=max_age_days, now=now)["stale"])

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
        # The merged catalog is only as fresh as the newest fetch that fed it.
        # max(), not "take the incoming one": merging an OLD catalog in must not
        # be able to make this one look newer, and must not silently age it either.
        mine, theirs = _parse_utc(self.fetched_at), _parse_utc(other.fetched_at)
        if theirs is not None and (mine is None or theirs > mine):
            self.fetched_at = other.fetched_at
            self.fetched_at_is_derived = other.fetched_at_is_derived
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


def _parse_utc(value: Any) -> float | None:
    """Parse a catalog timestamp into a POSIX time, or None if it is unusable.

    Returns None rather than a default: an unreadable timestamp is an unknown
    age, and an unknown age must not be silently scored as zero (brand new).
    """

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _ema(previous: float, observation: float, alpha: float = 0.25) -> float:
    return max(0.0, min(1.0, (previous * (1 - alpha)) + (observation * alpha)))


def _running_average(previous: float, observation: float, count: int) -> float:
    if count <= 1:
        return float(observation)
    return previous + ((float(observation) - previous) / count)


def _weighted_refresh(existing: float, incoming: float) -> float:
    return max(0.0, min(1.0, (existing * 0.75) + (incoming * 0.25)))
