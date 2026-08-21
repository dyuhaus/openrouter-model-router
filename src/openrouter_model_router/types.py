"""Core data types for model routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    """Describes the work a model needs to do."""

    task_type: str = "general"
    input_tokens: int = 1000
    output_tokens: int = 1000
    modalities: tuple[str, ...] = ("text",)
    required_capabilities: tuple[str, ...] = ()
    preference: str = "balanced"
    max_cost_usd: float | None = None
    min_context_tokens: int | None = None
    allow_models: tuple[str, ...] = ()
    block_models: tuple[str, ...] = ()
    fallback_model: str | None = None

    def normalized(self) -> "TaskSpec":
        return TaskSpec(
            task_type=self.task_type.strip().lower() or "general",
            input_tokens=max(0, int(self.input_tokens)),
            output_tokens=max(0, int(self.output_tokens)),
            modalities=tuple(sorted({m.strip().lower() for m in self.modalities if m.strip()})) or ("text",),
            required_capabilities=tuple(
                sorted({c.strip().lower() for c in self.required_capabilities if c.strip()})
            ),
            preference=(self.preference.strip().lower() or "balanced"),
            max_cost_usd=self.max_cost_usd,
            min_context_tokens=self.min_context_tokens,
            allow_models=tuple(sorted({m.strip() for m in self.allow_models if m.strip()})),
            block_models=tuple(sorted({m.strip() for m in self.block_models if m.strip()})),
            fallback_model=self.fallback_model,
        )


@dataclass
class ModelInfo:
    """A routable model and the facts used to score it."""

    id: str
    name: str = ""
    provider: str = ""
    context_length: int = 0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    #: False when the provider published no usable price for this model (missing
    #: pricing block, or a negative sentinel such as OpenRouter's "-1" on the
    #: auto-router). A $0.00 estimate from an unpriced model is not "free" -- it
    #: is "unmeasured", and cost accounting must be able to tell them apart.
    pricing_known: bool = True
    capabilities: tuple[str, ...] = ("text",)
    quality_score: float = 0.5
    speed_score: float = 0.5
    reliability_score: float = 0.65
    updated_at: str = ""
    source: str = "manual"
    stats: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        if not self.provider and "/" in self.id:
            self.provider = self.id.split("/", 1)[0]
        self.capabilities = tuple(sorted({c.strip().lower() for c in self.capabilities if c.strip()}))
        self.quality_score = _clamp01(self.quality_score)
        self.speed_score = _clamp01(self.speed_score)
        self.reliability_score = _clamp01(self.reliability_score)

    def estimated_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        prompt = max(0, input_tokens) * max(0.0, self.input_cost_per_million) / 1_000_000
        completion = max(0, output_tokens) * max(0.0, self.output_cost_per_million) / 1_000_000
        return prompt + completion

    def cost_estimate(self, input_tokens: int, output_tokens: int) -> float | None:
        """Cost in USD, or None when the provider published no usable price.

        Use this anywhere a human or a report will read the number.
        ``estimated_cost_usd`` returns 0.0 for an unpriced model, which is the
        right arithmetic for scoring and the wrong thing to print: $0.00 and
        "unknown" are different facts, and only one of them is safe to budget on.
        """

        if not self.pricing_known:
            return None
        return self.estimated_cost_usd(input_tokens, output_tokens)

    @property
    def is_free(self) -> bool:
        """Genuinely $0.00 -- a published price of zero, not an absent price."""

        return self.pricing_known and self.input_cost_per_million == 0.0 and self.output_cost_per_million == 0.0

    def supports(self, capabilities: set[str]) -> bool:
        return capabilities.issubset(set(self.capabilities))

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_raw:
            data.pop("raw", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or ""),
            provider=str(data.get("provider") or ""),
            context_length=int(data.get("context_length") or 0),
            input_cost_per_million=float(data.get("input_cost_per_million") or 0.0),
            output_cost_per_million=float(data.get("output_cost_per_million") or 0.0),
            pricing_known=bool(data.get("pricing_known", True)),
            capabilities=tuple(data.get("capabilities") or ("text",)),
            quality_score=float(data.get("quality_score", 0.5)),
            speed_score=float(data.get("speed_score", 0.5)),
            reliability_score=float(data.get("reliability_score", 0.65)),
            updated_at=str(data.get("updated_at") or ""),
            source=str(data.get("source") or "manual"),
            stats=dict(data.get("stats") or {}),
            raw=dict(data.get("raw") or {}),
        )


@dataclass(frozen=True)
class Selection:
    """The result of routing a task."""

    model: ModelInfo
    score: float
    estimated_cost_usd: float
    reasons: tuple[str, ...]
    candidates_considered: int

    @property
    def model_id(self) -> str:
        return self.model.id

    @property
    def estimated_cost_is_known(self) -> bool:
        """False when the $0.00 estimate means "no price", not "no charge"."""

        return self.model.pricing_known

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model.id,
            "model": self.model.to_dict(),
            "score": self.score,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_cost_is_known": self.model.pricing_known,
            "reasons": list(self.reasons),
            "candidates_considered": self.candidates_considered,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
