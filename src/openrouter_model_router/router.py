"""Task-aware model selection."""

from __future__ import annotations

from pathlib import Path

from .catalog import ModelCatalog
from .openrouter import OpenRouterClient
from .types import ModelInfo, Selection, TaskSpec


class ModelRouter:
    """Selects an OpenRouter model for a task."""

    def __init__(self, catalog: ModelCatalog | None = None) -> None:
        self.catalog = catalog or ModelCatalog.bootstrap()

    @classmethod
    def from_file(cls, path: str | Path | None = None, bootstrap: bool = True) -> "ModelRouter":
        return cls(ModelCatalog.load(path=path, bootstrap=bootstrap))

    def select(self, task: TaskSpec | None = None) -> Selection:
        spec = (task or TaskSpec()).normalized()
        candidates: list[tuple[ModelInfo, float, float, tuple[str, ...]]] = []
        for model in self.catalog:
            compatible, reasons = self._compatible(model, spec)
            if not compatible:
                continue
            estimated_cost = model.estimated_cost_usd(spec.input_tokens, spec.output_tokens)
            candidates.append((model, estimated_cost, 0.0, reasons))

        if not candidates and spec.fallback_model:
            fallback = self.catalog.get(spec.fallback_model)
            if fallback:
                estimated_cost = fallback.estimated_cost_usd(spec.input_tokens, spec.output_tokens)
                return Selection(
                    model=fallback,
                    score=0.0,
                    estimated_cost_usd=estimated_cost,
                    reasons=("fallback_model used after no compatible candidates matched",),
                    candidates_considered=0,
                )

        if not candidates:
            raise ValueError("no compatible model found for task")

        max_cost = max(cost for _, cost, _, _ in candidates) or 1.0
        weights = _preference_weights(spec.preference)
        scored: list[Selection] = []
        for model, estimated_cost, _, reasons in candidates:
            capability_bonus, capability_reasons = _capability_bonus(model, spec)
            cost_score = 1.0 - min(1.0, estimated_cost / max_cost)
            quality = min(1.0, model.quality_score + capability_bonus)
            score = (
                (quality * weights["quality"])
                + (model.speed_score * weights["speed"])
                + (cost_score * weights["cost"])
                + (model.reliability_score * weights["reliability"])
            )
            scored.append(
                Selection(
                    model=model,
                    score=round(score, 6),
                    estimated_cost_usd=estimated_cost,
                    reasons=tuple(reasons + capability_reasons + [f"preference={spec.preference}"]),
                    candidates_considered=len(candidates),
                )
            )

        return max(scored, key=lambda selection: (selection.score, -selection.estimated_cost_usd, selection.model.id))

    def chat_completion(
        self,
        *,
        client: OpenRouterClient,
        messages: list[dict],
        task: TaskSpec | None = None,
        **kwargs,
    ) -> dict:
        selection = self.select(task)
        response = client.chat_completion(selection.model.id, messages, **kwargs)
        return {"selection": selection, "response": response}

    def _compatible(self, model: ModelInfo, spec: TaskSpec) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if spec.allow_models and model.id not in set(spec.allow_models):
            return False, []
        if model.id in set(spec.block_models):
            return False, []

        needed_context = spec.min_context_tokens or (spec.input_tokens + spec.output_tokens)
        if model.context_length and model.context_length < needed_context:
            return False, []
        if model.context_length:
            reasons.append(f"context>={needed_context}")

        required = set(spec.required_capabilities)
        required.update(_capabilities_for_modalities(spec.modalities))
        if not model.supports(required):
            return False, []
        if required:
            reasons.append("required_capabilities=" + ",".join(sorted(required)))

        estimated_cost = model.estimated_cost_usd(spec.input_tokens, spec.output_tokens)
        if spec.max_cost_usd is not None and estimated_cost > spec.max_cost_usd:
            return False, []
        if spec.max_cost_usd is not None:
            reasons.append(f"cost<={spec.max_cost_usd}")

        return True, reasons


def _capabilities_for_modalities(modalities: tuple[str, ...]) -> set[str]:
    capabilities = set()
    if "image" in modalities or "vision" in modalities:
        capabilities.add("vision")
    if "audio" in modalities:
        capabilities.add("audio")
    return capabilities


def _capability_bonus(model: ModelInfo, spec: TaskSpec) -> tuple[float, list[str]]:
    capabilities = set(model.capabilities)
    text = spec.task_type.lower()
    bonus = 0.0
    reasons: list[str] = []
    if any(token in text for token in ("code", "coding", "program", "debug", "software")) and "coding" in capabilities:
        bonus += 0.09
        reasons.append("coding_match")
    if any(token in text for token in ("reason", "analysis", "math", "plan", "strategy")) and "reasoning" in capabilities:
        bonus += 0.09
        reasons.append("reasoning_match")
    if any(token in text for token in ("extract", "json", "schema", "structured")) and "json_mode" in capabilities:
        bonus += 0.06
        reasons.append("json_match")
    if ("image" in spec.modalities or "vision" in spec.modalities) and "vision" in capabilities:
        bonus += 0.08
        reasons.append("vision_match")
    return min(0.2, bonus), reasons


def _preference_weights(preference: str) -> dict[str, float]:
    table = {
        "cheap": {"quality": 0.22, "speed": 0.13, "cost": 0.55, "reliability": 0.10},
        "cost": {"quality": 0.22, "speed": 0.13, "cost": 0.55, "reliability": 0.10},
        "quality": {"quality": 0.60, "speed": 0.12, "cost": 0.13, "reliability": 0.15},
        "best": {"quality": 0.60, "speed": 0.12, "cost": 0.13, "reliability": 0.15},
        "fast": {"quality": 0.24, "speed": 0.50, "cost": 0.16, "reliability": 0.10},
        "balanced": {"quality": 0.42, "speed": 0.22, "cost": 0.24, "reliability": 0.12},
    }
    return table.get(preference, table["balanced"])
