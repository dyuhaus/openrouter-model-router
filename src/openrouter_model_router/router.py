"""Task-aware model selection, instrumented for cost."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .catalog import ModelCatalog
from .ledger import STATUS_COMPLETED, STATUS_ERROR, STATUS_GATE_FAILED, RunLedger, RunRecord
from .openrouter import ChatResult, OpenRouterClient, OpenRouterError
from .types import ModelInfo, Selection, TaskSpec

#: A gate takes the completion and returns either a bool, or the list of reasons
#: it rejected the output (empty list == passed).
Gate = Callable[[ChatResult], "bool | Iterable[str]"]

UNSELECTED_MODEL = "<no-model-selected>"


@dataclass
class RunOutcome:
    """Everything one instrumented model call produced, including its cost."""

    record: RunRecord
    selection: Selection | None = None
    result: ChatResult | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.record.succeeded

    @property
    def content(self) -> str:
        return self.result.content if self.result else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "selection": self.selection.to_dict() if self.selection else None,
            "result": self.result.to_dict() if self.result else None,
            "error": str(self.error) if self.error else None,
        }


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
        ledger: RunLedger | None = None,
        task_label: str = "",
        gate: Gate | None = None,
        attempt: int = 1,
        **kwargs,
    ) -> dict:
        """Select a model, call it, and return content plus what it cost.

        Backwards compatible: ``selection`` and ``response`` are still present.
        ``usage``, ``content``, ``estimated_cost_usd``, ``reported_cost_usd`` and
        ``record`` are new.
        """

        outcome = self.run(
            client=client,
            messages=messages,
            task=task,
            ledger=ledger,
            task_label=task_label,
            gate=gate,
            attempt=attempt,
            **kwargs,
        )
        if outcome.error is not None:
            raise outcome.error
        assert outcome.result is not None  # guaranteed when error is None
        return {
            "selection": outcome.selection,
            "response": outcome.result.response,
            "content": outcome.result.content,
            "usage": outcome.result.usage,
            "estimated_cost_usd": outcome.record.estimated_cost_usd,
            "reported_cost_usd": outcome.record.reported_cost_usd,
            "record": outcome.record,
            "outcome": outcome,
        }

    def run(
        self,
        *,
        client: OpenRouterClient,
        messages: list[dict],
        task: TaskSpec | None = None,
        ledger: RunLedger | None = None,
        task_label: str = "",
        gate: Gate | None = None,
        attempt: int = 1,
        **kwargs,
    ) -> RunOutcome:
        """Instrumented call. Records a ledger row for EVERY outcome.

        A run that errors, and a run whose output the gates reject, are both
        written to the ledger. That is deliberate: the retry multiplier is
        attempts-over-accepted-outputs, so a ledger that drops failures can only
        ever report 1.0 and the cost model stays a guess.
        """

        spec = (task or TaskSpec()).normalized()
        selection: Selection | None = None
        try:
            selection = self.select(spec)
        except ValueError as exc:
            return self._finish(
                ledger=ledger,
                record=RunRecord(
                    model=UNSELECTED_MODEL,
                    task_label=task_label,
                    status=STATUS_ERROR,
                    attempt=attempt,
                    estimated_input_tokens=spec.input_tokens,
                    estimated_output_tokens=spec.output_tokens,
                    error=f"{type(exc).__name__}: {exc}",
                    catalog_updated_at=self.catalog.updated_at,
                ),
                error=exc,
            )

        base = {
            "model": selection.model.id,
            "task_label": task_label,
            "attempt": attempt,
            "estimated_input_tokens": spec.input_tokens,
            "estimated_output_tokens": spec.output_tokens,
            "estimated_cost_usd": selection.estimated_cost_usd if selection.estimated_cost_is_known else None,
            "catalog_updated_at": self.catalog.updated_at,
        }

        try:
            result = client.chat(selection.model.id, messages, **kwargs)
        except OpenRouterError as exc:
            return self._finish(
                ledger=ledger,
                record=RunRecord(status=STATUS_ERROR, error=f"{type(exc).__name__}: {exc}", **base),
                selection=selection,
                error=exc,
            )

        failures = _normalize_gate(gate, result)
        record = RunRecord(
            status=STATUS_COMPLETED if not failures else STATUS_GATE_FAILED,
            gates_passed=not failures,
            gate_failures=failures,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            reported_cost_usd=result.usage.reported_cost_usd,
            latency_ms=result.latency_ms,
            usage_present=result.usage.present,
            usage_source_fields=dict(result.usage.source_fields),
            **base,
        )
        return self._finish(ledger=ledger, record=record, selection=selection, result=result)

    @staticmethod
    def _finish(
        *,
        ledger: RunLedger | None,
        record: RunRecord,
        selection: Selection | None = None,
        result: ChatResult | None = None,
        error: Exception | None = None,
    ) -> RunOutcome:
        if ledger is not None:
            ledger.append(record)
        return RunOutcome(record=record, selection=selection, result=result, error=error)

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


def _normalize_gate(gate: Gate | None, result: ChatResult) -> tuple[str, ...]:
    """Turn a gate's answer into a tuple of failure reasons (empty == passed).

    A gate that itself explodes is a FAILED gate, not a passed one. Treating an
    exception as "no failures reported" is exactly how a validator ends up
    indistinguishable from no validator at all.
    """

    if gate is None:
        return ()
    try:
        verdict = gate(result)
    except Exception as exc:  # noqa: BLE001 - a broken gate must not read as a pass
        return (f"gate raised {type(exc).__name__}: {exc}",)
    if verdict is True:
        return ()
    if verdict is False:
        return ("gate returned False",)
    if verdict is None:
        return ("gate returned None (no verdict)",)
    if isinstance(verdict, str):
        return (verdict,)
    try:
        return tuple(str(item) for item in verdict)
    except TypeError:
        return (f"gate returned an uninterpretable verdict: {verdict!r}",)


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
