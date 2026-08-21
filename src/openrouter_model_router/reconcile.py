"""Reconcile estimated cost against provider-reported cost.

This is the check that catches a stale price catalog. The router estimates from
catalog prices captured at refresh time; the provider reports what it actually
charged. When those two diverge past a tolerance, something on OUR side is wrong and
every downstream cost-per-course number is wrong with it. Two causes produce the
same signal and both matter:

* the catalog price is stale -- the provider repriced and nobody refreshed;
* the ``TaskSpec`` token counts are wrong -- the estimate was built from guessed
  input/output sizes that do not match what was actually sent.

The report tells the two apart itself, by comparing the ledger's
``estimated_input_tokens``/``estimated_output_tokens`` against the reported
``prompt_tokens``/``completion_tokens``: if the token counts agree and the cost
does not, the price is stale; if the token counts disagree too, the estimate was
built on the wrong sizes. Blaming the catalog for a token-estimate error would
send someone to refresh a catalog that is already correct.

Drift is reported per model and overall. A run that carries only one of the two
numbers is *excluded from the drift maths and counted separately*, because
silently treating a missing reported cost as $0.00 would manufacture a 100%
drift, and treating it as "matching" would hide a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .ledger import RunRecord

DEFAULT_TOLERANCE = 0.10  # 10% relative drift before a model is flagged.
DEFAULT_ABSOLUTE_FLOOR_USD = 0.0005  # Ignore drift on runs too cheap to matter.
TOKEN_TOLERANCE = 0.10  # Token drift beyond this means the estimate's sizes were wrong.

CAUSE_STALE_PRICE = "stale_catalog_price"
CAUSE_TOKEN_ESTIMATE = "wrong_token_estimate"
CAUSE_NO_PRICE = "no_catalog_price"

STATUS_OK = "ok"
STATUS_DRIFT = "drift"
STATUS_INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ModelDrift:
    model: str
    comparable_runs: int = 0
    estimated_cost_usd: float = 0.0
    reported_cost_usd: float = 0.0
    missing_reported: int = 0
    missing_estimate: int = 0
    estimated_tokens: int = 0
    reported_tokens: int = 0
    token_comparable_runs: int = 0
    flagged: bool = False
    cause: str | None = None
    reason: str = ""

    @property
    def token_drift(self) -> float | None:
        """Relative difference between estimated and actually-used tokens."""

        if self.token_comparable_runs == 0 or self.estimated_tokens <= 0:
            return None
        return round((self.reported_tokens - self.estimated_tokens) / self.estimated_tokens, 6)

    @property
    def absolute_drift_usd(self) -> float:
        return round(self.reported_cost_usd - self.estimated_cost_usd, 8)

    @property
    def relative_drift(self) -> float | None:
        if self.comparable_runs == 0 or self.estimated_cost_usd <= 0:
            return None
        return round((self.reported_cost_usd - self.estimated_cost_usd) / self.estimated_cost_usd, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "comparable_runs": self.comparable_runs,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "reported_cost_usd": round(self.reported_cost_usd, 8),
            "absolute_drift_usd": self.absolute_drift_usd,
            "relative_drift": self.relative_drift,
            "missing_reported": self.missing_reported,
            "missing_estimate": self.missing_estimate,
            "estimated_tokens": self.estimated_tokens,
            "reported_tokens": self.reported_tokens,
            "token_drift": self.token_drift,
            "flagged": self.flagged,
            "cause": self.cause,
            "reason": self.reason,
        }


@dataclass
class ReconciliationReport:
    tolerance: float = DEFAULT_TOLERANCE
    status: str = STATUS_INSUFFICIENT_DATA
    total_runs: int = 0
    comparable_runs: int = 0
    estimated_cost_usd: float = 0.0
    reported_cost_usd: float = 0.0
    missing_reported: int = 0
    missing_estimate: int = 0
    models: list[ModelDrift] = field(default_factory=list)

    @property
    def absolute_drift_usd(self) -> float:
        return round(self.reported_cost_usd - self.estimated_cost_usd, 8)

    @property
    def relative_drift(self) -> float | None:
        if self.comparable_runs == 0 or self.estimated_cost_usd <= 0:
            return None
        return round((self.reported_cost_usd - self.estimated_cost_usd) / self.estimated_cost_usd, 6)

    @property
    def flagged_models(self) -> list[ModelDrift]:
        return [m for m in self.models if m.flagged]

    @property
    def causes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for model in self.flagged_models:
            if model.cause:
                counts[model.cause] = counts.get(model.cause, 0) + 1
        return counts

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tolerance": self.tolerance,
            "total_runs": self.total_runs,
            "comparable_runs": self.comparable_runs,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "reported_cost_usd": round(self.reported_cost_usd, 8),
            "absolute_drift_usd": self.absolute_drift_usd,
            "relative_drift": self.relative_drift,
            "missing_reported": self.missing_reported,
            "missing_estimate": self.missing_estimate,
            "flagged_models": [m.model for m in self.flagged_models],
            "causes": self.causes,
            "models": [m.to_dict() for m in self.models],
        }


def reconcile(
    records: Iterable[RunRecord],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    absolute_floor_usd: float = DEFAULT_ABSOLUTE_FLOOR_USD,
) -> ReconciliationReport:
    """Compare estimated against reported cost and flag models that drift."""

    rows = list(records)
    report = ReconciliationReport(tolerance=tolerance, total_runs=len(rows))
    buckets: dict[str, ModelDrift] = {}

    for row in rows:
        bucket = buckets.setdefault(row.model, ModelDrift(model=row.model))
        has_estimate = row.estimated_cost_usd is not None
        has_reported = row.reported_cost_usd is not None
        if has_estimate and has_reported:
            bucket.comparable_runs += 1
            bucket.estimated_cost_usd += float(row.estimated_cost_usd)
            bucket.reported_cost_usd += float(row.reported_cost_usd)
            estimated_tokens = _sum_tokens(row.estimated_input_tokens, row.estimated_output_tokens)
            reported_tokens = _sum_tokens(row.prompt_tokens, row.completion_tokens)
            if estimated_tokens is not None and reported_tokens is not None:
                bucket.token_comparable_runs += 1
                bucket.estimated_tokens += estimated_tokens
                bucket.reported_tokens += reported_tokens
        elif has_estimate:
            bucket.missing_reported += 1
        elif has_reported:
            bucket.missing_estimate += 1
        else:
            bucket.missing_reported += 1
            bucket.missing_estimate += 1

    for bucket in buckets.values():
        relative = bucket.relative_drift
        if bucket.comparable_runs == 0:
            bucket.reason = "no run carried both an estimate and a reported cost"
        elif abs(bucket.absolute_drift_usd) < absolute_floor_usd:
            bucket.reason = f"drift ${abs(bucket.absolute_drift_usd):.6f} below ${absolute_floor_usd} floor"
        elif relative is None:
            bucket.flagged = True
            bucket.cause = CAUSE_NO_PRICE
            bucket.reason = (
                f"estimated ${bucket.estimated_cost_usd:.6f} but provider reported "
                f"${bucket.reported_cost_usd:.6f} - a zero estimate against a real charge "
                "means the catalog has no price for this model"
            )
        elif abs(relative) > tolerance:
            bucket.flagged = True
            token_drift = bucket.token_drift
            if token_drift is not None and abs(token_drift) > TOKEN_TOLERANCE:
                bucket.cause = CAUSE_TOKEN_ESTIMATE
                bucket.reason = (
                    f"cost off by {relative * 100:.1f}% (tolerance {tolerance * 100:.1f}%), "
                    f"and token count off by {token_drift * 100:.1f}% "
                    f"({bucket.estimated_tokens} estimated vs {bucket.reported_tokens} actual) - "
                    "the TaskSpec token sizes are wrong, not the catalog price"
                )
            elif token_drift is not None:
                bucket.cause = CAUSE_STALE_PRICE
                bucket.reason = (
                    f"cost off by {relative * 100:.1f}% (tolerance {tolerance * 100:.1f}%) "
                    f"while token counts agree within {token_drift * 100:.1f}% - "
                    "the catalog price is stale, run refresh"
                )
            else:
                bucket.cause = CAUSE_STALE_PRICE
                bucket.reason = (
                    f"cost off by {relative * 100:.1f}% (tolerance {tolerance * 100:.1f}%); "
                    "no token counts recorded, so this is most likely a stale catalog "
                    "price - run refresh"
                )
        else:
            bucket.reason = f"within tolerance ({relative * 100:.1f}%)"

        report.comparable_runs += bucket.comparable_runs
        report.estimated_cost_usd += bucket.estimated_cost_usd
        report.reported_cost_usd += bucket.reported_cost_usd
        report.missing_reported += bucket.missing_reported
        report.missing_estimate += bucket.missing_estimate

    report.models = sorted(buckets.values(), key=lambda m: m.model)

    if report.flagged_models:
        report.status = STATUS_DRIFT
    elif report.comparable_runs == 0:
        report.status = STATUS_INSUFFICIENT_DATA
    else:
        report.status = STATUS_OK
    return report


def _sum_tokens(first: int | None, second: int | None) -> int | None:
    if first is None or second is None:
        return None
    return int(first) + int(second)


def format_report(report: ReconciliationReport) -> str:
    """Human-readable reconciliation summary."""

    lines = [
        f"reconciliation: {report.status.upper()}",
        f"  runs                 {report.total_runs} ({report.comparable_runs} comparable)",
        f"  estimated            ${report.estimated_cost_usd:.6f}",
        f"  reported             ${report.reported_cost_usd:.6f}",
        f"  drift                ${report.absolute_drift_usd:+.6f}"
        + (f" ({report.relative_drift * 100:+.1f}%)" if report.relative_drift is not None else ""),
        f"  missing reported     {report.missing_reported}",
        f"  missing estimate     {report.missing_estimate}",
    ]
    for model in report.models:
        marker = "FLAG" if model.flagged else "ok  "
        lines.append(f"  [{marker}] {model.model}: {model.reason}")
    return "\n".join(lines)
