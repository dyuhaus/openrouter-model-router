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

It names a cause only when *every* run in the cost comparison also carried token
counts. On partial coverage the token sums and the cost sums cover different
subsets of runs and cannot be divided against each other, so the cause is
reported as ``undetermined`` rather than guessed.

Drift is reported per model and overall. A run that carries only one of the two
numbers is *excluded from the drift maths and counted separately*, because
silently treating a missing reported cost as $0.00 would manufacture a 100%
drift, and treating it as "matching" would hide a real one.

Excluded is not forgotten. Real dollars charged on a run the estimator could not
price are summed into ``unreconciled_reported_cost_usd`` and reported as their
own verdict. Dropping them would let a model with unknown pricing spend real
money underneath a report headed OK -- absence of an estimate rendering as $0.00
of spend, which is the same defect as an unpriced model rendering as free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .catalog import DEFAULT_MAX_AGE_DAYS, STALENESS_FRESH, ModelCatalog
from .ledger import RunRecord

DEFAULT_TOLERANCE = 0.10  # 10% relative drift before a model is flagged.
DEFAULT_ABSOLUTE_FLOOR_USD = 0.0005  # Ignore drift on runs too cheap to matter.
TOKEN_TOLERANCE = 0.10  # Token drift beyond this means the estimate's sizes were wrong.

CAUSE_STALE_PRICE = "stale_catalog_price"
CAUSE_TOKEN_ESTIMATE = "wrong_token_estimate"
CAUSE_NO_PRICE = "no_catalog_price"
CAUSE_UNDETERMINED = "undetermined"  # Cost is wrong; the evidence cannot say why.

STATUS_OK = "ok"
STATUS_DRIFT = "drift"
STATUS_INSUFFICIENT_DATA = "insufficient_data"

#: Verdict for a report where real money was charged on runs carrying no
#: estimate. The drift maths cannot say anything about those dollars, so the
#: report may not say "ok" about them either.
VERDICT_UNRECONCILED = "unreconciled_cost"

#: What a staleness answer looks like when no catalog was handed in. This is NOT
#: "fresh" - a report run without a catalog has checked nothing, and a check that
#: examined nothing may not report a pass.
CATALOG_NOT_CHECKED = {
    "status": "not_checked",
    "fresh": False,
    "stale": False,
    "checked": False,
    "reason": "no catalog was passed to reconcile(), so its age was never checked",
}


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
    #: Dollars the provider actually charged on runs that carried no estimate.
    #: These are outside the drift maths by design and must not vanish.
    unreconciled_reported_cost_usd: float = 0.0
    #: What the catalog says about this model's price, when a catalog was passed:
    #: True (published), False (UNKNOWN -- the -1 sentinel or no pricing block),
    #: or None (no catalog, or the model is not in it).
    pricing_known: bool | None = None
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
    def token_evidence_is_complete(self) -> bool:
        """True when every run in the cost comparison also carried token counts.

        When it is False the token sums cover a *different subset of runs* than
        the cost sums, so the two cannot be divided against each other to name a
        cause. Diagnosing from partial coverage would let one untokened,
        wildly-overcharged run hide behind a handful of well-behaved ones.
        """

        return self.comparable_runs > 0 and self.token_comparable_runs == self.comparable_runs

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
            "token_comparable_runs": self.token_comparable_runs,
            "token_evidence_is_complete": self.token_evidence_is_complete,
            "unreconciled_reported_cost_usd": round(self.unreconciled_reported_cost_usd, 8),
            "pricing_known": self.pricing_known,
            "pricing_status": _pricing_status(self.pricing_known),
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
    unreconciled_reported_cost_usd: float = 0.0
    absolute_floor_usd: float = DEFAULT_ABSOLUTE_FLOOR_USD
    models: list[ModelDrift] = field(default_factory=list)
    catalog: dict[str, Any] = field(default_factory=lambda: dict(CATALOG_NOT_CHECKED))

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
    def has_unreconciled_cost(self) -> bool:
        """Real dollars sit outside the comparison entirely.

        Not "small drift" -- unmeasured spend. The estimator produced no number
        for these runs, so no tolerance applies to them and nothing in the drift
        maths would ever mention them.
        """

        return self.unreconciled_reported_cost_usd > self.absolute_floor_usd

    @property
    def pricing_unknown_models(self) -> list[str]:
        """Ledger models the catalog cannot price. Empty when no catalog was passed."""

        return [m.model for m in self.models if m.pricing_known is False]

    @property
    def catalog_checked(self) -> bool:
        return bool(self.catalog.get("checked"))

    @property
    def catalog_is_stale(self) -> bool:
        """True only when a catalog WAS checked and failed the age test."""

        return self.catalog_checked and bool(self.catalog.get("stale"))

    @property
    def verdict(self) -> str:
        """The whole report's answer, not just the drift maths'.

        ``status`` describes cost drift alone, so a stale catalog or an empty
        comparison used to leave a report headed "OK" while the command exited
        non-zero. A header that disagrees with the exit code is how a reader ends
        up believing the reassuring half.
        """

        if self.status == STATUS_DRIFT:
            return "drift"
        if self.catalog_is_stale:
            return "stale_catalog"
        if self.has_unreconciled_cost:
            return VERDICT_UNRECONCILED
        if self.status == STATUS_INSUFFICIENT_DATA:
            return STATUS_INSUFFICIENT_DATA
        return STATUS_OK

    @property
    def ok(self) -> bool:
        """Everything the report can vouch for held.

        A stale catalog fails this even when every drift number is inside
        tolerance, because a drift comparison against prices nobody has refreshed
        is two unverified numbers agreeing with each other.
        """

        return self.status == STATUS_OK and not self.catalog_is_stale and not self.has_unreconciled_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "tolerance": self.tolerance,
            "total_runs": self.total_runs,
            "comparable_runs": self.comparable_runs,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "reported_cost_usd": round(self.reported_cost_usd, 8),
            "absolute_drift_usd": self.absolute_drift_usd,
            "relative_drift": self.relative_drift,
            "missing_reported": self.missing_reported,
            "missing_estimate": self.missing_estimate,
            "unreconciled_reported_cost_usd": round(self.unreconciled_reported_cost_usd, 8),
            "has_unreconciled_cost": self.has_unreconciled_cost,
            "pricing_unknown_models": self.pricing_unknown_models,
            "flagged_models": [m.model for m in self.flagged_models],
            "causes": self.causes,
            "catalog": dict(self.catalog),
            "catalog_is_stale": self.catalog_is_stale,
            "ok": self.ok,
            "models": [m.to_dict() for m in self.models],
        }


def reconcile(
    records: Iterable[RunRecord],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    absolute_floor_usd: float = DEFAULT_ABSOLUTE_FLOOR_USD,
    catalog: ModelCatalog | None = None,
    max_catalog_age_days: float = DEFAULT_MAX_AGE_DAYS,
) -> ReconciliationReport:
    """Compare estimated against reported cost and flag models that drift.

    Pass ``catalog`` to have the age of the prices checked as well. The drift
    maths compares an estimate built from catalog prices against what the
    provider charged; if nobody knows how old those prices are, "within
    tolerance" is not a finding.
    """

    rows = list(records)
    report = ReconciliationReport(
        tolerance=tolerance, total_runs=len(rows), absolute_floor_usd=absolute_floor_usd
    )
    if catalog is not None:
        staleness = catalog.staleness(max_age_days=max_catalog_age_days)
        staleness["checked"] = True
        report.catalog = staleness
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
            # Money that was really charged and that no estimate covers. Held
            # separately rather than dropped: a model the catalog cannot price
            # would otherwise spend real dollars invisibly under an OK report.
            bucket.unreconciled_reported_cost_usd += float(row.reported_cost_usd)
        else:
            bucket.missing_reported += 1
            bucket.missing_estimate += 1

    for bucket in buckets.values():
        if catalog is not None:
            entry = catalog.get(bucket.model)
            bucket.pricing_known = None if entry is None else entry.pricing_known
        relative = bucket.relative_drift
        if bucket.comparable_runs == 0:
            bucket.reason = "no run carried both an estimate and a reported cost"
            if bucket.unreconciled_reported_cost_usd > absolute_floor_usd:
                bucket.reason += (
                    f"; ${bucket.unreconciled_reported_cost_usd:.6f} was charged with no estimate to check it "
                    "against" + (" (catalog price UNKNOWN)" if bucket.pricing_known is False else "")
                )
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
            head = f"cost off by {relative * 100:.1f}% (tolerance {tolerance * 100:.1f}%)"
            if not bucket.token_evidence_is_complete:
                # Partial or absent token coverage. Say the cost is wrong; do NOT
                # name a cause the evidence cannot support.
                bucket.cause = CAUSE_UNDETERMINED
                covered = f"{bucket.token_comparable_runs} of {bucket.comparable_runs}"
                bucket.reason = (
                    f"{head}; only {covered} comparable run(s) carried token counts, "
                    "so a stale catalog price cannot be told apart from a wrong "
                    "TaskSpec token estimate - refresh the catalog first, then re-check"
                )
            elif abs(token_drift) > TOKEN_TOLERANCE:
                bucket.cause = CAUSE_TOKEN_ESTIMATE
                bucket.reason = (
                    f"{head}, and token count off by {token_drift * 100:.1f}% "
                    f"({bucket.estimated_tokens} estimated vs {bucket.reported_tokens} actual) - "
                    "the TaskSpec token sizes are wrong, not the catalog price"
                )
            else:
                bucket.cause = CAUSE_STALE_PRICE
                bucket.reason = (
                    f"{head} while token counts agree within {token_drift * 100:.1f}% - "
                    "the catalog price is stale, run refresh"
                )
        else:
            bucket.reason = f"within tolerance ({relative * 100:.1f}%)"

        report.comparable_runs += bucket.comparable_runs
        report.estimated_cost_usd += bucket.estimated_cost_usd
        report.reported_cost_usd += bucket.reported_cost_usd
        report.missing_reported += bucket.missing_reported
        report.missing_estimate += bucket.missing_estimate
        report.unreconciled_reported_cost_usd += bucket.unreconciled_reported_cost_usd

    report.models = sorted(buckets.values(), key=lambda m: m.model)

    if report.flagged_models:
        report.status = STATUS_DRIFT
    elif report.comparable_runs == 0:
        report.status = STATUS_INSUFFICIENT_DATA
    else:
        report.status = STATUS_OK
    return report


def _pricing_status(pricing_known: bool | None) -> str:
    """Never renders an unknown price as anything a reader could read as free."""

    if pricing_known is None:
        return "not_checked"
    return "known" if pricing_known else "UNKNOWN"


def _sum_tokens(first: int | None, second: int | None) -> int | None:
    if first is None or second is None:
        return None
    return int(first) + int(second)


def format_report(report: ReconciliationReport) -> str:
    """Human-readable reconciliation summary."""

    header = f"reconciliation: {report.verdict.upper()}"
    if report.verdict != report.status:
        header += f"  (cost drift: {report.status})"
    lines = [
        header,
        f"  runs                 {report.total_runs} ({report.comparable_runs} comparable)",
        f"  estimated            ${report.estimated_cost_usd:.6f}",
        f"  reported             ${report.reported_cost_usd:.6f}",
        f"  drift                ${report.absolute_drift_usd:+.6f}"
        + (f" ({report.relative_drift * 100:+.1f}%)" if report.relative_drift is not None else ""),
        f"  missing reported     {report.missing_reported}",
        f"  missing estimate     {report.missing_estimate}",
        f"  unreconciled charge  ${report.unreconciled_reported_cost_usd:.6f}"
        + ("  <-- real money no estimate covers" if report.has_unreconciled_cost else ""),
    ]
    catalog = report.catalog
    if not catalog.get("checked"):
        lines.append("  catalog age         NOT CHECKED (pass --catalog to check it)")
    else:
        # Name the actual status. Labelling an unverifiable legacy timestamp
        # "STALE" would be a second inaccuracy on top of the one being reported.
        status_name = str(catalog.get("status") or "unknown")
        marker = "ok" if status_name == STALENESS_FRESH else status_name.split("_")[0].upper()
        age = catalog.get("age_days")
        age_text = "unknown" if age is None else f"{age:.1f}d"
        lines.append(
            f"  catalog age         [{marker}] {age_text} "
            f"(fetched {catalog.get('fetched_at') or 'never'}, {catalog.get('models')} models)"
        )
        lines.append(f"                      {catalog.get('reason')}")

    for model in report.models:
        marker = "FLAG" if model.flagged else "ok  "
        price_note = ""
        if model.pricing_known is False:
            # Said out loud on every line for this model. "$0.00 estimated" and
            # "no price exists" must never look the same in a report a human
            # skims for a dollar figure.
            price_note = " [catalog price UNKNOWN - its estimates are not measurements]"
        lines.append(f"  [{marker}] {model.model}: {model.reason}{price_note}")

    if report.pricing_unknown_models:
        lines.append("")
        lines.append(
            "  !! UNPRICED MODELS IN THIS LEDGER: "
            + ", ".join(report.pricing_unknown_models)
        )
        lines.append("     The catalog publishes no price for these, so a $0.00 estimate")
        lines.append("     from them is an absence of measurement, not an absence of cost.")

    if report.has_unreconciled_cost:
        lines.append("")
        lines.append(
            f"  !! ${report.unreconciled_reported_cost_usd:.6f} WAS CHARGED ON {report.missing_estimate} RUN(S) "
            "CARRYING NO ESTIMATE."
        )
        lines.append("     Nothing above verified those dollars: they are outside the drift")
        lines.append("     maths entirely, and no tolerance was applied to them.")

    if report.catalog_is_stale:
        lines.append("")
        lines.append(
            f"  !! CATALOG NOT FRESH ({report.catalog.get('status')}): every estimated cost above"
        )
        lines.append("     was computed from prices of unknown currency. Run")
        lines.append("     `openrouter-model-router refresh` and re-run.")
    if report.status == STATUS_INSUFFICIENT_DATA:
        lines.append("")
        lines.append("  !! NOTHING WAS COMPARED: 0 runs carried both an estimate and a reported")
        lines.append("     cost. This is a configuration failure, not a clean bill of health.")
    return "\n".join(lines)
