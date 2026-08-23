"""Append-only JSONL run ledger: one record per model call, success or failure.

The point of this file is the failures. Cost-per-course today rests on a
*guessed* 1.35 retry multiplier; a ledger that only records successes can never
replace that guess, because the multiplier is by definition
``attempts / accepted outputs``. So every call is recorded -- completed,
gate-failed, and errored alike -- and :meth:`RunLedger.summary` reports the
multiplier as **measured** only when there is at least one accepted run to
divide by. With no data it says so instead of returning a number.

Format: one JSON object per line, standard library only, no schema migrations.
A truncated or hand-edited line is skipped and counted, never fatal.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 2

#: A run either produced an accepted artifact, produced one the gates rejected,
#: or never produced one at all. Only ``completed`` counts as a usable output.
STATUS_COMPLETED = "completed"
STATUS_GATE_FAILED = "gate_failed"
STATUS_ERROR = "error"
VALID_STATUSES = (STATUS_COMPLETED, STATUS_GATE_FAILED, STATUS_ERROR)


def default_ledger_path() -> Path:
    override = os.environ.get("OPENROUTER_MODEL_ROUTER_LEDGER")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "openrouter-model-router" / "runs.jsonl"


@dataclass
class RunRecord:
    """One model call."""

    model: str
    task_label: str = ""
    status: str = STATUS_COMPLETED
    timestamp: str = ""
    run_id: str = ""
    attempt: int = 1
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    reported_cost_usd: float | None = None
    latency_ms: float | None = None
    gates_passed: bool | None = None
    gate_failures: tuple[str, ...] = ()
    error: str | None = None
    usage_present: bool = False
    usage_source_fields: dict[str, str] = field(default_factory=dict)
    catalog_updated_at: str | None = None
    #: When the catalog that priced this run was last fetched from OpenRouter.
    #: Distinct from catalog_updated_at, which moves on any local edit: only this
    #: field can tell you the prices behind an estimate were months old.
    catalog_fetched_at: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {self.status!r}")
        if not self.timestamp:
            self.timestamp = _utc_now()
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:16]
        self.gate_failures = tuple(self.gate_failures)
        if self.gates_passed is None and self.status == STATUS_GATE_FAILED:
            self.gates_passed = False
        if self.gates_passed is None and self.status == STATUS_COMPLETED:
            self.gates_passed = True

    @property
    def succeeded(self) -> bool:
        """True only for a run whose output was produced AND passed its gates."""

        return self.status == STATUS_COMPLETED and self.gates_passed is not False

    @property
    def cost_drift_usd(self) -> float | None:
        if self.estimated_cost_usd is None or self.reported_cost_usd is None:
            return None
        return self.reported_cost_usd - self.estimated_cost_usd

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gate_failures"] = list(self.gate_failures)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - dataclass API
        payload = {k: v for k, v in data.items() if k in known}
        payload["gate_failures"] = tuple(payload.get("gate_failures") or ())
        payload["usage_source_fields"] = dict(payload.get("usage_source_fields") or {})
        return cls(**payload)


class RunLedger:
    """Append-only JSONL ledger of model calls."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path).expanduser() if path else default_ledger_path()

    def append(self, record: RunRecord) -> RunRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def record(self, **kwargs: Any) -> RunRecord:
        return self.append(RunRecord(**kwargs))

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self.read_all())

    def __len__(self) -> int:
        return len(self.read_all())

    def read_all(self) -> list[RunRecord]:
        records, _ = self.read_with_errors()
        return records

    def read_with_errors(self) -> tuple[list[RunRecord], list[str]]:
        if not self.path.exists():
            return [], []
        records: list[RunRecord] = []
        errors: list[str] = []
        for number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                records.append(RunRecord.from_dict(json.loads(raw)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"line {number}: {exc}")
        return records, errors

    def summary(self, records: Iterable[RunRecord] | None = None) -> dict[str, Any]:
        rows = list(records) if records is not None else self.read_all()
        total = len(rows)
        completed = [r for r in rows if r.succeeded]
        gate_failed = [r for r in rows if r.status == STATUS_GATE_FAILED or r.gates_passed is False]
        errored = [r for r in rows if r.status == STATUS_ERROR]

        estimated = _sum_optional(r.estimated_cost_usd for r in rows)
        reported = _sum_optional(r.reported_cost_usd for r in rows)
        # A total summed over rows that carried no estimate is a total of an
        # unknown share of the spend. Say how many rows it skipped rather than
        # letting the number pass for the whole bill.
        missing_estimate = sum(1 for r in rows if r.estimated_cost_usd is None)
        missing_reported = sum(1 for r in rows if r.reported_cost_usd is None)
        wasted = _sum_optional(
            r.reported_cost_usd if r.reported_cost_usd is not None else r.estimated_cost_usd
            for r in rows
            if not r.succeeded
        )

        if completed:
            multiplier: float | None = round(total / len(completed), 4)
            measured = True
            note = f"measured from {total} attempt(s) over {len(completed)} accepted run(s)"
        else:
            multiplier = None
            measured = False
            note = "no accepted runs yet - retry multiplier is NOT measurable from this ledger"

        return {
            "ledger": str(self.path),
            "runs": total,
            "completed": len(completed),
            "gate_failed": len(gate_failed),
            "errored": len(errored),
            "failed": total - len(completed),
            "prompt_tokens": int(_sum_optional(r.prompt_tokens for r in rows)),
            "completion_tokens": int(_sum_optional(r.completion_tokens for r in rows)),
            "estimated_cost_usd": round(estimated, 6),
            "reported_cost_usd": round(reported, 6),
            "runs_missing_estimate": missing_estimate,
            "runs_missing_reported": missing_reported,
            "estimated_cost_is_complete": missing_estimate == 0,
            "estimated_cost_note": (
                f"summed over {total - missing_estimate} of {total} run(s); "
                f"{missing_estimate} carried no estimate and contribute nothing to this total"
            ),
            "cost_of_failed_runs_usd": round(wasted, 6),
            "retry_multiplier": multiplier,
            "retry_multiplier_measured": measured,
            "retry_multiplier_note": note,
            "by_model": _by_model(rows),
        }


def _by_model(rows: list[RunRecord]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = out.setdefault(
            row.model,
            {
                "runs": 0,
                "completed": 0,
                "failed": 0,
                "estimated_cost_usd": 0.0,
                "reported_cost_usd": 0.0,
                "runs_missing_estimate": 0,
            },
        )
        bucket["runs"] += 1
        bucket["completed" if row.succeeded else "failed"] += 1
        if row.estimated_cost_usd is None:
            bucket["runs_missing_estimate"] += 1
        bucket["estimated_cost_usd"] = round(bucket["estimated_cost_usd"] + (row.estimated_cost_usd or 0.0), 6)
        bucket["reported_cost_usd"] = round(bucket["reported_cost_usd"] + (row.reported_cost_usd or 0.0), 6)
    return out


def _sum_optional(values: Iterable[Any]) -> float:
    total = 0.0
    for value in values:
        if value is not None:
            total += float(value)
    return total


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
