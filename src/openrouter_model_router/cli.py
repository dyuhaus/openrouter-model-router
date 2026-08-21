"""Command line interface for OpenRouter Model Router."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .catalog import (
    DEFAULT_MAX_AGE_DAYS,
    OPENROUTER_BASE_URL,
    STALENESS_FRESH,
    CatalogLoadError,
    CatalogRefreshError,
    ModelCatalog,
    default_catalog_path,
)
from .ledger import RunLedger
from .reconcile import DEFAULT_TOLERANCE, STATUS_INSUFFICIENT_DATA, format_report, reconcile
from .router import ModelRouter
from .types import TaskSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openrouter-model-router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    refresh = subparsers.add_parser(
        "refresh",
        aliases=["research", "update"],
        help="Refresh local catalog from OpenRouter /models",
    )
    refresh.add_argument("--catalog", default=None, help="Catalog path")
    refresh.add_argument("--base-url", default=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL))
    refresh.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    refresh.add_argument("--timeout", type=float, default=20.0)
    refresh.add_argument("--include-raw", action="store_true", help="Persist raw OpenRouter records")

    select = subparsers.add_parser("select", help="Select a model for a task")
    _add_task_args(select)
    select.add_argument("--catalog", default=None, help="Catalog path")

    outcome = subparsers.add_parser("record-outcome", help="Record observed success/latency/quality for a model")
    outcome.add_argument("model_id")
    outcome.add_argument("--catalog", default=None)
    outcome.add_argument("--success", action=argparse.BooleanOptionalAction, default=True)
    outcome.add_argument("--latency-ms", type=float, default=None)
    outcome.add_argument("--quality-score", type=float, default=None)

    estimate = subparsers.add_parser(
        "estimate",
        help="Estimate the cost of a request on one or more specific models",
    )
    estimate.add_argument("--catalog", default=None)
    estimate.add_argument("--input-tokens", type=int, default=40_000)
    estimate.add_argument("--output-tokens", type=int, default=120_000)
    estimate.add_argument("--model", dest="models", action="append", default=[], help="Repeatable")
    estimate.add_argument("--max-catalog-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS)

    ledger_cmd = subparsers.add_parser("ledger", help="Summarize the run ledger")
    ledger_cmd.add_argument("--ledger", default=None)

    status = subparsers.add_parser(
        "catalog-status",
        help="Report how old the local catalog is; exits non-zero when it is not fresh",
    )
    status.add_argument("--catalog", default=None)
    status.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS)
    status.add_argument("--json", action="store_true")

    recon = subparsers.add_parser("reconcile", help="Compare estimated against provider-reported cost")
    recon.add_argument("--ledger", default=None)
    recon.add_argument("--catalog", default=None, help="Catalog to age-check alongside the drift maths")
    recon.add_argument("--max-catalog-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS)
    recon.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    recon.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    recon.add_argument(
        "--fail-on-drift",
        action="store_true",
        help=(
            "Exit non-zero when any model drifts past the tolerance, when the checked "
            "catalog is stale, or when NOTHING could be compared"
        ),
    )

    args = parser.parse_args(argv)
    handlers = {
        "refresh": _refresh,
        "research": _refresh,
        "update": _refresh,
        "select": _select,
        "record-outcome": _record_outcome,
        "estimate": _estimate,
        "ledger": _ledger,
        "reconcile": _reconcile,
        "catalog-status": _catalog_status,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
        return 2
    try:
        return handler(args)
    except CatalogLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _add_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="general", help="Task type, such as coding, summarization, extraction")
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=1000)
    parser.add_argument("--modality", dest="modalities", action="append", default=[])
    parser.add_argument("--required-capability", dest="required_capabilities", action="append", default=[])
    parser.add_argument("--preference", default="balanced", choices=["balanced", "cheap", "cost", "fast", "quality", "best"])
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--min-context-tokens", type=int, default=None)
    parser.add_argument("--allow-model", dest="allow_models", action="append", default=[])
    parser.add_argument("--block-model", dest="block_models", action="append", default=[])
    parser.add_argument("--fallback-model", default=None)


def _refresh(args: argparse.Namespace) -> int:
    path = Path(args.catalog).expanduser() if args.catalog else default_catalog_path()
    api_key = os.environ.get(args.api_key_env)
    try:
        incoming = ModelCatalog.refresh_from_openrouter(base_url=args.base_url, api_key=api_key, timeout=args.timeout)
    except CatalogRefreshError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if len(incoming) == 0:
        print(
            "refresh fetched a catalog containing 0 models; refusing to overwrite "
            f"{path} with nothing - a catalog of nothing prices nothing",
            file=sys.stderr,
        )
        return 1

    current = ModelCatalog.load(path, bootstrap=False) if path.exists() else ModelCatalog()
    added, updated = current.merge(incoming)
    saved = current.save(path, include_raw=args.include_raw)
    coverage = current.pricing_coverage()
    print(
        json.dumps(
            {
                "catalog": str(saved),
                "models": len(current),
                "added": added,
                "updated": updated,
                "fetched_at": current.fetched_at,
                "authenticated": bool(api_key),
                "pricing": coverage,
                "staleness": current.staleness(),
            },
            indent=2,
        )
    )
    if coverage["total"] and coverage["priced"] == 0:
        print(
            "refresh wrote a catalog in which NO model carries a usable price; "
            "every cost estimate from it will be $0.00",
            file=sys.stderr,
        )
        return 1
    return 0


def _select(args: argparse.Namespace) -> int:
    router = ModelRouter.from_file(args.catalog)
    task = TaskSpec(
        task_type=args.task,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        modalities=tuple(args.modalities or ["text"]),
        required_capabilities=tuple(args.required_capabilities),
        preference=args.preference,
        max_cost_usd=args.max_cost,
        min_context_tokens=args.min_context_tokens,
        allow_models=tuple(args.allow_models),
        block_models=tuple(args.block_models),
        fallback_model=args.fallback_model,
    )
    try:
        selection = router.select(task)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(selection.to_dict(), indent=2, sort_keys=True))
    return 0


def _record_outcome(args: argparse.Namespace) -> int:
    path = Path(args.catalog).expanduser() if args.catalog else default_catalog_path()
    catalog = ModelCatalog.load(path)
    try:
        model = catalog.record_outcome(
            args.model_id,
            success=args.success,
            latency_ms=args.latency_ms,
            quality_score=args.quality_score,
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    catalog.save(path)
    print(json.dumps(model.to_dict(), indent=2, sort_keys=True))
    return 0


def _estimate(args: argparse.Namespace) -> int:
    catalog = ModelCatalog.load(args.catalog, bootstrap=False)
    if len(catalog) == 0:
        print(
            "catalog is empty - run `openrouter-model-router refresh` first, "
            "or every estimate below would be $0.00",
            file=sys.stderr,
        )
        return 1

    rows = []
    missing = []
    priced = 0
    for model_id in args.models or sorted(m.id for m in catalog)[:3]:
        model = catalog.get(model_id)
        if model is None:
            missing.append(model_id)
            continue
        # cost_estimate() returns None for an unpriced model. Printing null and
        # "UNKNOWN" instead of 0.0 is the whole point: $0.00 reads as a cheap
        # model and is how a bill arrives for a request the estimator called free.
        cost = model.cost_estimate(args.input_tokens, args.output_tokens)
        if cost is not None:
            priced += 1
        rows.append(
            {
                "model": model.id,
                "input_cost_per_million_usd": model.input_cost_per_million if model.pricing_known else None,
                "output_cost_per_million_usd": model.output_cost_per_million if model.pricing_known else None,
                "pricing_known": model.pricing_known,
                "pricing_status": "known" if model.pricing_known else "UNKNOWN",
                "context_length": model.context_length,
                "estimated_cost_usd": None if cost is None else round(cost, 6),
            }
        )

    staleness = catalog.staleness(max_age_days=args.max_catalog_age_days)
    print(
        json.dumps(
            {
                "catalog": str(Path(args.catalog).expanduser() if args.catalog else default_catalog_path()),
                "catalog_updated_at": catalog.updated_at,
                "catalog_fetched_at": catalog.fetched_at,
                "catalog_staleness": staleness,
                "catalog_models": len(catalog),
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "estimates": rows,
                "priced_estimates": priced,
                "unpriced_estimates": len(rows) - priced,
                "unknown_models": missing,
            },
            indent=2,
        )
    )

    if staleness["status"] != STALENESS_FRESH:
        print(f"WARNING: {staleness['reason']}", file=sys.stderr)
    if rows and priced == 0:
        print(
            f"none of the {len(rows)} requested model(s) carries a usable price, so this "
            "command produced no cost estimate at all - an UNKNOWN price is not $0.00",
            file=sys.stderr,
        )
        return 1
    return 1 if missing else 0


def _catalog_status(args: argparse.Namespace) -> int:
    """Freshness gate. Exit 0 ONLY for a catalog proven fresh.

    Not-fresh includes "I cannot tell": a missing, unparseable, or future fetch
    timestamp, and an empty catalog, all exit non-zero. A check that cannot see
    the thing it is checking has not passed.
    """

    path = Path(args.catalog).expanduser() if args.catalog else default_catalog_path()
    if not path.exists():
        print(f"no catalog at {path} - run `openrouter-model-router refresh`", file=sys.stderr)
        return 1
    catalog = ModelCatalog.load(path, bootstrap=False)
    report = catalog.staleness(max_age_days=args.max_age_days)
    report["catalog"] = str(path)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        marker = "FRESH" if report["fresh"] else report["status"].upper()
        print(f"catalog: {marker}")
        print(f"  path            {path}")
        print(f"  models          {report['models']}")
        print(f"  fetched_at      {report['fetched_at'] or 'never'}")
        print(f"  updated_at      {report['updated_at']}")
        age = report["age_days"]
        print(f"  age             {'unknown' if age is None else f'{age:.2f} days'}")
        print(f"  limit           {report['max_age_days']:g} days")
        print(f"  {report['reason']}")
    if not report["fresh"]:
        print(f"catalog is not fresh: {report['reason']}", file=sys.stderr)
        return 1
    return 0


def _ledger(args: argparse.Namespace) -> int:
    ledger = RunLedger(args.ledger)
    records, errors = ledger.read_with_errors()
    summary = ledger.summary(records)
    summary["unreadable_lines"] = errors
    print(json.dumps(summary, indent=2))
    return 0


def _reconcile(args: argparse.Namespace) -> int:
    ledger = RunLedger(args.ledger)
    catalog = None
    catalog_path = Path(args.catalog).expanduser() if args.catalog else None
    if catalog_path is not None:
        if not catalog_path.exists():
            print(f"no catalog at {catalog_path} - cannot age-check it", file=sys.stderr)
            return 1
        catalog = ModelCatalog.load(catalog_path, bootstrap=False)

    report = reconcile(
        ledger.read_all(),
        tolerance=args.tolerance,
        catalog=catalog,
        max_catalog_age_days=args.max_catalog_age_days,
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else format_report(report))

    if not args.fail_on_drift:
        return 0
    if report.flagged_models:
        return 1
    if report.catalog_is_stale:
        print(f"catalog is stale: {report.catalog.get('reason')}", file=sys.stderr)
        return 1
    if report.status == STATUS_INSUFFICIENT_DATA:
        # Zero comparable runs is a configuration failure, not a pass. Exiting 0
        # here is the exact shape of gate that prints a pass over nothing.
        print(
            "reconcile compared 0 runs: no ledger row carried both an estimated and a "
            "reported cost, so nothing was verified",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
