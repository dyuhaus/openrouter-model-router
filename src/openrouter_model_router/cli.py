"""Command line interface for OpenRouter Model Router."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .catalog import (
    OPENROUTER_BASE_URL,
    CatalogLoadError,
    CatalogRefreshError,
    ModelCatalog,
    default_catalog_path,
)
from .ledger import RunLedger
from .reconcile import DEFAULT_TOLERANCE, format_report, reconcile
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

    ledger_cmd = subparsers.add_parser("ledger", help="Summarize the run ledger")
    ledger_cmd.add_argument("--ledger", default=None)

    recon = subparsers.add_parser("reconcile", help="Compare estimated against provider-reported cost")
    recon.add_argument("--ledger", default=None)
    recon.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    recon.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    recon.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit non-zero when any model drifts past the tolerance",
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
                "authenticated": bool(api_key),
                "pricing": coverage,
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
    for model_id in args.models or sorted(m.id for m in catalog)[:3]:
        model = catalog.get(model_id)
        if model is None:
            missing.append(model_id)
            continue
        rows.append(
            {
                "model": model.id,
                "input_cost_per_million_usd": model.input_cost_per_million,
                "output_cost_per_million_usd": model.output_cost_per_million,
                "pricing_known": model.pricing_known,
                "context_length": model.context_length,
                "estimated_cost_usd": round(
                    model.estimated_cost_usd(args.input_tokens, args.output_tokens), 6
                ),
            }
        )

    print(
        json.dumps(
            {
                "catalog": str(Path(args.catalog).expanduser() if args.catalog else default_catalog_path()),
                "catalog_updated_at": catalog.updated_at,
                "catalog_models": len(catalog),
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "estimates": rows,
                "unknown_models": missing,
            },
            indent=2,
        )
    )
    return 1 if missing else 0


def _ledger(args: argparse.Namespace) -> int:
    ledger = RunLedger(args.ledger)
    records, errors = ledger.read_with_errors()
    summary = ledger.summary(records)
    summary["unreadable_lines"] = errors
    print(json.dumps(summary, indent=2))
    return 0


def _reconcile(args: argparse.Namespace) -> int:
    ledger = RunLedger(args.ledger)
    report = reconcile(ledger.read_all(), tolerance=args.tolerance)
    print(json.dumps(report.to_dict(), indent=2) if args.json else format_report(report))
    if args.fail_on_drift and report.flagged_models:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
