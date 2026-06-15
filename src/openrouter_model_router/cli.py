"""Command line interface for OpenRouter Model Router."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .catalog import OPENROUTER_BASE_URL, CatalogRefreshError, ModelCatalog, default_catalog_path
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

    args = parser.parse_args(argv)
    if args.command in {"refresh", "research", "update"}:
        return _refresh(args)
    if args.command == "select":
        return _select(args)
    if args.command == "record-outcome":
        return _record_outcome(args)
    parser.error(f"unknown command: {args.command}")
    return 2


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
    print(json.dumps({"catalog": str(saved), "models": len(current), "added": added, "updated": updated}, indent=2))
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


if __name__ == "__main__":
    raise SystemExit(main())
