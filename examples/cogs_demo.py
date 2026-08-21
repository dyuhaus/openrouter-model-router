#!/usr/bin/env python3
"""End-to-end COGS demo. Runs today, with no API key and no network.

Shows the whole measurement chain:

  1. real catalog prices  ->  a cost ESTIMATE that is not $0.00
  2. a model call         ->  usage read back off the response (REPORTED cost)
  3. every attempt        ->  one ledger row, successes and failures alike
  4. estimate vs reported ->  a reconciliation report that flags drift

The model calls go through :class:`FakeTransport`, so the only thing simulated
here is the provider. Everything else -- prices, arithmetic, ledger, drift
detection -- is the production code path.

Usage:
    PYTHONPATH=src python3 examples/cogs_demo.py [--catalog PATH]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

from openrouter_model_router import (
    FakeTransport,
    ModelCatalog,
    ModelRouter,
    OpenRouterClient,
    RunLedger,
    TaskSpec,
)
from openrouter_model_router.reconcile import format_report, reconcile
from openrouter_model_router.transport import HttpResponse

MODEL = "openai/gpt-4.1-mini"


def _response(prompt_tokens: int, completion_tokens: int, cost: float, content: str) -> dict:
    """A response shaped like OpenRouter's documented usage-accounting payload."""

    return {
        "id": "gen-demo",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
            "is_byok": False,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


def sources_gate(result) -> list[str]:
    """A stand-in content gate. Returns the reasons it rejected the output."""

    failures = []
    if "example.invalid" in result.content:
        failures.append("fabricated source: example.invalid")
    if not result.content.strip():
        failures.append("empty completion")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=None, help="Catalog path (default: the refreshed cache)")
    args = parser.parse_args(argv)

    catalog = ModelCatalog.load(args.catalog, bootstrap=True)
    coverage = catalog.pricing_coverage()
    print(f"catalog: {len(catalog)} models, {coverage['priced']} priced, "
          f"{coverage['pricing_unknown']} with unknown pricing\n")
    if coverage["priced"] == 0:
        print("No priced models. Run `openrouter-model-router refresh` first.", file=sys.stderr)
        return 1
    if catalog.get(MODEL) is None:
        print(f"{MODEL} is not in this catalog; refresh it.", file=sys.stderr)
        return 1

    router = ModelRouter(catalog)
    task = TaskSpec(task_type="general", input_tokens=8_000, output_tokens=2_000, allow_models=(MODEL,))

    selection = router.select(task)
    print(f"1. ESTIMATE  {selection.model_id}: ${selection.estimated_cost_usd:.6f} "
          f"for {task.input_tokens} in / {task.output_tokens} out "
          f"(price known: {selection.estimated_cost_is_known})\n")

    # Three attempts at one unit of work: rejected, provider error, accepted.
    # The middle response is deliberately overcharged to trigger drift.
    good = _response(8_000, 2_000, 0.0064, "Espresso extraction runs 25-30 seconds.")
    overcharged = copy.deepcopy(good)
    overcharged["usage"]["cost"] = 0.0260
    rejected = _response(8_000, 2_000, 0.0064, "See https://example.invalid/does-not-exist for the study.")

    transport = FakeTransport(
        [
            FakeTransport.json_response(rejected),
            HttpResponse(status=502, body=b"upstream unavailable"),
            FakeTransport.json_response(overcharged),
        ]
    )
    client = OpenRouterClient(api_key="sk-demo-fake-key", transport=transport)

    with tempfile.TemporaryDirectory() as tmp:
        ledger = RunLedger(Path(tmp) / "runs.jsonl")
        messages = [{"role": "user", "content": "Write lesson 3."}]

        print("2. CALLS")
        for attempt in (1, 2, 3):
            outcome = router.run(
                client=client,
                messages=messages,
                task=task,
                ledger=ledger,
                task_label="lesson-3",
                gate=sources_gate,
                attempt=attempt,
            )
            r = outcome.record
            reported = "n/a" if r.reported_cost_usd is None else f"${r.reported_cost_usd:.6f}"
            detail = r.error or (", ".join(r.gate_failures) or "accepted")
            print(f"   attempt {attempt}: {r.status:<12} reported={reported:<12} {detail}")

        print("\n3. LEDGER")
        print(json.dumps(ledger.summary(), indent=2))

        print("\n4. RECONCILIATION")
        report = reconcile(ledger.read_all())
        print(format_report(report))

        return 1 if report.flagged_models else 0


if __name__ == "__main__":
    raise SystemExit(main())
