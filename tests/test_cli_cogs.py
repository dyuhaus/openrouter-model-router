import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from openrouter_model_router.cli import main
from openrouter_model_router.ledger import STATUS_GATE_FAILED, RunLedger

from test_catalog_pricing import PAYLOAD
from openrouter_model_router import ModelCatalog


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CliCostTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.catalog_path = self.dir / "catalog.json"
        ModelCatalog.from_openrouter_payload(PAYLOAD).save(self.catalog_path)
        self.ledger_path = self.dir / "runs.jsonl"

    def test_estimate_prints_real_money_for_a_priced_model(self):
        code, out, _ = run_cli(
            [
                "estimate",
                "--catalog", str(self.catalog_path),
                "--input-tokens", "40000",
                "--output-tokens", "120000",
                "--model", "openai/gpt-4.1-mini",
            ]
        )

        self.assertEqual(code, 0)
        row = json.loads(out)["estimates"][0]
        # 40_000 * 0.15/1e6 + 120_000 * 0.60/1e6 = 0.006 + 0.072
        self.assertAlmostEqual(row["estimated_cost_usd"], 0.078)
        self.assertTrue(row["pricing_known"])

    def test_estimate_marks_an_unpriced_model_rather_than_quoting_zero(self):
        code, out, _ = run_cli(
            ["estimate", "--catalog", str(self.catalog_path), "--model", "openrouter/auto"]
        )

        row = json.loads(out)["estimates"][0]
        self.assertEqual(code, 0)
        self.assertEqual(row["estimated_cost_usd"], 0.0)
        self.assertFalse(row["pricing_known"], "a $0.00 quote must be labelled unpriced")

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_estimate_against_a_missing_catalog_fails_loudly(self):
        """The original bug was silent: no catalog, $0.00, exit 0."""

        code, _, err = run_cli(["estimate", "--catalog", str(self.dir / "nope.json")])

        self.assertEqual(code, 1)
        self.assertIn("run `openrouter-model-router refresh` first", err)

    def test_estimate_reports_an_unknown_model_id(self):
        code, out, _ = run_cli(
            ["estimate", "--catalog", str(self.catalog_path), "--model", "nobody/nothing"]
        )

        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["unknown_models"], ["nobody/nothing"])

    def test_ledger_command_summarizes_including_failures(self):
        ledger = RunLedger(self.ledger_path)
        ledger.record(model="a/b", reported_cost_usd=0.10)
        ledger.record(model="a/b", status=STATUS_GATE_FAILED, reported_cost_usd=0.09)

        code, out, _ = run_cli(["ledger", "--ledger", str(self.ledger_path)])
        summary = json.loads(out)

        self.assertEqual(code, 0)
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["retry_multiplier"], 2.0)

    def test_reconcile_command_reports_clean_when_costs_agree(self):
        ledger = RunLedger(self.ledger_path)
        ledger.record(model="a/b", estimated_cost_usd=0.10, reported_cost_usd=0.101)

        code, out, _ = run_cli(["reconcile", "--ledger", str(self.ledger_path), "--fail-on-drift"])

        self.assertEqual(code, 0)
        self.assertIn("OK", out)

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_reconcile_fail_on_drift_exits_nonzero_on_a_deliberate_mismatch(self):
        ledger = RunLedger(self.ledger_path)
        ledger.record(model="stale/model", estimated_cost_usd=0.10, reported_cost_usd=0.90)

        code, out, _ = run_cli(["reconcile", "--ledger", str(self.ledger_path), "--fail-on-drift"])

        self.assertEqual(code, 1, "a drift gate that never exits non-zero is not a gate")
        self.assertIn("DRIFT", out)
        self.assertIn("stale/model", out)

    def test_reconcile_json_output(self):
        ledger = RunLedger(self.ledger_path)
        ledger.record(model="stale/model", estimated_cost_usd=0.10, reported_cost_usd=0.90)

        code, out, _ = run_cli(["reconcile", "--ledger", str(self.ledger_path), "--json"])

        self.assertEqual(code, 0, "without --fail-on-drift the report is informational")
        self.assertEqual(json.loads(out)["flagged_models"], ["stale/model"])


if __name__ == "__main__":
    unittest.main()
