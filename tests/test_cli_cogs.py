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

    def test_estimate_reports_an_unpriced_model_as_unknown_not_as_zero(self):
        """UNKNOWN and $0.00 are different facts and only one is safe to budget on.

        Tightened from an earlier version of this test, which accepted
        `estimated_cost_usd == 0.0` with a `pricing_known: false` flag beside it.
        A reader (or a script summing the column) sees the number, not the flag,
        and 0.0 is the cheapest possible number. The estimate is now null, and
        the command exits non-zero because it produced no cost estimate at all.
        """

        code, out, err = run_cli(
            ["estimate", "--catalog", str(self.catalog_path), "--model", "openrouter/auto"]
        )

        payload = json.loads(out)
        row = payload["estimates"][0]
        self.assertIsNone(row["estimated_cost_usd"], "an unknown price must not print as a number")
        self.assertEqual(row["pricing_status"], "UNKNOWN")
        self.assertFalse(row["pricing_known"])
        self.assertEqual(payload["priced_estimates"], 0)
        self.assertEqual(code, 1, "an estimate command that priced 0 models must not exit 0")
        self.assertIn("not $0.00", err)

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


class RefreshCoverageGateTests(unittest.TestCase):
    """`refresh` must not report success after writing a catalog nothing can be
    costed from. That state is silent otherwise: exit 0, a file on disk, and
    every downstream estimate $0.00."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "catalog.json"

    @staticmethod
    def _payload(pricing):
        return {"data": [{"id": "vendor/model", "context_length": 100_000, **({"pricing": pricing} if pricing else {})}]}

    def _refresh_with(self, payload):
        from unittest import mock

        from openrouter_model_router import ModelCatalog

        built = ModelCatalog.from_openrouter_payload(payload)
        with mock.patch.object(ModelCatalog, "refresh_from_openrouter", return_value=built):
            return run_cli(["refresh", "--catalog", str(self.path)])

    def test_refresh_reports_pricing_coverage(self):
        code, out, _ = self._refresh_with(self._payload({"prompt": "0.000001", "completion": "0.000002"}))

        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["models"], 1)
        self.assertEqual(report["pricing"]["priced"], 1)
        self.assertEqual(report["pricing"]["pricing_unknown"], 0)
        self.assertFalse(report["authenticated"])

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_refresh_exits_nonzero_when_nothing_can_be_priced(self):
        code, out, err = self._refresh_with(self._payload({"prompt": "-1", "completion": "-1"}))

        self.assertEqual(code, 1, "a refresh that cannot price anything must not report success")
        self.assertEqual(json.loads(out)["pricing"]["priced"], 0)
        self.assertIn("$0.00", err)

    def test_refresh_with_no_pricing_block_at_all_also_fails(self):
        code, _, err = self._refresh_with(self._payload(None))

        self.assertEqual(code, 1)
        self.assertIn("NO model carries a usable price", err)
