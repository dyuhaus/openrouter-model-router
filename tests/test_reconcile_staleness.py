"""The reconciliation report must be able to say the catalog is old.

The report already told operators "the catalog price is stale, run refresh" as a
diagnosis. It could not check that claim: nothing in the package could measure a
catalog's age. These tests hold the two halves together - the drift maths and
the age of the prices it is built on - and pin the exit codes, because the
failure this branch exists to kill is a gate that prints a pass over nothing.
"""

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from openrouter_model_router import ModelCatalog, ModelInfo
from openrouter_model_router.catalog import STALENESS_FRESH, STALENESS_STALE
from openrouter_model_router.cli import main
from openrouter_model_router.ledger import RunLedger, RunRecord
from openrouter_model_router.reconcile import STATUS_INSUFFICIENT_DATA, STATUS_OK, format_report, reconcile

DAY = 86_400.0


def stamp(seconds_ago):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds_ago))


def catalog_aged(seconds_ago):
    return ModelCatalog(
        [ModelInfo(id="a/b", context_length=100_000, input_cost_per_million=1.0, output_cost_per_million=3.0)],
        fetched_at=stamp(seconds_ago),
    )


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class ReportCarriesCatalogAgeTests(unittest.TestCase):
    def _rows(self):
        return [RunRecord(model="a/b", estimated_cost_usd=0.10, reported_cost_usd=0.101)]

    def test_a_fresh_catalog_leaves_the_report_ok(self):
        report = reconcile(self._rows(), catalog=catalog_aged(2 * DAY))

        self.assertEqual(report.status, STATUS_OK)
        self.assertEqual(report.catalog["status"], STALENESS_FRESH)
        self.assertFalse(report.catalog_is_stale)
        self.assertTrue(report.ok)

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_a_ninety_day_old_catalog_makes_the_report_not_ok(self):
        """Costs agreeing to the penny is not a clean bill of health when both
        numbers came from prices nobody has refreshed in three months."""

        report = reconcile(self._rows(), catalog=catalog_aged(90 * DAY))

        self.assertEqual(report.catalog["status"], STALENESS_STALE)
        self.assertTrue(report.catalog_is_stale)
        self.assertFalse(report.ok, "a stale catalog must not report ok")
        self.assertIn("CATALOG NOT FRESH (stale)", format_report(report))
        self.assertTrue(report.to_dict()["catalog_is_stale"])

    def test_no_catalog_is_reported_as_not_checked_never_as_fresh(self):
        report = reconcile(self._rows())

        self.assertEqual(report.catalog["status"], "not_checked")
        self.assertFalse(report.catalog["fresh"])
        self.assertFalse(report.catalog_is_stale)
        self.assertIn("NOT CHECKED", format_report(report))

    def test_an_empty_ledger_says_nothing_was_compared(self):
        report = reconcile([], catalog=catalog_aged(DAY))

        self.assertEqual(report.status, STATUS_INSUFFICIENT_DATA)
        self.assertFalse(report.ok)
        self.assertIn("NOTHING WAS COMPARED", format_report(report))


    def test_the_header_never_says_ok_when_the_report_is_not(self):
        """A header disagreeing with the exit code is how a reader believes the
        reassuring half. `status` is drift-only; `verdict` is the whole answer."""

        stale = reconcile(self._rows(), catalog=catalog_aged(90 * DAY))
        empty = reconcile([], catalog=catalog_aged(DAY))
        clean = reconcile(self._rows(), catalog=catalog_aged(DAY))

        self.assertEqual(stale.verdict, "stale_catalog")
        self.assertEqual(stale.status, STATUS_OK, "drift really is inside tolerance")
        self.assertTrue(format_report(stale).startswith("reconciliation: STALE_CATALOG"))
        self.assertEqual(empty.verdict, STATUS_INSUFFICIENT_DATA)
        self.assertEqual(clean.verdict, STATUS_OK)
        self.assertTrue(format_report(clean).startswith("reconciliation: OK"))


class ReconcileCliExitCodeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.ledger_path = self.dir / "runs.jsonl"
        self.catalog_path = self.dir / "catalog.json"

    def _write_ledger(self, *records):
        ledger = RunLedger(self.ledger_path)
        for record in records:
            ledger.append(record)

    def _write_catalog(self, seconds_ago):
        catalog_aged(seconds_ago).save(self.catalog_path)

    def _matching_run(self):
        return RunRecord(model="a/b", estimated_cost_usd=0.10, reported_cost_usd=0.101)

    def test_clean_ledger_and_fresh_catalog_exit_zero(self):
        self._write_ledger(self._matching_run())
        self._write_catalog(DAY)

        code, out, _ = run_cli(
            ["reconcile", "--ledger", str(self.ledger_path), "--catalog", str(self.catalog_path), "--fail-on-drift"]
        )

        self.assertEqual(code, 0)
        self.assertIn("catalog age         [ok]", out)

    # --- NEGATIVE CONTROLS ------------------------------------------------
    def test_stale_catalog_exits_non_zero_even_with_no_drift(self):
        self._write_ledger(self._matching_run())
        self._write_catalog(90 * DAY)

        code, out, err = run_cli(
            ["reconcile", "--ledger", str(self.ledger_path), "--catalog", str(self.catalog_path), "--fail-on-drift"]
        )

        self.assertEqual(code, 1)
        self.assertIn("CATALOG NOT FRESH (stale)", out)
        self.assertIn("catalog is stale", err)

    def test_an_empty_ledger_does_not_pass_the_gate(self):
        """Zero comparable runs is a configuration failure, not a pass. This is
        the exact shape the old exit-0 had: verified nothing, reported success."""

        self.ledger_path.write_text("")
        self._write_catalog(DAY)

        code, out, err = run_cli(
            ["reconcile", "--ledger", str(self.ledger_path), "--catalog", str(self.catalog_path), "--fail-on-drift"]
        )

        self.assertEqual(code, 1)
        self.assertIn("NOTHING WAS COMPARED", out)
        self.assertIn("compared 0 runs", err)

    def test_runs_without_a_reported_cost_are_not_a_pass_either(self):
        self._write_ledger(RunRecord(model="a/b", estimated_cost_usd=0.10))
        self._write_catalog(DAY)

        code, _, err = run_cli(
            ["reconcile", "--ledger", str(self.ledger_path), "--catalog", str(self.catalog_path), "--fail-on-drift"]
        )

        self.assertEqual(code, 1)
        self.assertIn("compared 0 runs", err)

    def test_a_missing_catalog_path_fails_rather_than_skipping_the_check(self):
        self._write_ledger(self._matching_run())

        code, _, err = run_cli(
            ["reconcile", "--ledger", str(self.ledger_path), "--catalog", str(self.dir / "nope.json")]
        )

        self.assertEqual(code, 1)
        self.assertIn("cannot age-check", err)

    def test_json_output_carries_the_catalog_block(self):
        self._write_ledger(self._matching_run())
        self._write_catalog(90 * DAY)

        code, out, _ = run_cli(
            ["reconcile", "--ledger", str(self.ledger_path), "--catalog", str(self.catalog_path), "--json"]
        )

        payload = json.loads(out)
        self.assertEqual(code, 0)  # no --fail-on-drift: reporting mode
        self.assertEqual(payload["catalog"]["status"], STALENESS_STALE)
        self.assertTrue(payload["catalog_is_stale"])
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
