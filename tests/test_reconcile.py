import unittest

from openrouter_model_router.ledger import RunRecord
from openrouter_model_router.reconcile import (
    STATUS_DRIFT,
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
    format_report,
    reconcile,
)


def _run(model="a/b", estimated=None, reported=None):
    return RunRecord(model=model, estimated_cost_usd=estimated, reported_cost_usd=reported)


class ReconcileTests(unittest.TestCase):
    def test_matching_costs_reconcile_clean(self):
        report = reconcile([_run(estimated=0.10, reported=0.101), _run(estimated=0.20, reported=0.199)])

        self.assertEqual(report.status, STATUS_OK)
        self.assertTrue(report.ok)
        self.assertEqual(report.flagged_models, [])
        self.assertEqual(report.comparable_runs, 2)

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_deliberate_mismatch_is_flagged(self):
        """Estimate says $0.10, provider charged $0.40. A reconciler that does
        not raise its hand here is doing nothing at all."""

        report = reconcile([_run(model="stale/model", estimated=0.10, reported=0.40)])

        self.assertEqual(report.status, STATUS_DRIFT)
        self.assertFalse(report.ok)
        self.assertEqual([m.model for m in report.flagged_models], ["stale/model"])

        drift = report.flagged_models[0]
        self.assertAlmostEqual(drift.absolute_drift_usd, 0.30)
        self.assertAlmostEqual(drift.relative_drift, 3.0)
        self.assertIn("refresh the catalog", drift.reason)
        self.assertIn("FLAG", format_report(report))

    def test_zero_estimate_against_a_real_charge_is_flagged(self):
        """The exact bug this repo shipped: a $0.00 estimate that cost money."""

        report = reconcile([_run(model="unpriced/model", estimated=0.0, reported=0.42)])

        self.assertEqual(report.status, STATUS_DRIFT)
        self.assertIn("no price for this model", report.flagged_models[0].reason)

    def test_drift_inside_tolerance_is_not_flagged(self):
        report = reconcile([_run(estimated=1.00, reported=1.05)], tolerance=0.10)

        self.assertEqual(report.status, STATUS_OK)

    def test_tolerance_boundary_is_respected(self):
        loose = reconcile([_run(estimated=1.00, reported=1.20)], tolerance=0.25)
        tight = reconcile([_run(estimated=1.00, reported=1.20)], tolerance=0.05)

        self.assertEqual(loose.status, STATUS_OK)
        self.assertEqual(tight.status, STATUS_DRIFT)

    def test_missing_reported_cost_is_counted_not_treated_as_zero(self):
        report = reconcile([_run(estimated=0.50, reported=None)])

        self.assertEqual(report.status, STATUS_INSUFFICIENT_DATA)
        self.assertEqual(report.missing_reported, 1)
        self.assertEqual(report.comparable_runs, 0)
        self.assertEqual(report.estimated_cost_usd, 0.0)
        self.assertEqual(report.flagged_models, [])

    def test_missing_estimate_is_counted_separately(self):
        report = reconcile([_run(estimated=None, reported=0.50)])

        self.assertEqual(report.missing_estimate, 1)
        self.assertEqual(report.status, STATUS_INSUFFICIENT_DATA)

    def test_empty_ledger_is_insufficient_data_not_ok(self):
        report = reconcile([])

        self.assertEqual(report.status, STATUS_INSUFFICIENT_DATA)
        self.assertFalse(report.ok)

    def test_sub_cent_drift_is_below_the_floor(self):
        report = reconcile([_run(estimated=0.0001, reported=0.0003)])

        self.assertEqual(report.status, STATUS_OK)
        self.assertIn("below", report.models[0].reason)

    def test_per_model_isolation(self):
        report = reconcile(
            [
                _run(model="good/model", estimated=1.0, reported=1.0),
                _run(model="stale/model", estimated=1.0, reported=5.0),
            ]
        )

        self.assertEqual([m.model for m in report.flagged_models], ["stale/model"])
        self.assertEqual(report.status, STATUS_DRIFT)


if __name__ == "__main__":
    unittest.main()
