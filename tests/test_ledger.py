import json
import tempfile
import unittest
from pathlib import Path

from openrouter_model_router.ledger import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_GATE_FAILED,
    RunLedger,
    RunRecord,
)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "runs.jsonl"
        self.ledger = RunLedger(self.path)

    def test_append_writes_one_json_line_per_run(self):
        self.ledger.record(model="a/b", task_label="lesson-1", prompt_tokens=10, completion_tokens=5)
        self.ledger.record(model="a/b", task_label="lesson-2", prompt_tokens=20, completion_tokens=9)

        lines = self.path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["model"], "a/b")
        self.assertEqual(first["task_label"], "lesson-1")
        self.assertTrue(first["timestamp"])
        self.assertTrue(first["run_id"])
        self.assertEqual(len(self.ledger.read_all()), 2)

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_failed_runs_are_recorded_not_dropped(self):
        """A ledger that silently drops failures can only ever report a 1.0
        retry multiplier. Both failure kinds must survive a write/read cycle."""

        self.ledger.record(model="a/b", task_label="ok", status=STATUS_COMPLETED, reported_cost_usd=0.10)
        self.ledger.record(
            model="a/b",
            task_label="rejected",
            status=STATUS_GATE_FAILED,
            gate_failures=("fabricated source", "answer index out of range"),
            reported_cost_usd=0.09,
        )
        self.ledger.record(
            model="a/b",
            task_label="exploded",
            status=STATUS_ERROR,
            error="OpenRouterError: HTTP 502",
            estimated_cost_usd=0.08,
        )

        rows = self.ledger.read_all()
        self.assertEqual(len(rows), 3, "all three attempts must be on disk")

        statuses = [r.status for r in rows]
        self.assertIn(STATUS_GATE_FAILED, statuses)
        self.assertIn(STATUS_ERROR, statuses)

        gate_failed = next(r for r in rows if r.status == STATUS_GATE_FAILED)
        self.assertFalse(gate_failed.succeeded)
        self.assertIs(gate_failed.gates_passed, False)
        self.assertEqual(gate_failed.gate_failures, ("fabricated source", "answer index out of range"))

        errored = next(r for r in rows if r.status == STATUS_ERROR)
        self.assertFalse(errored.succeeded)
        self.assertIn("502", errored.error)

        summary = self.ledger.summary()
        self.assertEqual(summary["runs"], 3)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["gate_failed"], 1)
        self.assertEqual(summary["errored"], 1)

    def test_retry_multiplier_is_measured_from_attempts_over_accepted(self):
        for _ in range(3):
            self.ledger.record(model="a/b", status=STATUS_GATE_FAILED)
        self.ledger.record(model="a/b", status=STATUS_COMPLETED)

        summary = self.ledger.summary()

        self.assertTrue(summary["retry_multiplier_measured"])
        self.assertEqual(summary["retry_multiplier"], 4.0)
        self.assertNotEqual(summary["retry_multiplier"], 1.35, "must be measured, not the guess")

    def test_retry_multiplier_refuses_to_invent_a_number_without_data(self):
        self.ledger.record(model="a/b", status=STATUS_ERROR, error="boom")

        summary = self.ledger.summary()

        self.assertIsNone(summary["retry_multiplier"])
        self.assertFalse(summary["retry_multiplier_measured"])
        self.assertIn("NOT measurable", summary["retry_multiplier_note"])

    def test_cost_of_failed_runs_is_tracked(self):
        self.ledger.record(model="a/b", status=STATUS_COMPLETED, reported_cost_usd=1.0)
        self.ledger.record(model="a/b", status=STATUS_GATE_FAILED, reported_cost_usd=0.5)
        self.ledger.record(model="a/b", status=STATUS_ERROR, estimated_cost_usd=0.25)

        summary = self.ledger.summary()

        self.assertAlmostEqual(summary["cost_of_failed_runs_usd"], 0.75)

    def test_token_totals_are_integers(self):
        self.ledger.record(model="a/b", prompt_tokens=8000, completion_tokens=2000)
        self.ledger.record(model="a/b", prompt_tokens=8000, completion_tokens=2000)

        summary = self.ledger.summary()

        self.assertEqual(summary["prompt_tokens"], 16000)
        self.assertIsInstance(summary["prompt_tokens"], int)
        self.assertIsInstance(summary["completion_tokens"], int)

    def test_corrupt_line_is_reported_not_fatal(self):
        self.ledger.record(model="a/b")
        with self.path.open("a") as handle:
            handle.write("{not json\n")
        self.ledger.record(model="a/b")

        rows, errors = self.ledger.read_with_errors()

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("line 2", errors[0])

    def test_status_must_be_valid(self):
        with self.assertRaises(ValueError):
            RunRecord(model="a/b", status="probably-fine")

    def test_missing_ledger_file_reads_as_empty(self):
        empty = RunLedger(Path(self._tmp.name) / "nope.jsonl")
        self.assertEqual(empty.read_all(), [])
        self.assertEqual(empty.summary()["runs"], 0)


if __name__ == "__main__":
    unittest.main()
