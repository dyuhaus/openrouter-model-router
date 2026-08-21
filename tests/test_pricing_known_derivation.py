"""A price-less record must never load as a confident $0.00.

Round 2 taught `pricing_known` to every consumer: the router, the budget filter,
the estimate command. Round 3's attacker walked past all of it by changing what
reached them. `ModelInfo.from_dict` defaulted the flag to ``True`` and read the
prices as ``float(x or 0.0)``, so a catalog record with no pricing at all -- or
with the ``-1`` sentinel written straight to disk -- came back off disk as
"price known: $0.00". Every hardened consumer downstream then did exactly what it
was told, confidently, about a number that was never measured.

Hardening the comparison is not hardening the gate. The question upstream of
"does this price pass?" is "is this a price at all?", and that answer has to come
from the data, never from a flag beside it.

Every case below is a control on that boundary: what the record actually
contains decides, and the safe direction is always UNKNOWN.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from openrouter_model_router import ModelCatalog, ModelInfo, ModelRouter, RunLedger, TaskSpec
from openrouter_model_router.cli import main
from openrouter_model_router.ledger import RunRecord
from openrouter_model_router.reconcile import VERDICT_UNRECONCILED, format_report, reconcile
from openrouter_model_router.types import derive_pricing_known

IN, OUT = 34_000, 81_000

#: 34_000 * 2.0/1e6 + 81_000 * 6.0/1e6
REAL_COST = 0.554


def record(model_id: str, **overrides) -> dict:
    """A catalog record as it would sit on disk, hand-written or hand-edited."""

    row = {
        "id": model_id,
        "name": model_id,
        "context_length": 1_000_000,
        "capabilities": ["text", "long_context"],
        "quality_score": 0.6,
        "speed_score": 0.6,
        "reliability_score": 0.65,
        "source": "hand-written",
    }
    row.update(overrides)
    return row


NO_PRICE = record("vendor/no-price-at-all")
NULL_PRICE = record("vendor/null-price", input_cost_per_million=None, output_cost_per_million=None)
SENTINEL = record("vendor/sentinel", input_cost_per_million=-1.0, output_cost_per_million=-1.0)
HALF_PRICE = record("vendor/half-price", input_cost_per_million=2.0)
REAL = record(
    "vendor/real-price",
    input_cost_per_million=2.0,
    output_cost_per_million=6.0,
    pricing_known=True,
)
DECLARED_FREE = record(
    "vendor/genuinely-free",
    input_cost_per_million=0.0,
    output_cost_per_million=0.0,
    pricing_known=True,
)


class FromDictDerivesPricingTests(unittest.TestCase):
    """`from_dict` re-derives the flag from the prices. It does not trust it."""

    def test_a_record_with_no_pricing_at_all_loads_as_unknown(self):
        """THE REPRODUCTION. Before the fix this loaded pricing_known=True and
        `cost_estimate` returned 0.0 -- a confident zero over nothing."""

        model = ModelInfo.from_dict(NO_PRICE)

        self.assertFalse(model.pricing_known)
        self.assertFalse(model.is_free, "absent is not free")
        self.assertIsNone(model.cost_estimate(IN, OUT), "an absent price must not quote a number")

    def test_null_prices_load_as_unknown(self):
        model = ModelInfo.from_dict(NULL_PRICE)

        self.assertFalse(model.pricing_known)
        self.assertIsNone(model.cost_estimate(IN, OUT))

    def test_the_minus_one_sentinel_written_to_disk_loads_as_unknown(self):
        model = ModelInfo.from_dict(SENTINEL)

        self.assertFalse(model.pricing_known)
        self.assertIsNone(model.cost_estimate(IN, OUT))
        self.assertEqual(
            (model.input_cost_per_million, model.output_cost_per_million),
            (0.0, 0.0),
            "the sentinel is clamped for arithmetic, but never republished as a price",
        )

    def test_one_published_side_is_not_a_price(self):
        """A prompt price with no completion price cannot cost a request: the
        missing half would contribute $0.00 to every estimate."""

        model = ModelInfo.from_dict(HALF_PRICE)

        self.assertFalse(model.pricing_known)
        self.assertIsNone(model.cost_estimate(IN, OUT))

    def test_a_true_flag_cannot_conjure_a_price_that_is_not_there(self):
        """The exact shape of the bug: a flag on disk outranking the data.

        `pricing_known: true` beside no prices at all is a claim with no
        evidence, and the safe reading of it is UNKNOWN.
        """

        model = ModelInfo.from_dict(record("vendor/liar", pricing_known=True))

        self.assertFalse(model.pricing_known)
        self.assertIsNone(model.cost_estimate(IN, OUT))

    def test_a_true_flag_beside_the_sentinel_is_also_unknown(self):
        model = ModelInfo.from_dict(
            record(
                "vendor/liar-sentinel",
                input_cost_per_million=-1.0,
                output_cost_per_million=-1.0,
                pricing_known=True,
            )
        )

        self.assertFalse(model.pricing_known)

    def test_a_false_flag_is_never_overridden_upward(self):
        """Whoever wrote False knew something the numbers do not show."""

        model = ModelInfo.from_dict(
            record(
                "vendor/known-unpriceable",
                input_cost_per_million=2.0,
                output_cost_per_million=6.0,
                pricing_known=False,
            )
        )

        self.assertFalse(model.pricing_known)
        self.assertIsNone(model.cost_estimate(IN, OUT))

    def test_undeclared_double_zero_is_unknown_not_free(self):
        """Two undeclared zeros are what the OLD clamp wrote for a `-1` model.

        A schema_version 1 catalog is full of exactly this shape, so reading it
        as "free" would resurrect the original bug from every old file on disk.
        A declared zero (below) still means free.
        """

        model = ModelInfo.from_dict(
            record("vendor/ambiguous-zero", input_cost_per_million=0.0, output_cost_per_million=0.0)
        )

        self.assertFalse(model.pricing_known)
        self.assertFalse(model.is_free)

    def test_junk_in_a_price_field_is_unknown_rather_than_a_number(self):
        model = ModelInfo.from_dict(
            record("vendor/junk", input_cost_per_million="not-a-number", output_cost_per_million="1e")
        )

        self.assertFalse(model.pricing_known)
        self.assertEqual(model.input_cost_per_million, 0.0)

    # --- POSITIVE CONTROLS: the derivation must still say YES ---------------
    def test_a_real_record_still_prices_at_its_real_cost(self):
        model = ModelInfo.from_dict(REAL)

        self.assertTrue(model.pricing_known)
        self.assertAlmostEqual(model.cost_estimate(IN, OUT), REAL_COST)

    def test_a_declared_zero_price_is_still_free_and_still_known(self):
        model = ModelInfo.from_dict(DECLARED_FREE)

        self.assertTrue(model.pricing_known)
        self.assertTrue(model.is_free)
        self.assertEqual(model.cost_estimate(IN, OUT), 0.0)

    def test_an_undeclared_priced_record_is_known_without_any_flag(self):
        model = ModelInfo.from_dict(
            record("vendor/unflagged", input_cost_per_million=2.0, output_cost_per_million=6.0)
        )

        self.assertTrue(model.pricing_known)
        self.assertAlmostEqual(model.cost_estimate(IN, OUT), REAL_COST)


class ConstructorDerivesPricingTests(unittest.TestCase):
    def test_a_model_built_with_no_prices_at_all_is_unknown(self):
        """Same hole, reached through the constructor instead of a JSON file."""

        model = ModelInfo(id="vendor/bare", context_length=128_000)

        self.assertFalse(model.pricing_known)
        self.assertIsNone(model.cost_estimate(IN, OUT))

    def test_an_explicit_free_model_still_says_free(self):
        model = ModelInfo(id="vendor/free", input_cost_per_million=0.0, output_cost_per_million=0.0, pricing_known=True)

        self.assertTrue(model.pricing_known)
        self.assertTrue(model.is_free)

    def test_derivation_helper_is_directly_testable(self):
        self.assertFalse(derive_pricing_known(None, None))
        self.assertFalse(derive_pricing_known(-1, -1))
        self.assertFalse(derive_pricing_known(2.0, None))
        self.assertFalse(derive_pricing_known(None, None, True))
        self.assertFalse(derive_pricing_known(2.0, 6.0, False))
        self.assertFalse(derive_pricing_known(0.0, 0.0))
        self.assertTrue(derive_pricing_known(0.0, 0.0, True))
        self.assertTrue(derive_pricing_known(2.0, 6.0))


class CatalogRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "catalog.json"
        self.path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "updated_at": "2026-08-21T09:00:00Z",
                    "fetched_at": "2026-08-21T09:00:00Z",
                    "models": [NO_PRICE, SENTINEL, REAL],
                },
                indent=2,
            )
        )

    def _run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_loading_the_file_marks_two_of_three_unpriceable(self):
        catalog = ModelCatalog.load(self.path, bootstrap=False)

        coverage = catalog.pricing_coverage()
        self.assertEqual(coverage["total"], 3)
        self.assertEqual(coverage["priced"], 1)
        self.assertEqual(coverage["free"], 0, "an absent price is not a free price")
        self.assertEqual(coverage["pricing_unknown"], 2)
        self.assertEqual(coverage["unaccounted"], 0)

    def test_a_save_and_reload_does_not_launder_unknown_back_into_known(self):
        """The flag survives a round trip in the safe direction only."""

        catalog = ModelCatalog.load(self.path, bootstrap=False)
        rewritten = Path(self._tmp.name) / "again.json"
        catalog.save(rewritten)

        reloaded = ModelCatalog.load(rewritten, bootstrap=False)

        self.assertFalse(reloaded.get("vendor/no-price-at-all").pricing_known)
        self.assertFalse(reloaded.get("vendor/sentinel").pricing_known)
        self.assertTrue(reloaded.get("vendor/real-price").pricing_known)

    # --- NEGATIVE CONTROL ---------------------------------------------------
    def test_estimate_prints_null_for_the_unpriced_records_off_disk(self):
        code, out, err = self._run_cli(
            [
                "estimate",
                "--catalog",
                str(self.path),
                "--input-tokens",
                str(IN),
                "--output-tokens",
                str(OUT),
                "--model",
                "vendor/no-price-at-all",
                "--model",
                "vendor/sentinel",
            ]
        )
        payload = json.loads(out)

        self.assertEqual([row["pricing_status"] for row in payload["estimates"]], ["UNKNOWN", "UNKNOWN"])
        self.assertEqual([row["estimated_cost_usd"] for row in payload["estimates"]], [None, None])
        self.assertEqual(payload["priced_estimates"], 0)
        self.assertEqual(code, 1, "a command that produced no cost estimate has not succeeded")
        self.assertIn("not $0.00", err)

    # --- POSITIVE CONTROL ---------------------------------------------------
    def test_estimate_still_prints_real_money_for_the_real_record(self):
        code, out, _ = self._run_cli(
            [
                "estimate",
                "--catalog",
                str(self.path),
                "--input-tokens",
                str(IN),
                "--output-tokens",
                str(OUT),
                "--model",
                "vendor/real-price",
            ]
        )
        row = json.loads(out)["estimates"][0]

        self.assertEqual(code, 0)
        self.assertAlmostEqual(row["estimated_cost_usd"], REAL_COST)
        self.assertEqual(row["pricing_status"], "known")


class CheapestSelectionTests(unittest.TestCase):
    """A model nobody can price must not win a "cheapest" comparison."""

    def _catalog(self):
        return ModelCatalog([ModelInfo.from_dict(row) for row in (NO_PRICE, SENTINEL, REAL)])

    def test_the_cheap_route_picks_the_priced_model(self):
        selection = ModelRouter(self._catalog()).select(
            TaskSpec(preference="cheap", input_tokens=IN, output_tokens=OUT)
        )

        self.assertEqual(selection.model_id, "vendor/real-price")
        self.assertTrue(selection.estimated_cost_is_known)
        self.assertAlmostEqual(selection.cost_estimate_usd, REAL_COST)

    # --- POSITIVE CONTROL: the exclusion is driven by the price, not by luck -
    def test_the_same_model_wins_the_cheap_route_once_it_carries_a_price(self):
        """Give the loser a real, cheaper price and it must win immediately.

        Without this, "the unpriced model lost" could mean the scoring simply
        never favoured it, and the guard would be indistinguishable from nothing.
        """

        cheaper = dict(NO_PRICE)
        cheaper.update(input_cost_per_million=0.1, output_cost_per_million=0.2, pricing_known=True)
        catalog = ModelCatalog([ModelInfo.from_dict(row) for row in (cheaper, SENTINEL, REAL)])

        selection = ModelRouter(catalog).select(TaskSpec(preference="cheap", input_tokens=IN, output_tokens=OUT))

        self.assertEqual(selection.model_id, "vendor/no-price-at-all")
        self.assertAlmostEqual(selection.cost_estimate_usd, 34_000 * 0.1 / 1e6 + 81_000 * 0.2 / 1e6)

    def test_a_budget_ceiling_excludes_the_unpriced_records(self):
        router = ModelRouter(self._catalog())

        with self.assertRaises(ValueError):
            # $0.10 admits nothing: the real model costs $0.554, and the two
            # unpriced ones are excluded outright rather than slipping under on
            # a $0.00 estimate.
            router.select(TaskSpec(preference="cheap", input_tokens=IN, output_tokens=OUT, max_cost_usd=0.10))

    def test_selection_serializes_an_unknown_cost_as_null(self):
        selection = ModelRouter(self._catalog()).select(
            TaskSpec(preference="quality", allow_models=("vendor/sentinel",), input_tokens=IN, output_tokens=OUT)
        )
        payload = selection.to_dict()

        self.assertIsNone(payload["cost_estimate_usd"])
        self.assertEqual(payload["pricing_status"], "UNKNOWN")
        self.assertFalse(payload["estimated_cost_is_known"])


class UnreconciledSpendTests(unittest.TestCase):
    """Money charged on a run with no estimate may not vanish into an OK report."""

    def _catalog(self):
        return ModelCatalog(
            [ModelInfo.from_dict(REAL), ModelInfo.from_dict(NO_PRICE)],
            fetched_at="2026-08-21T09:00:00Z",
        )

    # --- NEGATIVE CONTROL ---------------------------------------------------
    def test_real_spend_on_an_unpriced_model_fails_the_report(self):
        report = reconcile(
            [
                RunRecord(model="vendor/real-price", estimated_cost_usd=0.554, reported_cost_usd=0.556),
                RunRecord(model="vendor/no-price-at-all", estimated_cost_usd=None, reported_cost_usd=0.42),
            ],
            catalog=self._catalog(),
        )

        self.assertAlmostEqual(report.unreconciled_reported_cost_usd, 0.42)
        self.assertTrue(report.has_unreconciled_cost)
        self.assertEqual(report.verdict, VERDICT_UNRECONCILED)
        self.assertFalse(report.ok, "$0.42 nobody compared to anything is not a clean bill of health")
        self.assertEqual(report.pricing_unknown_models, ["vendor/no-price-at-all"])

        text = format_report(report)
        self.assertIn("UNPRICED MODELS IN THIS LEDGER", text)
        self.assertIn("catalog price UNKNOWN", text)
        self.assertIn("$0.420000 WAS CHARGED", text)

    # --- POSITIVE CONTROL ---------------------------------------------------
    def test_the_same_ledger_reconciles_clean_once_every_run_carries_an_estimate(self):
        report = reconcile(
            [
                RunRecord(model="vendor/real-price", estimated_cost_usd=0.554, reported_cost_usd=0.556),
                RunRecord(model="vendor/real-price", estimated_cost_usd=0.420, reported_cost_usd=0.421),
            ],
            catalog=self._catalog(),
        )

        self.assertEqual(report.unreconciled_reported_cost_usd, 0.0)
        self.assertFalse(report.has_unreconciled_cost)
        self.assertTrue(report.ok)
        self.assertEqual(report.pricing_unknown_models, [])

    def test_the_cli_exits_nonzero_on_unreconciled_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "runs.jsonl"
            ledger = RunLedger(ledger_path)
            ledger.record(model="vendor/real-price", estimated_cost_usd=0.554, reported_cost_usd=0.556)
            ledger.record(model="vendor/no-price-at-all", reported_cost_usd=0.42)

            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["reconcile", "--ledger", str(ledger_path), "--fail-on-drift"])

        self.assertEqual(code, 1)
        self.assertIn("charged on 1 run(s) with no estimate", err.getvalue())

    def test_the_ledger_summary_says_how_much_of_the_spend_it_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = RunLedger(Path(tmp) / "runs.jsonl")
            ledger.record(model="vendor/real-price", estimated_cost_usd=0.554, reported_cost_usd=0.556)
            ledger.record(model="vendor/no-price-at-all", reported_cost_usd=0.42)

            summary = ledger.summary()

        self.assertEqual(summary["runs_missing_estimate"], 1)
        self.assertFalse(summary["estimated_cost_is_complete"])
        self.assertIn("1 carried no estimate", summary["estimated_cost_note"])
        self.assertEqual(summary["by_model"]["vendor/no-price-at-all"]["runs_missing_estimate"], 1)


if __name__ == "__main__":
    unittest.main()
