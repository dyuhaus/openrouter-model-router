"""An unknown price must never behave like a price of zero.

OpenRouter publishes ``-1`` for prompt/completion on models whose real price
depends on where the request is routed. An earlier bug clamped that to 0.0, which
is not merely inaccurate: zero is the *best possible* number in every cost
comparison, so the one model nobody can price wins the cheap route, passes every
budget ceiling, and adds $0.00 to the spend estimate while the bill arrives
anyway.

The clamp itself is fine - the arithmetic needs a number - but `pricing_known`
must gate every place the number is used to make a decision or shown to a human.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from openrouter_model_router import ModelCatalog, ModelInfo, ModelRouter, TaskSpec
from openrouter_model_router.catalog import model_from_openrouter
from openrouter_model_router.cli import main


def _catalog():
    return ModelCatalog(
        [
            ModelInfo(
                id="priced/known",
                context_length=1_000_000,
                input_cost_per_million=2.0,
                output_cost_per_million=6.0,
                capabilities=("text", "long_context"),
                quality_score=0.6,
                speed_score=0.6,
            ),
            ModelInfo(
                id="meta/unpriced",
                context_length=1_000_000,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
                pricing_known=False,
                capabilities=("text", "long_context"),
                quality_score=0.6,
                speed_score=0.6,
            ),
        ]
    )


class SentinelTests(unittest.TestCase):
    def test_the_minus_one_sentinel_is_unknown_and_not_free(self):
        model = model_from_openrouter(
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}, "context_length": 128_000}
        )

        self.assertFalse(model.pricing_known)
        self.assertFalse(model.is_free)
        self.assertIsNone(model.cost_estimate(34_000, 81_000), "an unknown price must estimate as None, not 0.0")
        self.assertEqual(model.estimated_cost_usd(34_000, 81_000), 0.0, "the raw arithmetic still returns a number")

    def test_one_sided_sentinel_is_also_unknown(self):
        """A published prompt price and a -1 completion price is still unpriceable."""

        model = model_from_openrouter(
            {"id": "vendor/half", "pricing": {"prompt": "0.000001", "completion": "-1"}, "context_length": 8_000}
        )

        self.assertFalse(model.pricing_known)
        self.assertIsNone(model.cost_estimate(1000, 1000))

    def test_a_real_zero_price_still_estimates_as_zero_dollars(self):
        """Free is a measurement. Unknown is not. They must stay distinguishable."""

        model = model_from_openrouter(
            {"id": "vendor/x:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 8_000}
        )

        self.assertTrue(model.pricing_known)
        self.assertEqual(model.cost_estimate(34_000, 81_000), 0.0)


class RouterUnknownPriceTests(unittest.TestCase):
    def test_an_unpriced_model_does_not_win_the_cheap_route(self):
        selection = ModelRouter(_catalog()).select(
            TaskSpec(preference="cheap", input_tokens=34_000, output_tokens=81_000)
        )

        self.assertEqual(selection.model.id, "priced/known")

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_an_unpriced_model_is_excluded_by_a_budget_ceiling(self):
        """A $0.01 ceiling that admits a model of unbounded price is not a ceiling."""

        router = ModelRouter(_catalog())

        selection = router.select(
            TaskSpec(
                preference="cheap",
                input_tokens=34_000,
                output_tokens=81_000,
                max_cost_usd=0.01,
                allow_models=("meta/unpriced",),
                fallback_model="meta/unpriced",
            )
        )

        # It survives only through the explicit fallback, which is a documented
        # override, and the selection still declares its estimate unknowable.
        self.assertEqual(selection.candidates_considered, 0)
        self.assertFalse(selection.estimated_cost_is_known)

    def test_the_unpriced_model_is_still_selectable_when_no_budget_is_set(self):
        selection = ModelRouter(_catalog()).select(
            TaskSpec(preference="quality", allow_models=("meta/unpriced",), input_tokens=100, output_tokens=100)
        )

        self.assertEqual(selection.model.id, "meta/unpriced")
        self.assertFalse(selection.estimated_cost_is_known)
        self.assertTrue(any("pricing_unknown" in reason for reason in selection.reasons))


class EstimateCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "catalog.json"
        _catalog().save(self.path)

    def _estimate(self, *models):
        argv = ["estimate", "--catalog", str(self.path), "--input-tokens", "34000", "--output-tokens", "81000"]
        for model in models:
            argv += ["--model", model]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, json.loads(out.getvalue()), err.getvalue()

    def test_a_priced_model_prints_real_money(self):
        code, payload, _ = self._estimate("priced/known")

        row = payload["estimates"][0]
        # 34_000 * 2.0/1e6 + 81_000 * 6.0/1e6 = 0.068 + 0.486
        self.assertAlmostEqual(row["estimated_cost_usd"], 0.554)
        self.assertEqual(row["pricing_status"], "known")
        self.assertEqual(code, 0)

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_an_unpriced_model_prints_null_and_fails_the_command(self):
        code, payload, err = self._estimate("meta/unpriced")

        row = payload["estimates"][0]
        self.assertIsNone(row["estimated_cost_usd"])
        self.assertIsNone(row["input_cost_per_million_usd"])
        self.assertEqual(row["pricing_status"], "UNKNOWN")
        self.assertEqual(payload["priced_estimates"], 0)
        self.assertEqual(code, 1)
        self.assertIn("not $0.00", err)

    def test_an_unstamped_catalog_warns_loudly_even_when_it_can_price(self):
        """A number computed from prices of unknown age still gets a warning."""

        code, _, err = self._estimate("priced/known")

        self.assertEqual(code, 0)
        self.assertIn("WARNING", err)
        self.assertIn("never refreshed", err)

    def test_a_mixed_request_prices_what_it_can_and_flags_the_rest(self):
        code, payload, _ = self._estimate("priced/known", "meta/unpriced")

        self.assertEqual(payload["priced_estimates"], 1)
        self.assertEqual(payload["unpriced_estimates"], 1)
        self.assertEqual([row["pricing_status"] for row in payload["estimates"]], ["known", "UNKNOWN"])


if __name__ == "__main__":
    unittest.main()
