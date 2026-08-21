import json
import tempfile
import unittest
from pathlib import Path

from openrouter_model_router import FakeTransport, ModelCatalog, ModelRouter, TaskSpec
from openrouter_model_router.catalog import CatalogRefreshError, model_from_openrouter
from openrouter_model_router.transport import HttpResponse

PAYLOAD = {
    "data": [
        {
            "id": "openai/gpt-4.1-mini",
            "name": "GPT-4.1 Mini",
            "context_length": 1_047_576,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools", "response_format"],
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "openrouter/auto",
            "name": "Auto Router",
            "context_length": 2_000_000,
            "pricing": {"prompt": "-1", "completion": "-1"},
        },
        {
            "id": "vendor/no-pricing-block",
            "name": "Mystery",
            "context_length": 200_000,
        },
    ]
}


class RefreshTests(unittest.TestCase):
    def test_refresh_populates_real_prices_over_a_fake_transport(self):
        transport = FakeTransport([FakeTransport.json_response(PAYLOAD)])

        catalog = ModelCatalog.refresh_from_openrouter(transport=transport)

        self.assertEqual(len(catalog), 3)
        model = catalog.get("openai/gpt-4.1-mini")
        self.assertAlmostEqual(model.input_cost_per_million, 0.15)
        self.assertAlmostEqual(model.output_cost_per_million, 0.60)
        self.assertTrue(model.pricing_known)

    def test_refresh_sends_no_authorization_header_when_there_is_no_key(self):
        """The /models endpoint is public; requiring a key would be the bug."""

        transport = FakeTransport([FakeTransport.json_response(PAYLOAD)])

        ModelCatalog.refresh_from_openrouter(transport=transport)

        self.assertNotIn("Authorization", transport.last_request.headers)
        self.assertTrue(transport.last_request.url.endswith("/models"))

    def test_refresh_forwards_a_key_when_one_is_given(self):
        transport = FakeTransport([FakeTransport.json_response(PAYLOAD)])

        ModelCatalog.refresh_from_openrouter(transport=transport, api_key="sk-fake")

        self.assertEqual(transport.last_request.headers["Authorization"], "Bearer sk-fake")

    def test_http_error_raises_rather_than_writing_an_empty_catalog(self):
        transport = FakeTransport([HttpResponse(status=503, body=b"upstream unavailable")])

        with self.assertRaises(CatalogRefreshError) as ctx:
            ModelCatalog.refresh_from_openrouter(transport=transport)

        self.assertIn("503", str(ctx.exception))
        self.assertIn("upstream unavailable", str(ctx.exception))


class PricingKnownTests(unittest.TestCase):
    def test_negative_sentinel_is_unknown_pricing_not_free(self):
        """OpenRouter publishes -1 on its meta-routers. Clamping that to 0.0 and
        calling it free is what made a 120K-token request estimate $0.00."""

        model = model_from_openrouter(
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}, "context_length": 2_000_000}
        )

        self.assertEqual(model.input_cost_per_million, 0.0)
        self.assertFalse(model.pricing_known)
        self.assertFalse(model.is_free)
        self.assertNotIn("cheap", model.capabilities)

    def test_absent_pricing_block_is_unknown_pricing(self):
        model = model_from_openrouter({"id": "vendor/no-pricing-block", "context_length": 200_000})

        self.assertFalse(model.pricing_known)
        self.assertFalse(model.is_free)

    def test_a_genuine_zero_price_is_free_and_known(self):
        model = model_from_openrouter(
            {"id": "vendor/model:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 32_000}
        )

        self.assertTrue(model.pricing_known)
        self.assertTrue(model.is_free)
        self.assertIn("cheap", model.capabilities)

    def test_pricing_coverage_counts_what_can_actually_be_costed(self):
        catalog = ModelCatalog.from_openrouter_payload(PAYLOAD)

        coverage = catalog.pricing_coverage()

        self.assertEqual(coverage["total"], 3)
        self.assertEqual(coverage["priced"], 1)
        self.assertEqual(coverage["pricing_unknown"], 2)
        self.assertIn("openrouter/auto", coverage["pricing_unknown_ids"])

    def test_bootstrap_catalog_declares_its_prices_unknown(self):
        """The 1-model fallback must not masquerade as a priced catalog.

        This is the exact failure mode the repo shipped: with no catalog on
        disk, `ModelRouter.from_file()` silently fell back to this entry and
        quoted $0.00 for a 120K-token request as though that were a real price.
        """

        catalog = ModelCatalog.bootstrap()
        coverage = catalog.pricing_coverage()

        self.assertEqual(coverage["priced"], 0)
        self.assertEqual(coverage["pricing_unknown"], 1)
        self.assertFalse(catalog.get("openrouter/auto").pricing_known)
        self.assertFalse(catalog.get("openrouter/auto").is_free)

    def test_bootstrap_selection_labels_its_zero_estimate_as_unknown(self):
        selection = ModelRouter().select(TaskSpec(input_tokens=20_000, output_tokens=100_000))

        self.assertEqual(selection.estimated_cost_usd, 0.0)
        self.assertFalse(
            selection.estimated_cost_is_known,
            "a $0.00 quote from the bootstrap fallback must not look like a measurement",
        )

    def test_pricing_known_survives_save_and_load(self):
        catalog = ModelCatalog.from_openrouter_payload(PAYLOAD)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            catalog.save(path)
            self.assertIn("pricing_known", json.loads(path.read_text())["models"][0])
            loaded = ModelCatalog.load(path)

        self.assertFalse(loaded.get("openrouter/auto").pricing_known)
        self.assertTrue(loaded.get("openai/gpt-4.1-mini").pricing_known)

    def test_selection_reports_whether_its_estimate_is_knowable(self):
        router = ModelRouter(ModelCatalog.from_openrouter_payload(PAYLOAD))

        priced = router.select(TaskSpec(allow_models=("openai/gpt-4.1-mini",), input_tokens=40_000, output_tokens=120_000))
        unpriced = router.select(TaskSpec(allow_models=("openrouter/auto",), input_tokens=40_000, output_tokens=120_000))

        self.assertTrue(priced.estimated_cost_is_known)
        self.assertGreater(priced.estimated_cost_usd, 0)
        self.assertFalse(unpriced.estimated_cost_is_known)
        self.assertEqual(unpriced.estimated_cost_usd, 0.0)
        self.assertFalse(unpriced.to_dict()["estimated_cost_is_known"])


if __name__ == "__main__":
    unittest.main()
