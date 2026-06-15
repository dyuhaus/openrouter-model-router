import tempfile
import unittest
from pathlib import Path

from openrouter_model_router.catalog import ModelCatalog, model_from_openrouter


class CatalogTests(unittest.TestCase):
    def test_openrouter_payload_infers_capabilities_and_prices(self):
        raw = {
            "id": "example/coder-flash",
            "name": "Coder Flash",
            "description": "Fast coding model with vision support",
            "context_length": 131072,
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "supported_parameters": ["tools", "response_format"],
            "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
        }

        model = model_from_openrouter(raw)

        self.assertAlmostEqual(model.input_cost_per_million, 0.2)
        self.assertAlmostEqual(model.output_cost_per_million, 0.8)
        self.assertIn("coding", model.capabilities)
        self.assertIn("fast", model.capabilities)
        self.assertIn("long_context", model.capabilities)
        self.assertIn("tool_use", model.capabilities)
        self.assertIn("json_mode", model.capabilities)
        self.assertIn("vision", model.capabilities)

    def test_catalog_save_and_load_round_trip(self):
        catalog = ModelCatalog.bootstrap()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            catalog.save(path)
            loaded = ModelCatalog.load(path)

        self.assertEqual(len(loaded), len(catalog))
        self.assertIsNotNone(loaded.get("openrouter/auto"))

    def test_negative_provider_price_sentinels_are_clamped(self):
        raw = {
            "id": "openrouter/auto",
            "pricing": {"prompt": "-1", "completion": "-1"},
            "context_length": 128000,
        }

        model = model_from_openrouter(raw)

        self.assertEqual(model.input_cost_per_million, 0.0)
        self.assertEqual(model.output_cost_per_million, 0.0)
        self.assertEqual(model.estimated_cost_usd(1000, 1000), 0.0)

    def test_from_openrouter_payload_requires_data_list(self):
        with self.assertRaises(Exception):
            ModelCatalog.from_openrouter_payload({"bad": []})

    def test_merge_preserves_observed_stats(self):
        catalog = ModelCatalog.bootstrap()
        catalog.record_outcome("openrouter/auto", success=True, latency_ms=1000, quality_score=0.9)
        incoming = ModelCatalog.bootstrap()

        added, updated = catalog.merge(incoming)

        self.assertEqual(added, 0)
        self.assertEqual(updated, 1)
        self.assertGreater(catalog.get("openrouter/auto").stats["runs"], 0)


if __name__ == "__main__":
    unittest.main()
