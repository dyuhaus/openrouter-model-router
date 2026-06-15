import unittest

from openrouter_model_router import ModelCatalog, ModelInfo, ModelRouter, TaskSpec


def _catalog():
    return ModelCatalog(
        [
            ModelInfo(
                id="cheap/text-fast",
                context_length=32000,
                input_cost_per_million=0.05,
                output_cost_per_million=0.10,
                capabilities=("text", "cheap", "fast", "json_mode"),
                quality_score=0.45,
                speed_score=0.9,
                reliability_score=0.7,
            ),
            ModelInfo(
                id="premium/reasoning-code",
                context_length=200000,
                input_cost_per_million=4.0,
                output_cost_per_million=12.0,
                capabilities=("text", "reasoning", "coding", "tool_use", "json_mode", "long_context"),
                quality_score=0.92,
                speed_score=0.45,
                reliability_score=0.8,
            ),
            ModelInfo(
                id="vision/general",
                context_length=128000,
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
                capabilities=("text", "vision", "json_mode"),
                quality_score=0.72,
                speed_score=0.6,
                reliability_score=0.75,
            ),
        ]
    )


class RouterTests(unittest.TestCase):
    def test_selects_premium_for_quality_coding_task(self):
        router = ModelRouter(_catalog())
        selection = router.select(
            TaskSpec(
                task_type="coding",
                required_capabilities=("coding", "tool_use"),
                preference="quality",
            )
        )

        self.assertEqual(selection.model.id, "premium/reasoning-code")
        self.assertIn("coding_match", selection.reasons)

    def test_selects_cheap_model_for_cheap_preference(self):
        router = ModelRouter(_catalog())
        selection = router.select(TaskSpec(task_type="json extraction", preference="cheap"))

        self.assertEqual(selection.model.id, "cheap/text-fast")

    def test_filters_by_vision_modality(self):
        router = ModelRouter(_catalog())
        selection = router.select(TaskSpec(task_type="image extraction", modalities=("text", "image")))

        self.assertEqual(selection.model.id, "vision/general")

    def test_respects_max_cost(self):
        router = ModelRouter(_catalog())
        selection = router.select(
            TaskSpec(
                task_type="general",
                input_tokens=10_000,
                output_tokens=1_000,
                max_cost_usd=0.01,
                preference="quality",
            )
        )

        self.assertEqual(selection.model.id, "cheap/text-fast")

    def test_fallback_model_when_filters_remove_all_candidates(self):
        router = ModelRouter(_catalog())
        selection = router.select(
            TaskSpec(
                required_capabilities=("audio",),
                fallback_model="cheap/text-fast",
            )
        )

        self.assertEqual(selection.model.id, "cheap/text-fast")


if __name__ == "__main__":
    unittest.main()
