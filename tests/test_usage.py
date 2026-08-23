import unittest

from openrouter_model_router.usage import TokenUsage, parse_usage

from fixtures import ALIAS_SHAPE_RESPONSE, NO_USAGE_RESPONSE, REAL_SHAPE_RESPONSE


class UsageParsingTests(unittest.TestCase):
    def test_reads_documented_openrouter_shape(self):
        usage = parse_usage(REAL_SHAPE_RESPONSE)

        self.assertTrue(usage.present)
        self.assertEqual(usage.prompt_tokens, 194)
        self.assertEqual(usage.completion_tokens, 88)
        self.assertEqual(usage.total_tokens, 282)
        self.assertAlmostEqual(usage.reported_cost_usd, 0.00007830)
        self.assertIs(usage.is_byok, False)
        self.assertEqual(usage.cached_tokens, 0)
        self.assertEqual(usage.reasoning_tokens, 0)
        self.assertEqual(usage.upstream_inference_cost_usd, 0.0)
        self.assertEqual(usage.missing_fields, ())
        self.assertEqual(usage.source_fields["prompt_tokens"], "prompt_tokens")
        self.assertEqual(usage.source_fields["cost"], "cost")

    def test_reads_alternate_field_names_and_records_which_it_used(self):
        usage = parse_usage(ALIAS_SHAPE_RESPONSE)

        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 50)
        self.assertAlmostEqual(usage.reported_cost_usd, 0.0025)
        self.assertEqual(usage.source_fields["prompt_tokens"], "input_tokens")
        self.assertEqual(usage.source_fields["completion_tokens"], "output_tokens")
        self.assertEqual(usage.source_fields["cost"], "total_cost")
        # total_tokens really was absent, and the parser says so.
        self.assertIn("total_tokens", usage.missing_fields)
        self.assertEqual(usage.resolved_total_tokens(), 150)

    def test_absent_usage_is_unknown_not_zero(self):
        """The whole point: a missing cost must never read as $0.00."""

        usage = parse_usage(NO_USAGE_RESPONSE)

        self.assertFalse(usage.present)
        self.assertIsNone(usage.prompt_tokens)
        self.assertIsNone(usage.completion_tokens)
        self.assertIsNone(usage.reported_cost_usd)
        self.assertFalse(usage.has_reported_cost)
        self.assertNotEqual(usage.reported_cost_usd, 0.0)
        self.assertIn("cost", usage.missing_fields)

    def test_zero_cost_is_distinguishable_from_missing_cost(self):
        reported_zero = parse_usage({"usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}})
        not_reported = parse_usage({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

        self.assertEqual(reported_zero.reported_cost_usd, 0.0)
        self.assertTrue(reported_zero.has_reported_cost)
        self.assertIsNone(not_reported.reported_cost_usd)
        self.assertFalse(not_reported.has_reported_cost)

    def test_malformed_payloads_do_not_raise(self):
        for payload in (None, [], "nope", {"usage": "nope"}, {"usage": {"prompt_tokens": "many"}}):
            usage = parse_usage(payload)
            self.assertIsInstance(usage, TokenUsage)
            self.assertIsNone(usage.reported_cost_usd)

    def test_unnested_generation_stats_shape(self):
        usage = parse_usage({"tokens_prompt": 12, "tokens_completion": 7, "total_cost": 0.001})

        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertAlmostEqual(usage.reported_cost_usd, 0.001)


if __name__ == "__main__":
    unittest.main()
