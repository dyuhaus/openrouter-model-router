import copy
import tempfile
import unittest
from pathlib import Path

from openrouter_model_router import (
    FakeTransport,
    ModelRouter,
    OpenRouterClient,
    OpenRouterError,
    RunLedger,
    TaskSpec,
)
from openrouter_model_router.ledger import STATUS_COMPLETED, STATUS_ERROR, STATUS_GATE_FAILED
from openrouter_model_router.reconcile import STATUS_DRIFT, reconcile
from openrouter_model_router.router import UNSELECTED_MODEL
from openrouter_model_router.transport import HttpResponse

from fixtures import NO_USAGE_RESPONSE, REAL_SHAPE_RESPONSE, test_catalog


def _client(*responses, api_key="sk-fake-test-key"):
    transport = FakeTransport([FakeTransport.json_response(r) if isinstance(r, dict) else r for r in responses])
    return OpenRouterClient(api_key=api_key, transport=transport), transport


class LiveKeyGuardTests(unittest.TestCase):
    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_no_key_fails_loudly_and_sends_nothing(self):
        """Asserting only 'it raised' cannot tell 'refused to send' from
        'sent, then failed'. The transport call count is the artifact."""

        transport = FakeTransport([FakeTransport.json_response(REAL_SHAPE_RESPONSE)])
        client = OpenRouterClient(api_key=None, transport=transport)

        with self.assertRaises(OpenRouterError) as ctx:
            client.chat("openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])

        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
        self.assertEqual(transport.call_count, 0, "no request may leave the process without a key")

    def test_blank_key_is_also_no_key(self):
        transport = FakeTransport([FakeTransport.json_response(REAL_SHAPE_RESPONSE)])
        client = OpenRouterClient(api_key="   ", transport=transport)

        with self.assertRaises(OpenRouterError):
            client.chat("openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])
        self.assertEqual(transport.call_count, 0)

    # --- POSITIVE CONTROL -------------------------------------------------
    def test_with_a_key_the_request_is_actually_sent(self):
        """Proves the guard above blocks on the key, not on being broken."""

        client, transport = _client(REAL_SHAPE_RESPONSE)

        client.chat("openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])

        self.assertEqual(transport.call_count, 1)
        sent = transport.last_request
        self.assertEqual(sent.method, "POST")
        self.assertTrue(sent.url.endswith("/chat/completions"))
        self.assertEqual(sent.headers["Authorization"], "Bearer sk-fake-test-key")
        self.assertEqual(sent.json_body()["model"], "openai/gpt-4.1-mini")


class UsageReadbackTests(unittest.TestCase):
    def test_chat_returns_usage_alongside_content(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        result = client.chat("openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "Espresso is 25-30 seconds.")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 194)
        self.assertEqual(result.completion_tokens, 88)
        self.assertAlmostEqual(result.reported_cost_usd, 0.00007830)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_missing_usage_does_not_become_zero_cost(self):
        client, _ = _client(NO_USAGE_RESPONSE)

        result = client.chat("some/quiet-model", [{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
        self.assertIsNone(result.reported_cost_usd)
        self.assertFalse(result.usage.present)

    def test_http_error_surfaces_the_body(self):
        client, _ = _client(HttpResponse(status=402, body=b'{"error":"insufficient credits"}'))

        with self.assertRaises(OpenRouterError) as ctx:
            client.chat("openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])

        self.assertIn("402", str(ctx.exception))
        self.assertIn("insufficient credits", str(ctx.exception))

    def test_non_json_body_is_an_error_not_a_silent_empty_result(self):
        client, _ = _client(HttpResponse(status=200, body=b"<html>gateway timeout</html>"))

        with self.assertRaises(OpenRouterError):
            client.chat("openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])

    def test_legacy_chat_completion_still_returns_the_raw_payload(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        payload = client.chat_completion("openai/gpt-4.1-mini", [{"role": "user", "content": "hi"}])

        self.assertEqual(payload["choices"][0]["message"]["content"], "Espresso is 25-30 seconds.")


class InstrumentedRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = RunLedger(Path(self._tmp.name) / "runs.jsonl")
        self.router = ModelRouter(test_catalog())
        self.task = TaskSpec(
            task_type="general",
            input_tokens=194,
            output_tokens=88,
            allow_models=("openai/gpt-4.1-mini",),
        )

    def test_successful_run_is_recorded_with_both_costs(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        outcome = self.router.run(
            client=client,
            messages=[{"role": "user", "content": "hi"}],
            task=self.task,
            ledger=self.ledger,
            task_label="lesson-3",
        )

        self.assertTrue(outcome.succeeded)
        record = self.ledger.read_all()[-1]
        self.assertEqual(record.status, STATUS_COMPLETED)
        self.assertEqual(record.task_label, "lesson-3")
        self.assertEqual(record.prompt_tokens, 194)
        self.assertEqual(record.completion_tokens, 88)
        self.assertAlmostEqual(record.reported_cost_usd, 0.00007830)
        # 194 in @ $0.15/M + 88 out @ $0.60/M
        self.assertAlmostEqual(record.estimated_cost_usd, 194 * 0.15 / 1e6 + 88 * 0.60 / 1e6)
        self.assertTrue(record.usage_present)
        self.assertTrue(record.gates_passed)

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_gate_failure_is_recorded_as_a_failed_run(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        outcome = self.router.run(
            client=client,
            messages=[{"role": "user", "content": "hi"}],
            task=self.task,
            ledger=self.ledger,
            task_label="lesson-4",
            gate=lambda result: ["fabricated source: https://example.invalid/does-not-exist"],
        )

        self.assertFalse(outcome.succeeded)
        record = self.ledger.read_all()[-1]
        self.assertEqual(record.status, STATUS_GATE_FAILED)
        self.assertIs(record.gates_passed, False)
        self.assertIn("fabricated source", record.gate_failures[0])
        # The failed attempt still cost money, and the ledger still knows it.
        self.assertAlmostEqual(record.reported_cost_usd, 0.00007830)

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_api_error_is_recorded_as_a_failed_run(self):
        client, _ = _client(HttpResponse(status=502, body=b"bad gateway"))

        outcome = self.router.run(
            client=client,
            messages=[{"role": "user", "content": "hi"}],
            task=self.task,
            ledger=self.ledger,
            task_label="lesson-5",
        )

        self.assertFalse(outcome.succeeded)
        self.assertIsInstance(outcome.error, OpenRouterError)
        record = self.ledger.read_all()[-1]
        self.assertEqual(record.status, STATUS_ERROR)
        self.assertIn("502", record.error)
        self.assertIsNone(record.reported_cost_usd)

    def test_selection_failure_is_still_a_ledger_row(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        outcome = self.router.run(
            client=client,
            messages=[{"role": "user", "content": "hi"}],
            task=TaskSpec(required_capabilities=("audio",)),
            ledger=self.ledger,
            task_label="lesson-6",
        )

        self.assertFalse(outcome.succeeded)
        record = self.ledger.read_all()[-1]
        self.assertEqual(record.model, UNSELECTED_MODEL)
        self.assertEqual(record.status, STATUS_ERROR)

    def test_a_gate_that_raises_counts_as_failed(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        def exploding_gate(result):
            raise KeyError("sources")

        outcome = self.router.run(
            client=client,
            messages=[{"role": "user", "content": "hi"}],
            task=self.task,
            ledger=self.ledger,
            gate=exploding_gate,
        )

        self.assertFalse(outcome.succeeded)
        self.assertIn("gate raised KeyError", self.ledger.read_all()[-1].gate_failures[0])

    def test_unpriced_model_records_no_estimate_rather_than_zero(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        self.router.run(
            client=client,
            messages=[{"role": "user", "content": "hi"}],
            task=TaskSpec(allow_models=("meta/unpriced-router",), input_tokens=194, output_tokens=88),
            ledger=self.ledger,
        )

        record = self.ledger.read_all()[-1]
        self.assertIsNone(record.estimated_cost_usd, "an unknown price is not $0.00")
        self.assertIsNotNone(record.reported_cost_usd)

    def test_multiplier_and_reconciliation_over_a_real_run_sequence(self):
        expensive = copy.deepcopy(REAL_SHAPE_RESPONSE)
        expensive["usage"]["cost"] = 0.0400  # provider charged ~500x the estimate

        client, _ = _client(REAL_SHAPE_RESPONSE, expensive, REAL_SHAPE_RESPONSE)
        messages = [{"role": "user", "content": "hi"}]

        self.router.run(client=client, messages=messages, task=self.task, ledger=self.ledger,
                        gate=lambda r: ["two identical options"])
        self.router.run(client=client, messages=messages, task=self.task, ledger=self.ledger, attempt=2)
        self.router.run(client=client, messages=messages, task=self.task, ledger=self.ledger, attempt=3)

        summary = self.ledger.summary()
        self.assertEqual(summary["runs"], 3)
        self.assertEqual(summary["failed"], 1)
        self.assertAlmostEqual(summary["retry_multiplier"], 1.5)

        report = reconcile(self.ledger.read_all())
        self.assertEqual(report.status, STATUS_DRIFT)
        self.assertIn("openai/gpt-4.1-mini", [m.model for m in report.flagged_models])

    def test_ledger_is_optional(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        outcome = self.router.run(client=client, messages=[{"role": "user", "content": "hi"}], task=self.task)

        self.assertTrue(outcome.succeeded)

    def test_legacy_router_chat_completion_shape_is_preserved_and_extended(self):
        client, _ = _client(REAL_SHAPE_RESPONSE)

        out = self.router.chat_completion(
            client=client, messages=[{"role": "user", "content": "hi"}], task=self.task, ledger=self.ledger
        )

        self.assertEqual(out["selection"].model_id, "openai/gpt-4.1-mini")
        self.assertIn("choices", out["response"])
        self.assertEqual(out["content"], "Espresso is 25-30 seconds.")
        self.assertAlmostEqual(out["reported_cost_usd"], 0.00007830)
        self.assertEqual(len(self.ledger.read_all()), 1)

    def test_legacy_router_chat_completion_still_raises_on_error(self):
        client, _ = _client(HttpResponse(status=500, body=b"boom"))

        with self.assertRaises(OpenRouterError):
            self.router.chat_completion(
                client=client, messages=[{"role": "user", "content": "hi"}], task=self.task, ledger=self.ledger
            )

        self.assertEqual(self.ledger.read_all()[-1].status, STATUS_ERROR)


if __name__ == "__main__":
    unittest.main()
