import unittest

from openrouter_model_router.openrouter import OpenRouterClient, OpenRouterError


class OpenRouterClientTests(unittest.TestCase):
    def test_headers_include_optional_attribution(self):
        client = OpenRouterClient(api_key="test", referer="https://example.test", app_title="Router")

        headers = client._headers()

        self.assertEqual(headers["Authorization"], "Bearer test")
        self.assertEqual(headers["HTTP-Referer"], "https://example.test")
        self.assertEqual(headers["X-OpenRouter-Title"], "Router")

    def test_chat_requires_key(self):
        client = OpenRouterClient(api_key=None)

        with self.assertRaises(OpenRouterError):
            client.chat_completion("openrouter/auto", [{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
