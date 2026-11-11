from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.metrics import run_generation_expansion as runner
from phone_assistant.llm import openai_client

SECRET = "sk-test-secret-value"


def reply(status=200, body=None, text=""):
    response = Mock(status_code=status, text=text)
    response.json.return_value = body
    return response


class OpenAIClientTests(unittest.TestCase):
    def call(self, response, **kwargs):
        with patch.object(openai_client, "get_secret", return_value=SECRET), \
                patch.object(openai_client.requests, "post", return_value=response) as post:
            result = openai_client.call_openai([{"text": "question"}, {"inline_data": {"mime_type": "image/png", "data": "QUJD"}}],
                                               temperature=0, max_output_tokens=50, **kwargs)
        return result, post

    def test_payload_uses_data_uri_image_json_mode_and_authorization_header(self):
        body = {"model": "gpt-4.1-mini-2025", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}}
        result, post = self.call(reply(body=body), response_mime_type="application/json", model="gpt-4.1-mini")
        sent = post.call_args.kwargs
        self.assertEqual(sent["headers"]["Authorization"], f"Bearer {SECRET}")
        self.assertEqual(sent["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(sent["json"]["max_completion_tokens"], 50)
        self.assertEqual(sent["json"]["temperature"], 0)
        blocks = sent["json"]["messages"][0]["content"]
        self.assertEqual(blocks[1]["image_url"]["url"], "data:image/png;base64,QUJD")
        self.assertEqual(result["usage"], {"promptTokenCount": 100, "candidatesTokenCount": 7,
                                           "thoughtsTokenCount": 0, "totalTokenCount": 107})
        self.assertEqual(result["model_version"], "gpt-4.1-mini-2025")
        self.assertEqual(post.call_count, 1)

    def test_reasoning_tokens_are_billed_as_output_but_counted_once(self):
        usage = openai_client.map_usage({"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40,
                                         "completion_tokens_details": {"reasoning_tokens": 20}})
        self.assertEqual((usage["candidatesTokenCount"], usage["thoughtsTokenCount"]), (10, 20))
        self.assertEqual(runner.usage_cost_usd(usage, {"input_usd_per_million": 1, "output_usd_per_million": 2}),
                         (10 + 30 * 2) / 1e6)

    def test_http_error_is_redacted_and_not_retried(self):
        result, post = self.call(reply(status=429, text=f"rate limit for {SECRET}"))
        self.assertIsNone(result["text"])
        self.assertTrue(result["error"].startswith("OpenAI API 429"))
        self.assertNotIn(SECRET, result["error"])
        self.assertEqual(post.call_count, 1)

    def test_missing_key_sends_nothing(self):
        with patch.object(openai_client, "get_secret", return_value=None), patch.object(openai_client.requests, "post") as post:
            result = openai_client.call_openai([{"text": "x"}])
        self.assertFalse(result["enabled"])
        post.assert_not_called()

    def test_client_error_is_not_charged_but_server_error_is_charged_worst_case(self):
        self.assertTrue("OpenAI API 400: bad".startswith(runner.HTTP_CLIENT_ERROR_PREFIXES))
        self.assertFalse("OpenAI API 500: down".startswith(runner.HTTP_CLIENT_ERROR_PREFIXES))


class ProviderBudgetTests(unittest.TestCase):
    BUDGET = {"model": "gemini-2.5-flash", "input_usd_per_million": 0.3, "output_usd_per_million": 2.5,
              "pricing_checked_date": "x", "scenario_usd": 1}

    def test_openai_budget_is_a_new_protocol_with_its_own_model_and_prices(self):
        result = runner.provider_budget(self.BUDGET, "openai", "gpt-4.1-mini")
        self.assertEqual((result["provider"], result["model"]), ("openai", "gpt-4.1-mini"))
        self.assertEqual((result["input_usd_per_million"], result["output_usd_per_million"]), (0.40, 1.60))

    def test_unknown_openai_model_requires_explicit_prices(self):
        with self.assertRaisesRegex(ValueError, "No known price"):
            runner.provider_budget(self.BUDGET, "openai", "some-new-model")
        result = runner.provider_budget(self.BUDGET, "openai", "some-new-model", 1.0, 2.0)
        self.assertEqual(result["output_usd_per_million"], 2.0)

    def test_gemini_budget_is_untouched(self):
        self.assertIs(runner.provider_budget(self.BUDGET, "gemini"), self.BUDGET)


if __name__ == "__main__":
    unittest.main()
