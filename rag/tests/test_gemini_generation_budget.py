from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.llm.gemini_client import generate_answer, GENERATED_ANSWER_TOKEN_LIMIT


class GeminiGenerationBudgetTests(unittest.TestCase):
    def contexts(self):
        return [{"doc_id": "B000000001", "title": "Catalog phone", "evidence": "Storage 128GB", "catalog_facts": {"storage_gb": 128}}]

    def test_low_budget_generation_sets_zero_thinking_and_keeps_normalized_citations(self):
        response = Mock(status_code=200)
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Storage 128GB [1]"}]}}]}
        with patch('phone_assistant.llm.gemini_client.get_secret', return_value="unit-test-secret"), patch('phone_assistant.llm.gemini_client.gemini_model_name', return_value="gemini-2.5-flash"), patch('phone_assistant.llm.gemini_client.requests.post', return_value=response) as post:
            result = generate_answer("Storage?", self.contexts(), include_images=False)
        config = post.call_args.kwargs["json"]["generationConfig"]
        self.assertEqual(config["thinkingConfig"]["thinkingBudget"], 0)
        self.assertEqual(config["maxOutputTokens"], GENERATED_ANSWER_TOKEN_LIMIT)
        self.assertIn("[B000000001]", result["answer"])
        self.assertEqual(post.call_count, 1)

    def test_default_generator_does_not_repeat_a_quota_failure(self):
        response = Mock(status_code=429, text="quota exhausted", headers={})
        with patch('phone_assistant.llm.gemini_client.get_secret', return_value="unit-test-secret"), patch('phone_assistant.llm.gemini_client.gemini_model_name', return_value="gemini-2.5-flash"), patch('phone_assistant.llm.gemini_client.requests.post', return_value=response) as post, patch('phone_assistant.llm.gemini_client.time.sleep') as sleep:
            result = generate_answer("Storage?", self.contexts(), include_images=False)
        self.assertIsNone(result["answer"])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_each_supplied_image_is_associated_with_its_exact_catalog_id(self):
        contexts = self.contexts(); contexts[0]["product_images"] = ["fixture.png"]
        with patch('phone_assistant.llm.gemini_client.image_part', return_value={"inline_data": {"mime_type": "image/png", "data": "AA=="}}), patch('phone_assistant.llm.gemini_client.call_gemini', return_value={"enabled": True, "text": "Storage [1]", "error": None}) as api:
            generate_answer("Describe the image", contexts)
        parts = api.call_args.args[0]
        self.assertIn("[B000000001]", parts[1]["text"])
        self.assertIn("inline_data", parts[2])


if __name__ == "__main__":
    unittest.main()
