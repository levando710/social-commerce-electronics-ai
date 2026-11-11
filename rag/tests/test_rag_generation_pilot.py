import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.metrics.evaluate_rag_generation_pilot import BudgetedClient, MAXIMUM_REQUESTS, parse_judgment


class GenerationPilotTests(unittest.TestCase):
    def test_supported_claim_requires_exact_text_or_supplied_image_reference(self):
        claims = [
            {"text": "128GB", "supported": True, "evidence_type": "text", "doc_id": "a", "quote": "128GB"},
            {"text": "blue", "supported": True, "evidence_type": "image", "doc_id": "a", "image_id": "image_1"},
            {"text": "blue", "supported": True, "evidence_type": "image", "doc_id": "b", "image_id": "image_1"},
            {"text": "256GB", "supported": True, "evidence_type": "text", "doc_id": "a", "quote": "256GB"},
        ]
        result = parse_judgment(json.dumps({"relevance": 2, "claims": claims}), {"a": "Storage 128GB"}, [{"image_id": "image_1", "doc_id": "a"}])
        self.assertEqual([c["supported"] for c in result["claims"]], [True, True, None, None])

    def test_boolean_relevance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema/relevance"):
            parse_judgment('{"claims":[],"relevance":true}', {}, [])

    def test_completed_request_is_reused_and_error_stops_later_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('phone_assistant.evaluation.metrics.evaluate_rag_generation_pilot.call_gemini', return_value={"text": "saved answer", "error": None}) as api:
                first = BudgetedClient(root).call("first", [], 16)
                self.assertEqual(BudgetedClient(root).call("first", [], 16), first)
                self.assertEqual(api.call_count, 1)
            with patch('phone_assistant.evaluation.metrics.evaluate_rag_generation_pilot.time.sleep'), patch('phone_assistant.evaluation.metrics.evaluate_rag_generation_pilot.call_gemini', return_value={"text": None, "error": "Gemini API 429"}) as api:
                client = BudgetedClient(root)
                client.call("second", [], 16)
                self.assertEqual(client.call("third", [], 16)["status"], "not_attempted")
                self.assertEqual(BudgetedClient(root).call("fourth", [], 16)["status"], "not_attempted")
                self.assertEqual(api.call_count, 1)

    def test_interrupted_request_and_exhausted_budget_do_not_send_network_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{"request_id": str(i), "started_unix": 0} for i in range(MAXIMUM_REQUESTS)]
            (root / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with patch('phone_assistant.evaluation.metrics.evaluate_rag_generation_pilot.call_gemini') as api:
                client = BudgetedClient(root)
                self.assertEqual(client.call("0", [], 16)["status"], "interrupted_request")
                self.assertEqual(client.call("new", [], 16)["status"], "not_attempted")
                api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
