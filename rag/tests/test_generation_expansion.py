import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.reviews.prepare_generation_expansion import build_cases, canonical_rows, cost_scenario, validate_pairs


class GenerationExpansionTests(unittest.TestCase):
    def test_selection_ignores_english_and_duplicate_ids_are_rejected(self):
        original = {"query_id": "intent_vi_0", "intent_id": "intent", "kind": "retrieval", "contexts": [{}]}
        english = {**original, "query_id": "intent_en_0"}
        self.assertEqual(canonical_rows([english, original]), [original])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            canonical_rows([original, original])

    def test_missing_canonical_and_nonempty_abstention_are_rejected(self):
        row = {"query_id": "outside_en", "intent_id": "outside", "kind": "expected_abstention", "contexts": []}
        with self.assertRaisesRegex(ValueError, "canonical"):
            canonical_rows([row])
        row.update(query_id="outside_vi", contexts=[{}])
        with self.assertRaisesRegex(ValueError, "abstention"):
            canonical_rows([row])

    def test_pairs_share_prompts_and_keep_images_out_of_text_arm_without_api(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "catalog.img"
            Image.new("RGB", (4, 4)).save(image, format="PNG")
            row = {"query_id": "phone_vi_0", "intent_id": "phone", "kind": "retrieval", "query": "phone",
                   "contexts": [{"doc_id": "product", "title": "phone", "product_images": [str(image)]}],
                   "gold_target": "SHOULD_NOT_ENTER_PROMPT"}
            with patch('phone_assistant.llm.gemini_client.requests.post') as post:
                cases = build_cases({"development": [row]})
                post.assert_not_called()
            self.assertEqual(len(cases), 6)
            self.assertTrue(all(not r["images"] for r in cases if r["arm"] == "text_only"))
            self.assertTrue(all(len(r["images"]) == 1 for r in cases if r["arm"] == "text_image"))
            self.assertTrue(all("SHOULD_NOT_ENTER_PROMPT" not in r["prompt"] for r in cases))
            tampered = copy.deepcopy(cases)
            tampered[0]["prompt"] += "changed"
            with self.assertRaisesRegex(ValueError, "Paired evidence"):
                validate_pairs(tampered)

    def test_missing_images_do_not_silently_become_image_comparisons(self):
        row = {"query_id": "phone_vi_0", "intent_id": "phone", "kind": "retrieval", "query": "phone",
               "contexts": [{"doc_id": "product", "product_images": []}]}
        with self.assertRaisesRegex(ValueError, "valid supplied image"):
            build_cases({"development": [row]})

    def test_empty_context_controls_are_retained_and_budget_counts_judging(self):
        row = {"query_id": "outside_vi", "intent_id": "outside", "kind": "expected_abstention",
               "query": "tablet", "contexts": []}
        cases = build_cases({"development": [row]})
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(not row["images"] and not row["contexts"] for row in cases))
        budget = cost_scenario(104)
        self.assertEqual(budget["maximum_requests_no_retries"], 208)
        self.assertAlmostEqual(budget["scenario_usd"], 1.43936)


if __name__ == "__main__":
    unittest.main()
