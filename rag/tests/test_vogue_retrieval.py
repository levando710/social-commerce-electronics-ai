from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.metrics.evaluate_vogue_retrieval import observations, metric_rows, inputs
from phone_assistant.recommendation.recsys_data import sha256


class VogueRetrievalTests(unittest.TestCase):
    def test_query_is_first_seeker_turn_without_future_assistant_or_goal(self):
        transcript = {"gt_items": [22], "conversation_content": [
            {"content": {"role": "Assistant", "utterances": ["hello"]}},
            {"content": {"role": "Seeker", "utterances": ["a jacket", "for cold weather"]}},
            {"content": {"role": "Assistant", "utterances": ["buy hidden item 22"]}},
            {"content": {"role": "Seeker", "utterances": ["yes I choose 22"]}}]}
        self.assertEqual(observations(transcript), "a jacket for cold weather")

    def test_multi_selection_metrics_keep_unretrieved_gold_in_recall_denominator(self):
        result = metric_rows([["1", "2", "3", "4", "5"]], [["1", "9"]], 10)
        self.assertEqual(result["precision@5"], .2)
        self.assertEqual(result["recall@5"], .5)
        self.assertEqual(result["mrr@5"], 1.)
        self.assertLess(result["ndcg@5"], 1.)

    def test_published_catalog_and_prefixes_have_expected_distinct_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'data/external/vogue'
            metadata = source / "data/metadata/item_1.json"
            image = metadata.with_suffix(".png")
            transcript = source / "data/conversation_trials/transcripts/c1.json"
            metadata.parent.mkdir(parents=True); transcript.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({"product_name": "jacket", "categories": ["clothing"],
                "reviews": "LEAK", "product_rating": "LEAK"}))
            image.write_bytes(b"fixture image hash; encoder not called by input parsing")
            transcript.write_text(json.dumps({"conversation_id": 1, "session_code": "s1", "gt_items": [1],
                "conversation_content": [{"content": {"role": "Seeker", "utterances": ["a jacket"]}},
                    {"content": {"role": "Assistant", "utterances": ["LEAK future suggestion"]}}]}))
            manifest = {"vogue": {"revision": "fixture", "files": [
                {"path": p.relative_to(source).as_posix(), "sha256": sha256(p)} for p in (metadata, image, transcript)]}}
            (root / 'data/external/download_manifest.json').write_text(json.dumps(manifest))
            with patch('phone_assistant.evaluation.metrics.evaluate_vogue_retrieval.ROOT', root), patch('phone_assistant.evaluation.metrics.evaluate_vogue_retrieval.SOURCE', source), patch('phone_assistant.evaluation.metrics.evaluate_vogue_retrieval.CATALOG_LAST_ITEM', 1):
                documents, images, queries, labels, contract = inputs()
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(images), 1)
            self.assertEqual(queries[0]["query"], "a jacket")
            self.assertEqual(labels, [["1"]])
            self.assertNotIn("LEAK", documents[0]["text"])
            self.assertIn("source_revision", contract)


if __name__ == "__main__":
    unittest.main()
