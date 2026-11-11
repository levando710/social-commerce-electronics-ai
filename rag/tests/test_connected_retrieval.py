from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import metrics, ordered, query_chunks


class ConnectedIRTests(unittest.TestCase):
    def test_incomplete_qrel_metrics_and_user_macro_averaging(self):
        rows = [{"user_id": "a", "rank": 1, "top_items": ["x", "y"]},
                {"user_id": "a", "rank": 1, "top_items": ["x", "y"]},
                {"user_id": "b", "rank": 100, "top_items": ["z", "y"]}]
        result = metrics(rows, 4, k_values=(2,))
        self.assertEqual(result["recall@2"], .5)
        self.assertEqual(result["precision@2"], .25)
        self.assertEqual(result["ndcg@2"], .5)
        self.assertEqual(result["mrr@2"], .5)
        self.assertEqual(result["coverage@2"], .75)
        self.assertEqual(result["users"], 2)

    def test_masked_images_and_stable_ties(self):
        ranking = ordered(np.asarray([1., 1., 99.]), ["b", "a", "missing"], np.asarray([True, True, False]))
        self.assertEqual([r[0] for r in ranking], ["a", "b"])

    def test_clip_queries_chunk_tail_instead_of_dropping_long_context(self):
        text = " ".join(f"word{i}" for i in range(70))
        chunks = query_chunks(text, "clip")
        self.assertGreater(len(chunks), 1)
        self.assertIn("word69", chunks[-1])
        self.assertTrue(all(len(chunk.split()) <= 32 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
