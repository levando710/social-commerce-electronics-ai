"""Offline invariants for neural ranking, fusion, and held-out interactions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.retrieval.search.dense_search import DenseRetriever, corpus_fingerprint, rank_vectors, reciprocal_rank_fusion, text_chunks
from phone_assistant.data.preparation.prepare_recsys import collect_interactions, k_core_size, popularity_metrics, temporal_split


class RetrievalTests(unittest.TestCase):
    def test_rrf_rewards_agreement_and_does_not_sum_raw_cosine_with_bm25(self):
        rows = reciprocal_rank_fusion({
            "sparse": [("a", 1000000), ("b", 1)],
            "dense": [("b", 0.7), ("c", 0.6)],
        })
        self.assertEqual(rows[0][0], "b")
        self.assertAlmostEqual(rows[0][1], 1 / 62 + 1 / 61)
        self.assertEqual(reciprocal_rank_fusion({"x": [("a", 1)]}, weights={"x": 0}), [])

    def test_multiple_chunks_do_not_duplicate_or_inflate_product_rank(self):
        import numpy as np
        vectors = np.array([[1, 0], [0.6, 0.8], [0, 1]])
        rows = rank_vectors(vectors, ["a", "a", "b"], [1, 0], 5)
        self.assertEqual(rows, [("a", 1.0), ("b", 0.0)])

    def test_chunking_keeps_tail_and_removes_handcrafted_visual_tags(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = text_chunks({"text": text + "\nVisual evidence: brightness_high"})
        self.assertIn("word299", chunks[-1])
        self.assertNotIn("brightness_high", " ".join(chunks))
        self.assertTrue(all(len(chunk.split()) <= 80 for chunk in chunks))

    def test_stale_corpus_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "manifest.json").write_text(json.dumps({
                "schema_version": 1, "corpus_sha256": corpus_fingerprint([{"doc_id": "old"}]),
                "channels": {},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                DenseRetriever(directory, [{"doc_id": "new"}])


def event(user, item, timestamp, rating=5):
    return {"user_id": user, "item_id": item, "timestamp": timestamp, "rating": rating}


class RecsysTests(unittest.TestCase):
    def test_latest_rating_applied_before_positive_filter(self):
        reviews = [
            {"user_id": "u", "parent_asin": "a", "timestamp": 1, "rating": 5},
            {"user_id": "u", "parent_asin": "a", "timestamp": 2, "rating": 1},
            {"user_id": "v", "parent_asin": "b", "timestamp": 3, "rating": 4},
        ]
        rows, audit = collect_interactions(reviews, {"a", "b"})
        self.assertEqual([row["item_id"] for row in rows], ["b"])
        self.assertNotEqual(rows[0]["user_id"], "v")
        self.assertEqual(audit["duplicate_user_item_rows"], 1)

    def test_temporal_split_keeps_singletons_and_strictly_separates_events(self):
        rows = [event("u", "a", 1), event("u", "b", 2), event("u", "c", 3),
                event("v", "a", 1), event("w", "a", 1), event("w", "b", 2)]
        train, validation, test, ties = temporal_split(rows)
        self.assertEqual((len(train), len(validation), len(test), ties), (3, 1, 2, 0))
        self.assertEqual(validation[0]["item_id"], "b")
        pair_sets = [{(row["user_id"], row["item_id"]) for row in split}
                     for split in (train, validation, test)]
        self.assertFalse(pair_sets[0] & pair_sets[1] or pair_sets[0] & pair_sets[2] or pair_sets[1] & pair_sets[2])

    def test_tied_last_timestamp_does_not_leak_into_holdout(self):
        train, validation, test, ties = temporal_split([event("u", "a", 1), event("u", "b", 1)])
        self.assertEqual((len(train), len(validation), len(test), ties), (2, 0, 0, 1))

    def test_pop_counts_train_only_and_keeps_cold_targets_in_denominator(self):
        train = [event("u", "a", 1), event("v", "b", 1)]
        result = popularity_metrics(train, [event("u", "c", 2)], {"a", "b", "c"}, k_values=(1, 2))
        self.assertEqual(result["all"]["hr@1"], 0)
        self.assertEqual(result["all"]["hr@2"], 1)
        self.assertEqual(result["cold_item"]["users"], 1)
        self.assertIsNone(result["warm_item"]["hr@1"])

    def test_two_core_is_iterative(self):
        rows = [event("u", "a", 1), event("u", "b", 2), event("v", "b", 3)]
        self.assertEqual(k_core_size(rows)["interactions"], 0)


if __name__ == "__main__":
    unittest.main()
