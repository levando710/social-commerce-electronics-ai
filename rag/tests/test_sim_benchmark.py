"""Leakage and evaluation checks for the aligned public-data benchmark."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.preparation.prepare_sim_benchmark import assign_cohorts, clean_user, initial_query, safe_product
from phone_assistant.evaluation.metrics.evaluate_sim_benchmark import aggregate, rank_target
from phone_assistant.recommendation.recsys_data import metadata_text


def event(item, timestamp, rating=5):
    return {"item": item, "ts": timestamp, "rating": rating}


class SimBenchmarkTests(unittest.TestCase):
    def test_future_events_and_target_do_not_enter_history(self):
        row = {"user_id": "u", "target": event("t", 20),
               "interaction_history": [event("a", 5), event("b", 8), event("c", 30), event("t", 20)]}
        clean, audit = clean_user(row, {"a", "b", "c", "t"})
        self.assertEqual({e["item_id"] for e in clean["history"]}, {"a", "b"})
        self.assertEqual(audit["removed_nonpast_or_target_history_events"], 2)

    def test_latest_negative_feedback_is_not_a_positive(self):
        row = {"user_id": "u", "target": event("t", 20),
               "interaction_history": [event("a", 1), event("a", 2, 1), event("b", 3)]}
        clean, audit = clean_user(row, {"a", "b", "t"}, min_history=1)
        self.assertEqual([e["item_id"] for e in clean["history"]], ["b"])
        self.assertEqual(audit["nonpositive_unique_past_items"], 1)

    def test_previously_seen_target_rejects_novel_item_case_even_if_disliked(self):
        row = {"user_id": "u", "target": event("t", 20),
               "interaction_history": [event("a", 1), event("b", 2), event("t", 3, 1)]}
        clean, audit = clean_user(row, {"a", "b", "t"})
        self.assertIsNone(clean)
        self.assertEqual(audit["excluded_previously_seen_target"], 1)

    def test_first_message_excludes_assistant_and_future_user_content(self):
        text = "[USR]: need a case\n[CRS]: SECRET ANSWER\n[USR]: FINAL TARGET"
        self.assertEqual(initial_query(text), "need a case")
        self.assertIsNone(initial_query("hidden prompt [USR]: not a valid prefix"))

    def test_catalog_allowlist_discards_generated_descriptions_and_ratings(self):
        product = safe_product({"parent_asin": "a", "title": "phone", "details": '{"Brand":"A"}',
                                "generated_description": "HIDDEN", "visual_description": "HIDDEN",
                                "rating_number": 100, "reviews": "HIDDEN"})
        self.assertNotIn("generated_description", product)
        self.assertNotIn("HIDDEN", metadata_text(product))

    def test_cohorts_are_stable_under_input_order(self):
        users = [f"u{i:02}" for i in range(20)]
        self.assertEqual(assign_cohorts(users), assign_cohorts(reversed(users)))
        self.assertEqual(list(assign_cohorts(users).values()).count("test"), 2)

    def test_metrics_average_personas_within_user_first(self):
        rows = [{"user_id": "a", "rank": 1}, {"user_id": "a", "rank": 1},
                {"user_id": "b", "rank": 100}]
        self.assertEqual(aggregate(rows)["hr@10"], .5)

    def test_seen_items_are_excluded_and_ties_are_stable(self):
        self.assertEqual(rank_target(np.array([99., 1., 1.]), 2, {0}), 2)
        with self.assertRaises(ValueError):
            rank_target(np.array([1., 2.]), 0, {0})


if __name__ == "__main__":
    unittest.main()
