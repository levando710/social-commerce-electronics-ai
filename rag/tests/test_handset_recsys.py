from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.preparation.prepare_handset_recsys import select_catalog, split_with_context, export_split
from phone_assistant.recommendation.recsys_data import InteractionData


class HandsetDatasetTests(unittest.TestCase):
    def test_catalog_precedence_and_metadata_allowlist_are_independent_of_feedback(self):
        product = {"parent_asin": "p", "title": "Samsung phone", "details": {}, "rating_number": 999, "reviews": ["secret"]}
        catalog, source = select_catalog([("first", [product], []),
            ("second", [{**product, "title": "Another phone"}, {"parent_asin": "c", "title": "case for iPhone 13"}], [])])
        self.assertEqual(set(catalog), {"p"})
        self.assertEqual(source["p"], "first")
        self.assertNotIn("rating_number", catalog["p"][0])
        self.assertNotIn("reviews", catalog["p"][0])

    def test_low_degree_users_supply_context_without_extra_holdouts(self):
        rows = [{"user_id": user, "item_id": str(i), "timestamp": i} for user, length in (("a", 1), ("b", 2), ("c", 4))
                for i in range(1, length + 1)]
        splits, tied = split_with_context(rows)
        self.assertEqual(len(splits["train"]), 5)
        self.assertEqual(splits["validation"], [{"user_id": "c", "item_id": "3", "timestamp": 3}])
        self.assertEqual(splits["test"], [{"user_id": "c", "item_id": "4", "timestamp": 4}])
        self.assertEqual(tied, 0)

    def test_equal_last_timestamps_do_not_create_artificial_holdout_order(self):
        rows = [{"user_id": "a", "item_id": str(i), "timestamp": t} for i, t in enumerate((1, 2, 2))]
        splits, tied = split_with_context(rows)
        self.assertEqual(splits["validation"], [])
        self.assertEqual(splits["test"], [])
        self.assertEqual(len(splits["train"]), 3)
        self.assertEqual(tied, 1)

    def test_export_is_frozen_readable_and_does_not_copy_review_enriched_text(self):
        products = [{"parent_asin": str(i), "title": f"Samsung phone {i}", "details": {}, "reviews": ["private review"]}
                    for i in range(4)]
        documents = [{"doc_id": str(i), "text": "private review and average rating", "reviews": ["private review"]} for i in range(4)]
        catalog, provenance = select_catalog([("fixture", products, documents)])
        events = [{"user_id": "u", "item_id": str(i), "timestamp": i + 1, "rating": 5., "feedback_type": "positive_review"} for i in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary, _ = export_split(directory, catalog, provenance, events, {"archive_audit": {}})
            frozen = InteractionData(directory)
            self.assertEqual(summary["split_events"], {"train": 2, "validation": 1, "test": 1})
            self.assertNotIn("private review", (directory / "documents.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("average rating", (directory / "documents.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(frozen.n_items, 4)
            (directory / "train.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                InteractionData(directory)
