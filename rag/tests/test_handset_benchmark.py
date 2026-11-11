from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.data.preparation.prepare_handset_benchmark import gold_fields, judge_fields
from phone_assistant.evaluation.metrics.evaluate_handset_rag import metrics


class HandsetBenchmarkTests(unittest.TestCase):
    def test_gold_uses_raw_fields_and_leaves_missing_metadata_unjudged(self):
        fields = gold_fields({"title": "8GB RAM 128GB phone", "reviews": ["8GB RAM"], "details": {"Brand": "Samsung"}})
        self.assertIsNone(fields["ram_gb"])
        self.assertEqual(judge_fields(fields, {"ram_gb": 8}), "unjudged")
        self.assertEqual(judge_fields(fields, {"brand": "apple", "ram_gb": 8}), "irrelevant")

    def test_raw_gold_units_do_not_depend_on_serving_parser(self):
        fields = gold_fields({"details": {"RAM": "8192 MB", "Memory Storage Capacity": "1 TB", "Battery Capacity": "4000 Milliamp Hours"}})
        self.assertEqual(fields["ram_gb"], 8)
        self.assertEqual(fields["storage_gb"], 1024)
        self.assertEqual(judge_fields(fields, {"battery_min_mah": 4000}), "relevant")

    def test_macro_average_does_not_treat_paraphrases_as_independent_groups(self):
        rows = [{"intent_id": "a", "ids": ["yes"]}, {"intent_id": "a", "ids": []}, {"intent_id": "b", "ids": ["yes"]}]
        qrels = {name: {"yes": "relevant"} for name in ("a", "b")}
        result = metrics(rows, qrels, 1, k=1)
        self.assertEqual(result["precision@1"], .75)
        self.assertEqual(result["intents"], 2)
        self.assertEqual(metrics([rows[1]], qrels, 1, k=1)["ndcg@1"], 0.)
