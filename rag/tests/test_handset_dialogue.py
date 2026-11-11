from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.conversation.handset_dialogue import HandsetQueryPlanner, handset_facts

DOC = {"doc_id": "p", "title": "Samsung phone 128GB", "details": {"Brand": "Samsung"}}


class HandsetDialogueTests(unittest.TestCase):
    def test_explicit_unit_normalization_and_processor_quotes(self):
        product = {"details": {"RAM": "8192 MB", "Memory Storage Capacity": "128 GB", "Battery Capacity": "4,000 Milliamp Hours"},
                   "features": ["Powered by Snapdragon 888 processor"], "reviews": ["Exynos 9999"]}
        facts = handset_facts(DOC, product)
        self.assertEqual((facts["ram_gb"], facts["storage_gb"], facts["battery_mah"]), (8, 128, 4000))
        self.assertEqual(facts["processor_claims"], ["Snapdragon 888"])
        self.assertEqual(facts["fact_evidence"]["ram_gb"][0]["quote"], "8192 MB")
        self.assertNotIn("Exynos", str(facts))

    def test_ambiguous_specs_remain_unknown(self):
        facts = handset_facts(DOC, {"details": {"RAM": "4GB or 8GB"}})
        self.assertIsNone(facts["ram_gb"])

    def test_suspicious_ram_is_quarantined_without_inventing_a_replacement(self):
        facts = handset_facts(DOC, {"details": {"RAM": "128 GB", "Memory Storage Capacity": "128 GB"}})
        self.assertIsNone(facts["ram_gb"])
        self.assertIn("ram_capacity_requires_review", facts["fact_warnings"])
        self.assertEqual(facts["fact_evidence"]["ram_gb"][0]["quote"], "128 GB")

    def test_numeric_citations_resolve_only_to_supplied_contexts(self):
        from phone_assistant.llm.gemini_client import normalize_answer_citations
        answer, audit = normalize_answer_citations("Storage [1]; another [3]", [{"doc_id": "B012345678"}])
        self.assertEqual(answer, "Storage [B012345678]; another [3]")
        self.assertEqual(audit, {"normalized": 1, "unresolved_numeric": [3]})

    def test_followup_rewrite_carries_constraints_and_soft_needs(self):
        planner = HandsetQueryPlanner([DOC])
        first = planner("Samsung phone 128GB 8GB RAM để chơi game")
        second = planner("còn 256GB thì sao", first)
        self.assertEqual(second["constraints"]["storage_gb"], 256)
        self.assertEqual(second["constraints"]["ram_gb"], 8)
        self.assertIn("8 GB RAM", second["query_rewrite"])
        self.assertIn("samsung", second["query_rewrite"])
        self.assertIn("gaming", second["preferences"])

    def test_brand_negation_removes_redundant_maker_and_old_model(self):
        planner = HandsetQueryPlanner([DOC])
        first = planner("Samsung Galaxy S21 phone 128GB")
        second = planner("không phải Samsung", first)
        self.assertEqual(second["constraints"]["brands"], [])
        self.assertEqual(second["constraints"]["avoid_brands"], ["samsung"])
        self.assertNotIn("manufacturer_names", second["constraints"])
        self.assertNotIn("model_terms", second["constraints"])
        self.assertEqual(planner("chọn Samsung", second)["constraints"]["avoid_brands"], [])

    def test_reset_clears_explicit_and_soft_preferences(self):
        planner = HandsetQueryPlanner([DOC])
        reset = planner("bắt đầu lại Nokia", planner("Samsung 128GB để chơi game"))
        self.assertNotIn("storage_gb", reset["constraints"])
        self.assertEqual(reset["preferences"], [])
