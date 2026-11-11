from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.retrieval.queries.product_constraints import ProductQueryPlanner, connected_facts, connected_verdict, model_names, product_types

CASE = {"doc_id": "case", "title": "Spigen case for iPhone 13 Pro Max 128GB", "product_image_paths": ["photo"],
        "details": {"Brand": "Spigen", "Compatible Phone Models": "iPhone 13 Pro Max"}}
PHONE = {"doc_id": "phone", "title": "Samsung Galaxy A14 128GB unlocked", "product_image_paths": ["photo"],
         "details": {"Brand": "Samsung", "Memory Storage Capacity": "128GB", "RAM": "8GB"}}


class ProductConstraintTests(unittest.TestCase):
    def test_case_maker_and_compatible_phone_are_separate(self):
        planner = ProductQueryPlanner([CASE, PHONE])
        plan = planner("Spigen case for iPhone 13 Pro Max")
        facts = connected_facts(CASE)
        self.assertEqual(facts["manufacturer_name"], "spigen")
        self.assertNotIn("apple", facts["brands"])
        self.assertEqual(plan["constraints"]["compatibility_models"], ["iphone 13 pro max"])
        self.assertTrue(connected_verdict(facts, plan["constraints"])["eligible"])
        self.assertIsNone(facts["storage_gb"])

    def test_pro_does_not_match_pro_max_and_hardware_is_not_inferred_for_case(self):
        planner = ProductQueryPlanner([CASE])
        self.assertFalse(connected_verdict(connected_facts(CASE), planner("case for iPhone 13 Pro")["constraints"])["eligible"])
        verdict = connected_verdict(connected_facts(CASE), planner("case for iPhone 13 Pro Max 128GB")["constraints"])
        self.assertFalse(verdict["eligible"])
        self.assertIn("compatibility_storage_gb", verdict["unknown"])

    def test_explicit_device_switch_resets_accessory_manufacturer_and_compatibility(self):
        planner = ProductQueryPlanner([CASE, PHONE])
        previous = planner("Spigen case for iPhone 13 Pro Max")
        current = planner("Samsung phone 128GB", previous)
        self.assertEqual(current["constraints"]["product_type"], "phone")
        self.assertEqual(current["constraints"]["manufacturer_names"], ["samsung"])
        self.assertNotIn("compatibility_models", current["constraints"])
        self.assertTrue(connected_verdict(connected_facts(PHONE), current["constraints"])["eligible"])

    def test_accessory_followup_retains_kind_and_updates_exact_model(self):
        planner = ProductQueryPlanner([CASE])
        first = planner("case for iPhone 13")
        second = planner("for iPhone 13 Pro Max", first)
        self.assertEqual(second["constraints"]["product_type"], "case")
        self.assertEqual(second["constraints"]["compatibility_models"], ["iphone 13 pro max"])

    def test_product_kinds_and_connector_requirements(self):
        self.assertEqual(product_types("case-friendly tempered glass screen protector for iPhone"), ["screen_protector"])
        self.assertEqual(product_types("In case I need a Samsung phone"), ["phone"])
        planner = ProductQueryPlanner([])
        query = planner("USB-C to Lightning cable")
        facts = connected_facts({"title": "USB-C cable", "details": {"Connector Type": "USB-C"}})
        self.assertFalse(connected_verdict(facts, query["constraints"])["eligible"])
        self.assertEqual(model_names("iPhone13ProMax"), ["iphone 13 pro max"])

    def test_existing_phone_is_not_automatically_a_request_to_buy_a_phone(self):
        planner = ProductQueryPlanner([])
        for text in ("I need something for my phone", "something to protect my phone", "make using my phone easier"):
            self.assertNotIn("product_type", planner(text)["constraints"])
        self.assertEqual(planner("I need a new phone")["constraints"]["product_type"], "phone")

    def test_battery_thousands_and_ranges_do_not_turn_into_zero_or_exact_match(self):
        planner = ProductQueryPlanner([])
        facts = connected_facts({"title": "Rugged phone 10050mAh", "details": {}})
        plan = planner("rugged phone over 10,000 mAh")
        self.assertEqual(plan["constraints"]["battery_min_mah"], 10001)
        self.assertTrue(connected_verdict(facts, plan["constraints"])["eligible"])
        plan = planner("power bank around 10,000 to 13,000 mAh")
        self.assertEqual(plan["constraints"]["battery_min_mah"], 10000)
        self.assertEqual(plan["constraints"]["battery_max_mah"], 13000)
        self.assertNotIn("battery_mah", plan["constraints"])
        self.assertNotIn("battery_mah", planner("phone around 10,000 mAh")["constraints"])
        self.assertEqual(connected_facts({"title": "Power bank 13,000mAh", "details": {}})["battery_mah"], 13000)
        self.assertFalse(connected_verdict({**facts, "battery_mah": 9000}, planner("phone at least 10,000mAh")["constraints"])["eligible"])


if __name__ == "__main__":
    unittest.main()
