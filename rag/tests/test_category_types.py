from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.retrieval.queries.category_types import category_type, coverage_report, schema_from_data


def item(*path, **details):
    return {"categories": list(path), "details": details}


class CategoryTypeTests(unittest.TestCase):
    def test_family_and_role_come_from_the_category_path(self):
        self.assertEqual(category_type(item("Electronics", "Computers & Accessories", "Computers & Tablets", "Laptops"))
                         ["family"], "computers")
        self.assertEqual(category_type(item("Electronics", "Computers & Accessories", "Computers & Tablets", "Laptops"))
                         ["role"], "device")
        self.assertEqual(category_type(item("Electronics", "Computers & Accessories", "Tablet Accessories", "Cases"))
                         ["role"], "accessory")
        self.assertEqual(category_type(item("Electronics", "Computers & Accessories", "Computer Components", "Internal Hard Drives"))
                         ["role"], "component")
        self.assertEqual(category_type(item("Electronics", "Electronics Warranties", "Computer Warranties"))["role"], "service")
        self.assertEqual(category_type(item("Electronics", "Television & Video", "Televisions"))["family"], "tv_video")

    def test_missing_or_unknown_paths_are_not_guessed(self):
        self.assertEqual(category_type({"categories": []})["family"], "unassigned")
        self.assertEqual(category_type({"categories": ["Electronics"]})["family"], "unassigned")
        self.assertEqual(category_type(item("Electronics", "Something New", "Gizmos"))["family"], "other")

    def test_schema_keeps_fields_common_inside_a_family_and_drops_generic_ones(self):
        laptops = [item("Electronics", "Computers & Accessories", "Computers & Tablets", "Laptops",
                        RAM="8 GB", **{"Item Weight": "4 pounds"}) for _ in range(30)]
        schema = schema_from_data(laptops, min_items=10)
        self.assertEqual(schema["computers"]["items"], 30)
        self.assertIn("RAM", schema["computers"]["attributes"])
        self.assertNotIn("Item Weight", schema["computers"]["attributes"])

    def test_coverage_report_counts_assigned_items_and_rule_agreement(self):
        products = [item("Electronics", "Camera & Photo", "Digital Cameras"), {"categories": []}]
        report = coverage_report(products, rule_types=lambda p: [])
        self.assertEqual((report["items"], report["with_family"]), (2, 1))
        self.assertEqual(report["rule_detector_silent"], 2)


if __name__ == "__main__":
    unittest.main()
