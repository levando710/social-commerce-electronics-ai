from pathlib import Path
import unittest

from tools.renumber_report_tables import apply_mapping, build_mapping

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = """**Bảng 2.1. A.**

Xem Bảng 2.7b và Bảng 2.7 ở trên, cũng như Bảng C.1.

**Bảng 2.7. B.**

**Bảng 2.7b. C.**

**Bảng 2.8. D.**

**Bảng C.1. phụ lục.**
"""


class RenumberTests(unittest.TestCase):
    def test_sequential_numbers_follow_caption_order_and_appendix_is_untouched(self):
        mapping = build_mapping(SAMPLE)
        self.assertEqual(mapping, {"2.1": "2.1", "2.7": "2.2", "2.7b": "2.3", "2.8": "2.4"})
        result = apply_mapping(SAMPLE, mapping)
        self.assertIn("Xem Bảng 2.3 và Bảng 2.2 ở trên, cũng như Bảng C.1.", result)
        self.assertIn("**Bảng 2.4. D.**", result)
        self.assertIn("**Bảng C.1. phụ lục.**", result)

    def test_longer_label_is_not_mistaken_for_its_prefix(self):
        mapping = {"2.7": "2.2", "2.7b": "2.3"}
        self.assertEqual(
            apply_mapping("Bảng 2.7b rồi Bảng 2.7.", mapping), "Bảng 2.3 rồi Bảng 2.2."
        )

    def test_figures_are_renumbered_independently_of_tables(self):
        text = (
            "**Hình 2.1. A.**"
            + chr(10)
            + "**Hình 2.1b. B.**"
            + chr(10)
            + "Xem Hình 2.1b và Bảng 2.1."
            + chr(10)
            + "**Bảng 2.1. T.**"
            + chr(10)
        )
        mapping = build_mapping(text, "Hình")
        self.assertEqual(mapping, {"2.1": "2.1", "2.1b": "2.2"})
        result = apply_mapping(text, mapping, "Hình")
        self.assertIn("Xem Hình 2.2 và Bảng 2.1.", result)

    def test_reference_without_caption_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_mapping("Xem Bảng 2.9", {"2.1": "2.1"})

    def test_duplicate_caption_is_rejected(self):
        with self.assertRaises(ValueError):
            build_mapping("**Bảng 2.1. A.**\n**Bảng 2.1. B.**\n")


if __name__ == "__main__":
    unittest.main()
