from pathlib import Path
import unittest

from tools import check_report_numbers as checker

ROOT = Path(__file__).resolve().parents[1]

REPORT = "### 2.6.3. x\n| a | b |\n|---|---|\n| m | 0.123456 | -0.050000 | 41.406 |\n## Thảo luận kết quả\n"


class ReportNumberTests(unittest.TestCase):
    def values(self, *numbers):
        found = set()
        for number in numbers:
            checker.collect_values(number, found)
        return found

    def test_present_values_including_negative_and_scaled_are_matched(self):
        values = self.values([0.123456, -0.05, 0.041406])  # 0.041406 s == 41.406 ms
        self.assertEqual(checker.unmatched(REPORT, values), [])

    def test_a_value_absent_from_the_sources_is_reported(self):
        values = self.values([0.123456, 0.041406])
        self.assertEqual([number for _, number in checker.unmatched(REPORT, values)], [0.05])

    def test_numbers_outside_the_results_section_are_ignored(self):
        text = "| x | 9.999999 |\n" + REPORT
        self.assertEqual(len(checker.table_numbers(text)), 3)


if __name__ == "__main__":
    unittest.main()
