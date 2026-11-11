from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores


class PairedAnalysisTests(unittest.TestCase):
    def test_constant_paired_difference_has_degenerate_interval(self):
        result = paired_bootstrap({"a": 0., "b": .5}, {"a": .25, "b": .75}, repetitions=100)
        self.assertEqual(result["mean_delta"], .25)
        self.assertEqual(result["percentile_ci95"], [.25, .25])

    def test_unpaired_users_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_bootstrap({"a": 0}, {"b": 1}, repetitions=100)

    def test_seeds_averaged_before_user_bootstrap_and_missing_users_rejected(self):
        runs = [{"test_ranks": [{"user_index": 0, "rank": 1}, {"user_index": 1, "rank": 100}]},
                {"test_ranks": [{"user_index": 0, "rank": 100}, {"user_index": 1, "rank": 1}]}]
        self.assertEqual(user_scores(runs, "hr@10"), {"0": .5, "1": .5})
        runs[1]["test_ranks"].pop()
        with self.assertRaises(ValueError):
            user_scores(runs, "hr@10")


if __name__ == "__main__":
    unittest.main()
