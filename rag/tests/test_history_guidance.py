from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.metrics.evaluate_history_guidance import history_profiles, rank_metrics


class HistoryGuidanceTests(unittest.TestCase):
    def test_profile_uses_only_supplied_train_edges_and_empty_user_remains_zero(self):
        text = np.eye(3, dtype=np.float32)
        profiles = history_profiles(np.asarray([[0, 0], [0, 1], [1, 1]]), text, 3)
        self.assertEqual(float(profiles[0, 2]), 0)
        np.testing.assert_array_equal(profiles[2], np.zeros(3))
        np.testing.assert_allclose(profiles[0], [2 ** -.5, 2 ** -.5, 0], atol=1e-6)

    def test_two_personas_are_averaged_within_user_before_macro_average(self):
        rows = [{"query_index": i, "user_index": user, "item_index": 0} for i, user in enumerate([0, 0, 1])]
        scores = np.zeros((3, 12), dtype=np.float32)
        scores[0, 0] = 1
        scores[1, 0] = -1
        scores[2, 0] = 1
        metrics, _ = rank_metrics(scores, rows)
        self.assertEqual(metrics["hr@10"], .75)


if __name__ == "__main__":
    unittest.main()
