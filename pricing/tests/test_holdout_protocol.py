from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phone_pricing.training.evaluate_holdout_protocol import median_baseline, metrics, split_three_way


def frame(rows=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"model": rng.choice(["a", "b"], rows), "capacity_gb": rng.choice([64, 128], rows),
                         "log_price": rng.normal(15, 0.3, rows)})


class HoldoutProtocolTests(unittest.TestCase):
    def test_validation_never_overlaps_test_and_test_is_seed_independent(self):
        data = frame()
        train, validation, test = split_three_way(data, 42)
        _, _, other_test = split_three_way(data, 43)
        self.assertTrue(test.index.equals(other_test.index))
        self.assertFalse(set(validation.index) & set(test.index))
        self.assertFalse(set(train.index) & set(test.index))
        self.assertFalse(set(train.index) & set(validation.index))

    def test_median_baseline_uses_group_median_with_global_fallback(self):
        train = pd.DataFrame({"model": ["a", "a", "a"], "capacity_gb": [64, 64, 64], "log_price": [1.0, 2.0, 9.0]})
        test = pd.DataFrame({"model": ["a", "z"], "capacity_gb": [64, 64]})
        self.assertEqual(list(median_baseline(train, test)), [2.0, 2.0])

    def test_metrics_are_computed_in_currency_units(self):
        result = metrics(np.log1p(np.array([100.0, 200.0])), np.log1p(np.array([110.0, 180.0])))
        self.assertAlmostEqual(result["mape_percent"], 10.0, places=6)
        self.assertAlmostEqual(result["rmse_vnd"], np.sqrt((100 + 400) / 2), places=6)


if __name__ == "__main__":
    unittest.main()
