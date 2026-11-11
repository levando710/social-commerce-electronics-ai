import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.evaluation.analysis.analyze_category_recsys import PAIRS, analyze, parse_pairs


def run(method, seed, ranks):
    return {"method": method, "seed": seed, "test": {"cold_item": {"users": 1, "hr@10": 0.0}},
            "test_ranks": [{"user_index": user, "item_index": 0, "rank": rank} for user, rank in enumerate(ranks)]}


class CategoryAnalysisTests(unittest.TestCase):
    def test_delta_sign_follows_second_minus_first_and_output_is_not_overwritten(self):
        methods = {name for pair in PAIRS for name in pair}
        runs = []
        for seed in (1, 2):
            for name in methods:
                good = name in ("graph_multimodal", "graph_text_only")  # better ranks than the others
                runs.append(run(name, seed, [1 if good else 50] * 40))
        experiment = {"runs": runs, "popularity": {"test": {k: {"hr@10": 0.1} for k in ("all", "warm_item", "cold_item")}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "experiment.json"
            path.write_text(json.dumps(experiment), encoding="utf-8")
            result = analyze(path, Path(folder) / "out")
            delta = result["paired_bootstrap"]["graph_multimodal_gcl_minus_graph_multimodal"]["hr@10"]["mean_delta"]
            self.assertLess(delta, 0)  # GCL ranks worse than graph_multimodal in this fixture
            with self.assertRaises(FileExistsError):
                analyze(path, Path(folder) / "out")

    def test_custom_pairs_are_parsed_and_malformed_ones_rejected(self):
        self.assertEqual(parse_pairs("lightgcn:bm3, lightgcn:graph_text_only"),
                         (("lightgcn", "bm3"), ("lightgcn", "graph_text_only")))
        for bad in ("", "a", "a:b:c", "a:"):
            with self.assertRaises(ValueError):
                parse_pairs(bad)


if __name__ == "__main__":
    unittest.main()
