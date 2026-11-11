import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.recommendation.anchored_joint import AnchoredJointRanker, VARIANTS, ranking_scores, semantic_scores
from phone_assistant.training.train_anchored_joint import train_one, seal_selection, ALPHAS, SEEDS
from test_conversational_joint import fixture


class AnchoredJointTests(unittest.TestCase):
    def test_residual_is_bounded_and_zero_alpha_preserves_semantic_scores(self):
        rng = np.random.default_rng(42)
        query, catalog = rng.normal(size=(3, 5)), rng.normal(size=(7, 5))
        base = semantic_scores(query, catalog)
        weight, items = rng.normal(size=(4, 5)), rng.normal(size=(7, 4))
        np.testing.assert_array_equal(ranking_scores(weight, items, query, base, 0), base)
        result = ranking_scores(weight, items, query, base, .3)
        self.assertLessEqual(float(np.abs(result - base).max()), .3 + 1e-6)
        self.assertTrue(np.isfinite(ranking_scores(np.zeros_like(weight), items, query, base, .3)).all())

    def test_query_gradients_reach_graph_only_in_joint_arms(self):
        data, features, supervision, _, _ = fixture()
        query, users, targets = supervision.batch("train", [0])
        base = torch.from_numpy(semantic_scores(query.numpy(), features["text"]))
        for detach in (True, False):
            torch.manual_seed(42)
            model = AnchoredJointRanker(data, features, alpha=.3, dim=4, detach_items=detach)
            model.query_losses(query, users, targets, base)["total"].backward()
            self.assertGreater(float(model.query_projection.weight.grad.abs().sum()), 0)
            graph_gradient = model.core.item_embedding.weight.grad
            self.assertEqual(graph_gradient is None, detach)
            if not detach:
                self.assertGreater(float(graph_gradient.abs().sum()), 0)

    def test_resume_matches_and_fitting_never_scores_test(self):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, features, supervision, _, _ = fixture(root)
            base = semantic_scores(supervision.vectors, features["text"])
            before = features["text"].copy()
            args = SimpleNamespace(resume=False, dim=4, layers=2, epochs=4, batch_size=2,
                query_batch_size=1, eval_every=2, max_epochs_this_invocation=0)
            contract = {"test": "resume fixture"}
            from phone_assistant.training.train_recsys import evaluate_vectors
            from phone_assistant.recommendation.anchored_joint import evaluate_queries
            def rec_guard(*values):
                self.assertEqual(values[-1], "validation")
                return evaluate_vectors(*values)
            def query_guard(*values):
                self.assertIn(values[-2], ("train", "validation"))
                return evaluate_queries(*values)
            with patch('phone_assistant.training.train_anchored_joint.evaluate_vectors', side_effect=rec_guard), patch('phone_assistant.training.train_anchored_joint.evaluate_queries', side_effect=query_guard):
                full = train_one("residual_bidirectional", .3, 42, data, features, supervision, base, args, root / "full", contract)
                args.max_epochs_this_invocation = 2
                self.assertIsNone(train_one("residual_bidirectional", .3, 42, data, features, supervision, base, args, root / "resume", contract))
                args.resume = True; args.max_epochs_this_invocation = 0
                resumed = train_one("residual_bidirectional", .3, 42, data, features, supervision, base, args, root / "resume", contract)
            self.assertEqual(full["loss_history"], resumed["loss_history"])
            self.assertFalse(full["test_evaluated"])
            np.testing.assert_array_equal(before, features["text"])
            with np.load(root / "full/embeddings.npz") as first, np.load(root / "resume/embeddings.npz") as second:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])

    def test_alpha_selection_requires_full_grid_and_uses_validation_only(self):
        runs = [{"alpha": alpha, "method": method, "seed": seed, "selection_score": .2 if alpha == .3 else .1,
            "test": {"deliberately_misleading_score": 100 if alpha == .1 else 0}}
            for alpha in ALPHAS for method in VARIANTS for seed in SEEDS]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "training_grid.json").write_text(json.dumps({"runs": runs}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                seal_selection(runs[:-1], root)
            result = seal_selection(runs, root)
            self.assertEqual(result["alpha"], .3)
            self.assertFalse(result["test_evaluated_before_selection"])
            self.assertEqual(seal_selection(runs, root), result)


if __name__ == "__main__":
    unittest.main()
