from collections import defaultdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.recommendation.conversational_joint import ConversationalJointRanker, QuerySupervision
from phone_assistant.recommendation.recsys_data import InteractionData
from phone_assistant.training.train_conversational_joint import train_one


def fixture(directory=None):
    data = SimpleNamespace(n_users=3, n_items=5, item_ids=list("abcde"), user_ids=["u0", "u1", "u2"],
        maps={"users": {f"u{i}": i for i in range(3)}, "items": {v: i for i, v in enumerate("abcde")}},
        pairs={"train": np.asarray([[0, 0], [0, 1], [1, 1], [2, 2]]),
               "validation": np.asarray([[1, 3]]), "test": np.asarray([[2, 4]])},
        train_seen=defaultdict(set, {0: {0, 1}, 1: {1}, 2: {2}}), validation_seen=defaultdict(set, {1: {3}}),
        item_degree=np.asarray([1, 2, 1, 0, 0]), directory=directory)
    data.negatives = lambda users, rng: InteractionData.negatives(data, users, rng)
    features = {"text": np.eye(5, dtype=np.float32), "image": np.eye(5, dtype=np.float32),
        "image_mask": np.ones(5, dtype=bool), "specs": {v: ["brand=one"] for v in "abcde"}}
    queries = [{"dialogue_id": f"q{i}", "user_id": f"u{i}", "split": split, "query": "visible user words"}
               for i, split in enumerate(("train", "validation", "test"))]
    labels = [{"user_id": f"u{i}", "cohort": split, "item_id": item}
              for i, (split, item) in enumerate((("train", "b"), ("validation", "d"), ("test", "e")))]
    supervision = QuerySupervision(data, np.eye(5, dtype=np.float32)[:3], queries, labels)
    return data, features, supervision, queries, labels


class ConversationalJointTests(unittest.TestCase):
    def test_goal_joins_are_split_checked_and_train_sampler_excludes_heldout_queries(self):
        data, _, supervision, queries, labels = fixture()
        inputs, users, targets = supervision.batch("train", [0])
        self.assertEqual(users.tolist(), [0]); self.assertEqual(targets.tolist(), [1])
        self.assertEqual(inputs.shape, (1, 5))
        labels[0]["item_id"] = "d"
        with self.assertRaisesRegex(ValueError, "designated frozen split"):
            QuerySupervision(data, supervision.vectors, queries, labels)

    def test_gradient_decoupling_keeps_query_head_trainable_without_graph_feedback(self):
        data, features, supervision, _, _ = fixture()
        inputs, users, targets = supervision.batch("train", [0])
        for detach in (True, False):
            model = ConversationalJointRanker(data, features, dim=4, detach_items=detach, alignment_weight=0.)
            model.query_losses(inputs, users, targets)["total"].backward()
            self.assertGreater(float(model.query_projection.weight.grad.abs().sum()), 0.)
            gradient = model.core.item_embedding.weight.grad
            self.assertEqual(gradient is None, detach)
            if not detach:
                self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_resume_preserves_graph_and_query_randomness_and_frozen_features(self):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, features, supervision, _, _ = fixture(root)
            before = features["text"].copy()
            args = SimpleNamespace(resume=False, dim=4, layers=2, epochs=4, batch_size=2,
                                   query_batch_size=1, max_epochs_this_invocation=0)
            contract = {"method": "joint_bidirectional", "seed": 42}
            full = train_one("joint_bidirectional", 42, data, features, supervision, args, root / "full", contract)
            args.max_epochs_this_invocation = 2
            self.assertIsNone(train_one("joint_bidirectional", 42, data, features, supervision, args, root / "resumed", contract))
            self.assertFalse((root / "resumed/training.json").exists())
            args.resume = True; args.max_epochs_this_invocation = 0
            resumed = train_one("joint_bidirectional", 42, data, features, supervision, args, root / "resumed", contract)
            with np.load(root / "full/embeddings.npz") as a, np.load(root / "resumed/embeddings.npz") as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(full["loss_history"], resumed["loss_history"])
            self.assertEqual(full["query_test"], resumed["query_test"])
            np.testing.assert_array_equal(before, features["text"])


if __name__ == "__main__":
    unittest.main()
