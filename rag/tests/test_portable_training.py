from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.training.train_recsys_portable import fit, restore_model
from phone_assistant.recommendation.recsys_data import InteractionData
from test_recsys_models import tiny_data, tiny_features


def portable_data(directory):
    data = tiny_data()
    data.directory = directory
    data.user_ids = ["u", "v"]
    data.negatives = lambda users, rng: InteractionData.negatives(data, users, rng)
    return data


class PortableTrainingTests(unittest.TestCase):
    def test_mutating_feature_inputs_aborts_before_test_artifacts_are_written(self):
        from phone_assistant.recommendation.recsys_models import make_model
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            data = portable_data(directory)
            features = tiny_features()
            model = make_model("bm3", data, features, dim=4, layers=1)
            # Deliberately simulate an unsafe future model that aliases catalog arrays.
            model.text_embedding.weight = torch.nn.Parameter(torch.as_tensor(features["text"]))
            args = SimpleNamespace(resume=False, dim=4, layers=1, reg=1e-4,
                ssl_weight=.05, learning_rate=.01, device="cpu", epochs=1,
                eval_every=1, batch_size=0, max_epochs_this_invocation=0)
            with patch('phone_assistant.training.train_recsys_portable.make_model', return_value=model):
                with self.assertRaisesRegex(ValueError, "mutated the shared feature"):
                    fit("bm3", 42, data, features, args, directory / "unsafe", {"method": "fixture"})
            self.assertFalse((directory / "unsafe/training.json").exists())
            self.assertFalse((directory / "unsafe/embeddings.npz").exists())

    def test_bm3_then_hierarchical_graph_preserves_features_and_reconstructs_selected_vectors(self):
        from phone_assistant.recommendation.recsys_models import make_model
        from phone_assistant.training.train_recsys import exported
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            data = portable_data(directory)
            features = tiny_features()
            features["spec_hierarchy"] = {"value_to_attribute": {"brand=apple": "brand", "brand=samsung": "brand"},
                                         "attribute_to_group": {"brand": "identity"}}
            original = {key: features[key].copy() for key in ("text", "image", "image_mask")}
            args = SimpleNamespace(resume=False, dim=4, layers=2, reg=1e-4,
                ssl_weight=.05, learning_rate=.01, device="cpu", epochs=2,
                eval_every=1, batch_size=0, max_epochs_this_invocation=0)
            for method in ("bm3", "graph_text_only", "graph_multimodal", "graph_multimodal_gcl"):
                fit(method, 42, data, features, args, directory / method, {"method": method})
                for key in original:
                    np.testing.assert_array_equal(features[key], original[key])
                reconstructed = make_model(method, data, features, dim=4, layers=2, reg=1e-4, ssl_weight=.05)
                state = torch.load(directory / method / "checkpoint.pt", weights_only=True)
                restore_model(reconstructed, state)
                with np.load(directory / method / "embeddings.npz") as saved:
                    for key, value in zip(("users", "items", "bias"), exported(reconstructed)):
                        np.testing.assert_allclose(value, saved[key], atol=1e-6, rtol=1e-5)

    def test_epoch_resume_matches_uninterrupted_training_including_rng_and_optimizer(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for method in ("lightgcn", "bm3", "graph_multimodal_gcl"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                data = portable_data(directory)
                args = SimpleNamespace(resume=False, dim=4, layers=1, reg=1e-4,
                    ssl_weight=.05, learning_rate=.001, device="cpu", epochs=4,
                    eval_every=2, batch_size=1, max_epochs_this_invocation=0)
                contract = {"method": method, "seed": 42, "inputs": "fixture"}
                fit(method, 42, data, tiny_features(), args, directory / "full", contract)
                args.max_epochs_this_invocation = 2
                partial = fit(method, 42, data, tiny_features(), args, directory / "resumed", contract)
                self.assertIsNone(partial)
                self.assertFalse((directory / "resumed/training.json").exists())
                args.resume = True
                args.max_epochs_this_invocation = 0
                result = fit(method, 42, data, tiny_features(), args, directory / "resumed", contract)
                with np.load(directory / "full/embeddings.npz") as full, np.load(directory / "resumed/embeddings.npz") as resumed:
                    for name in full.files:
                        np.testing.assert_array_equal(full[name], resumed[name])
                import json
                full_result = json.loads((directory / "full/training.json").read_text())
                self.assertEqual(full_result["loss_history"], result["loss_history"])
                self.assertEqual(full_result["test"], result["test"])
                self.assertEqual(result, fit(method, 42, data, tiny_features(), args, directory / "resumed", contract))

    def test_resume_rejects_changed_contract_and_completed_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            data = portable_data(directory)
            args = SimpleNamespace(resume=False, dim=4, layers=1, reg=1e-4,
                ssl_weight=.05, learning_rate=.001, device="cpu", epochs=2,
                eval_every=1, batch_size=0, max_epochs_this_invocation=1)
            contract = {"data": "original"}
            fit("lightgcn", 42, data, None, args, directory / "run", contract)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                fit("lightgcn", 42, data, None, args, directory / "run", {"data": "changed"})
            args.max_epochs_this_invocation = 0
            fit("lightgcn", 42, data, None, args, directory / "run", contract)
            (directory / "run/embeddings.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                fit("lightgcn", 42, data, None, args, directory / "run", contract)


if __name__ == "__main__":
    unittest.main()
