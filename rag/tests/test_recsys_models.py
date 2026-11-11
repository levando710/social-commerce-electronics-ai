import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.recommendation.recsys_data import InteractionData, metadata_text
from phone_assistant.recommendation.recsys_models import BM3, LightGCN, METHODS, info_nce, make_model, normalized_adjacency, typed_graph
from phone_assistant.training.train_recsys import evaluate_vectors


def tiny_data():
    return SimpleNamespace(n_users=2, n_items=3, item_ids=["a", "b", "c"],
        pairs={"train": np.array([[0, 0], [1, 1]]), "validation": np.array([[0, 1]]),
               "test": np.array([[1, 2]])},
        train_seen={0: {0}, 1: {1}}, validation_seen={0: {1}, 1: set()}, item_degree=np.array([1, 1, 0]))


def tiny_features():
    return {"text": np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
            "image": np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32),
            "image_mask": np.array([True, True, False]),
            "specs": {"a": ["brand=apple"], "b": ["brand=samsung"], "c": ["brand=apple"]}}


class GraphTests(unittest.TestCase):
    def test_bm3_training_never_mutates_shared_numpy_feature_store(self):
        features = tiny_features()
        original = {key: features[key].copy() for key in ("text", "image", "image_mask")}
        model = BM3(tiny_data(), features, dim=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        loss = model.loss(torch.tensor([0, 1]), torch.tensor([0, 1]), None)
        loss["total"].backward()
        optimizer.step()
        for key in original:
            np.testing.assert_array_equal(features[key], original[key])
        self.assertFalse(np.array_equal(model.text_embedding.weight.detach().numpy(), original["text"]))

    def test_lightgcn_propagation_matches_hand_computed_train_graph(self):
        model = LightGCN(tiny_data(), dim=2, layers=1)
        with torch.no_grad():
            model.user_embedding.weight.copy_(torch.tensor([[2., 0], [0, 2.]]))
            model.item_embedding.weight.copy_(torch.tensor([[4., 0], [0, 4.], [1., 1.]]))
        users, items = model.forward()
        torch.testing.assert_close(users, torch.tensor([[3., 0], [0, 3.]]))
        torch.testing.assert_close(items, torch.tensor([[3., 0], [0, 3.], [.5, .5]]))

    def test_typed_graph_uses_train_edges_and_actual_image_nodes_only(self):
        edges, offsets, images, schema = typed_graph(tiny_data(), tiny_features())
        user_edges = edges[edges[:, 0] < 2].tolist()
        self.assertEqual(user_edges, [[0, 2], [1, 3]])
        self.assertEqual(schema["nodes"]["image"], 2)
        self.assertEqual(schema["nodes"]["spec"], 2)
        self.assertNotIn([0, 3], user_edges)  # validation edge
        self.assertNotIn([1, 4], user_edges)  # test edge
        _, _, no_images, schema = typed_graph(tiny_data(), tiny_features(), use_image=False)
        self.assertEqual(len(no_images), 0)
        self.assertEqual(schema["undirected_edges"]["item_image"], 0)

    def test_dropout_keeps_adjacency_symmetric(self):
        torch.manual_seed(42)
        matrix = normalized_adjacency(torch.tensor([[0, 1], [1, 2], [2, 3]]), 4, .5).to_dense()
        torch.testing.assert_close(matrix, matrix.T)
        self.assertTrue(torch.isfinite(matrix).all())

    def test_contrastive_loss_prefers_correct_node_alignment(self):
        vectors = torch.eye(4, requires_grad=True)
        aligned = info_nce(vectors, vectors)
        permuted = info_nce(vectors, vectors.roll(1, 0))
        self.assertLess(float(aligned.detach()), float(permuted.detach()))
        aligned.backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())

    def test_all_models_backpropagate_and_export_equivalent_scores(self):
        torch.set_num_threads(2)
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(7)
                model = make_model(method, tiny_data(), tiny_features(), dim=4)
                loss = model.loss(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 2]))
                self.assertTrue(torch.isfinite(loss["total"]))
                loss["total"].backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.eval()
                users, items, bias = model.export_vectors()
                torch.testing.assert_close(model.score_all(torch.tensor([0, 1])), users @ items.T + bias)

    def test_bm3_target_has_no_gradient(self):
        online = torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3, requires_grad=True)
        BM3.distance(online, target).backward()
        self.assertIsNotNone(online.grad)
        self.assertIsNone(target.grad)


class ProtocolTests(unittest.TestCase):
    def test_catalog_text_does_not_include_feedback_fields(self):
        product = {"title": "phone", "average_rating": "LEAK_RATING", "reviews": ["LEAK_REVIEW"],
                   "rating_number": "LEAK_COUNT", "details": {"Customer Reviews": "LEAK_DETAILS", "Color": "Blue"}}
        text = metadata_text(product)
        self.assertNotIn("LEAK", text)
        self.assertIn("Color: Blue", text)

    def test_full_catalog_ranking_masks_seen_and_counts_cold_target(self):
        data = tiny_data()
        users, items = np.ones((2, 1)), np.array([[3.], [2.], [1.]])
        metrics, ranks = evaluate_vectors(users, items, np.zeros(3), data, "test", k_values=(1, 2))
        self.assertEqual(ranks[0]["rank"], 2)
        self.assertEqual(metrics["all"]["hr@1"], 0)
        self.assertEqual(metrics["all"]["hr@2"], 1)
        self.assertEqual(metrics["cold_item"]["users"], 1)

    def test_real_popularity_protocol_matches_existing_baseline(self):
        data = InteractionData(Path(__file__).resolve().parents[1] / 'data/fixtures/recommendation')
        metrics, _ = evaluate_vectors(np.ones((data.n_users, 1)), data.item_degree[:, None],
                                      np.zeros(data.n_items), data, "test")
        baseline = json.loads((data.directory / "popularity_metrics.json").read_text())["test"]
        for group in baseline:
            self.assertEqual(metrics[group]["users"], baseline[group]["users"])
            for key in ("hr@5", "ndcg@5", "hr@10", "ndcg@10"):
                if baseline[group][key] is None:
                    self.assertIsNone(metrics[group][key])
                else:
                    self.assertAlmostEqual(metrics[group][key], baseline[group][key])


if __name__ == "__main__":
    unittest.main()
