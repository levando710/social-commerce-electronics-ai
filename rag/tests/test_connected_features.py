import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.common.catalog_schema import catalog_text, make_hierarchy
from phone_assistant.features.build_connected_features import atomic_json, finalize, image_urls, save_vector, saved_vector, text_hash


class ConnectedFeatureTests(unittest.TestCase):
    def test_explicit_specs_hierarchy_has_no_feedback_or_inferred_facts(self):
        product = {"parent_asin": "a", "title": "Samsung phone", "details": {
            "RAM": "8 GB", "Battery Capacity": "5000 mAh", "Customer Reviews": "LEAK",
            "Best Sellers Rank": "LEAK", "Color": "Blue"}, "reviews": "LEAK", "generated_description": "LEAK"}
        text = catalog_text(product)
        self.assertNotIn("LEAK", text)
        self.assertIn("ram: 8 gb", text)
        specs, hierarchy = make_hierarchy([product])
        self.assertIn("ram=8 gb", specs["a"])
        self.assertEqual(hierarchy["value_to_attribute"]["ram=8 gb"], "ram")
        self.assertEqual(hierarchy["attribute_to_group"]["ram"], "compute")
        self.assertNotIn("cpu", hierarchy["attribute_to_group"])

    def test_vector_resume_checks_input_and_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.assertIsNone(saved_vector(directory, "a", "input", 2))
            save_vector(directory, "a", "input", np.asarray([3., 4.]))
            np.testing.assert_allclose(saved_vector(directory, "a", "input", 2), [.6, .8])
            with self.assertRaises(ValueError):
                saved_vector(directory, "a", "changed", 2)
            with self.assertRaises(ValueError):
                save_vector(directory, "b", "input", np.zeros(2))

    def test_only_supported_public_image_urls_and_same_view_fallbacks(self):
        urls = image_urls({"images": [{"variant": "SIDE", "large": "https://m.media-amazon.com/side.jpg"},
            {"variant": "MAIN", "hi_res": "http://127.0.0.1/private", "large": "https://m.media-amazon.com/main.jpg"}]})
        self.assertEqual(urls, ["https://m.media-amazon.com/main.jpg"])

    def test_hierarchical_graph_edges_and_message_passing(self):
        import torch
        from phone_assistant.recommendation.recsys_models import make_model, typed_graph
        from test_recsys_models import tiny_data, tiny_features
        features = tiny_features()
        features["spec_hierarchy"] = {"value_to_attribute": {"brand=apple": "brand", "brand=samsung": "brand"},
                                     "attribute_to_group": {"brand": "identity"}}
        edges, offsets, _, schema = typed_graph(tiny_data(), features)
        self.assertEqual(schema["nodes"]["spec_attribute"], 1)
        self.assertEqual(schema["nodes"]["spec_group"], 1)
        self.assertEqual(schema["undirected_edges"]["spec_attribute"], 2)
        self.assertEqual(schema["undirected_edges"]["item_attribute"], 3)
        self.assertIn([offsets["spec_attribute"], offsets["spec_group"]], edges.tolist())
        model = make_model("graph_multimodal_gcl", tiny_data(), features, dim=4)
        loss = model.loss(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 2]))
        loss["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        features["spec_hierarchy"]["value_to_attribute"]["invented=review"] = "brand"
        with self.assertRaises(ValueError):
            typed_graph(tiny_data(), features)

    def test_finalization_marks_genuine_missing_images_and_rejects_incomplete_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("text", "clip", "images"):
                (root / name).mkdir()
            products = [{"parent_asin": "a", "title": "Phone", "details": {"Brand": "A"}}]
            data = SimpleNamespace(n_items=1, item_ids=["a"])
            contract = {"encoders": {"text": {"model": "fixture"}, "clip": {"model": "fixture"}}}
            atomic_json(root / "images/a.json", {"item_id": "a", "status": "failed", "errors": ["no image"]})
            with self.assertRaisesRegex(ValueError, "Text phase"):
                finalize(products, data, root, root / "features", contract)
            save_vector(root / "text", "a", text_hash(products[0]), np.ones(384))
            finalize(products, data, root, root / "features", contract)
            with np.load(root / "features/features.npz", allow_pickle=False) as saved:
                self.assertFalse(bool(saved["image_mask"][0]))
                self.assertEqual(float(saved["image"].sum()), 0.)
            manifest = json.loads((root / "features/manifest.json").read_text())
            self.assertEqual(manifest["items_with_images"], 0)


if __name__ == "__main__":
    unittest.main()
