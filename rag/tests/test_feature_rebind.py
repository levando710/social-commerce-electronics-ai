import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.recommendation.recsys_data import sha256
from phone_assistant.features.rebind_catalog_features import rebind
from phone_assistant.data.preparation.prepare_sparse_electronics import thin_training


class FeatureRebindTests(unittest.TestCase):
    def test_missing_feedback_mask_is_deterministic_train_only_and_keeps_each_user(self):
        rows = [{"user_id": "u", "item_id": str(i), "timestamp": i} for i in range(6)] + [
            {"user_id": "v", "item_id": "9", "timestamp": 9}]
        selected = thin_training(rows, .5, 42)
        self.assertEqual(selected, thin_training(list(reversed(rows)), .5, 42))
        self.assertEqual(len(selected), 4)
        self.assertEqual({r["user_id"] for r in selected}, {"u", "v"})
        self.assertTrue(all(r in rows for r in selected))
    def fixture(self, root):
        source, target, features = (root / name for name in ("source", "target", "features"))
        for p in (source, target, features): p.mkdir()
        for p in (source, target):
            (p / "products.jsonl").write_text('same metadata\n')
            (p / "manifest.json").write_text(p.name)
        (features / "features.npz").write_bytes(b"immutable feature fixture")
        manifest = {"source_metadata_sha256": sha256(source / "products.jsonl"),
            "artifacts_sha256": {"features.npz": sha256(features / "features.npz")}}
        (features / "manifest.json").write_text(json.dumps(manifest))
        return source, target, features, manifest

    def test_identical_catalog_reuse_changes_binding_without_changing_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target, features, manifest = self.fixture(root)
            original = (features / "manifest.json").read_bytes()
            data = SimpleNamespace(item_ids=["a", "b"], n_items=2)
            with patch('phone_assistant.features.rebind_catalog_features.InteractionData', return_value=data), patch('phone_assistant.features.rebind_catalog_features.load_features', return_value=({}, manifest)):
                rebind(source, features, target, root / "output")
            self.assertEqual((features / "manifest.json").read_bytes(), original)
            self.assertEqual((root / "output/features.npz").read_bytes(), (features / "features.npz").read_bytes())
            result = json.loads((root / "output/manifest.json").read_text())
            self.assertEqual(result["split_manifest_sha256"], sha256(target / "manifest.json"))

    def test_different_order_or_metadata_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target, features, manifest = self.fixture(root)
            first, second = SimpleNamespace(item_ids=["a", "b"]), SimpleNamespace(item_ids=["b", "a"])
            with patch('phone_assistant.features.rebind_catalog_features.InteractionData', side_effect=[first, second]), patch('phone_assistant.features.rebind_catalog_features.load_features', return_value=({}, manifest)):
                with self.assertRaisesRegex(ValueError, "Ordered item IDs"):
                    rebind(source, features, target, root / "output")
            self.assertFalse((root / "output").exists())
            (target / "products.jsonl").write_text('changed metadata\n')
            with patch('phone_assistant.features.rebind_catalog_features.InteractionData', return_value=first), patch('phone_assistant.features.rebind_catalog_features.load_features', return_value=({}, manifest)):
                with self.assertRaisesRegex(ValueError, "metadata differs"):
                    rebind(source, features, target, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
