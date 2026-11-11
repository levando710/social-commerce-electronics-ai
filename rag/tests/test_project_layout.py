"""Regression checks for unchanged records and relocated physical assets."""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
project_layout = importlib.import_module("phone_assistant.common.project_layout")
read_jsonl = importlib.import_module("phone_assistant.recommendation.recsys_data").read_jsonl


class LayoutTests(unittest.TestCase):
    def test_historical_record_keeps_exact_path_but_asset_resolves_to_new_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "artifacts/cache/photos/example.img"
            actual.parent.mkdir(parents=True)
            actual.write_bytes(b"asset")
            legacy = "D:/old-checkout/rag/.cache/photos/example.img"
            record = root / "records.jsonl"
            original = json.dumps({"product_image_paths": [legacy]}) + "\n"
            record.write_bytes(original.encode())
            before = record.read_bytes()
            with patch.object(project_layout, "MODULE_ROOT", root):
                row = read_jsonl(record)[0]
                self.assertEqual(row["product_image_paths"], [legacy])
                self.assertEqual(project_layout.asset_path(legacy), actual)
                self.assertEqual(project_layout.asset_path(legacy).read_bytes(), b"asset")
            self.assertEqual(record.read_bytes(), before)

    def test_source_audit_selects_exact_archived_bytes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "src/example.py"
            archived = root / "archive/example.py"
            current.parent.mkdir()
            archived.parent.mkdir()
            current.write_bytes(b"new layout")
            archived.write_bytes(b"recorded source")
            expected = hashlib.sha256(b"recorded source").hexdigest()
            with (
                patch.object(project_layout, "MODULE_ROOT", root),
                patch.object(project_layout, "SNAPSHOT_ROOT", archived.parent),
            ):
                self.assertEqual(project_layout.recorded_source("example.py", expected), archived)
                archived.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "No exact recorded source"):
                    project_layout.recorded_source("example.py", expected)

    def test_external_files_and_urls_are_not_rewritten(self):
        for value in (
            "https://example.com/rag/.cache/file",
            "D:/personal/photos/image.jpg",
            "relative.jpg",
        ):
            self.assertEqual(project_layout.rebind_paths(value), value)


if __name__ == "__main__":
    unittest.main()
