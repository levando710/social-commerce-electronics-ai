"""Check pricing package paths and CLI dispatch without collecting or training."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
paths = importlib.import_module("phone_pricing.common.project_paths")


class PricingPackageTests(unittest.TestCase):
    def test_registry_and_runtime_paths_resolve_from_module_root(self):
        registry = json.loads((ROOT.parent / "config/pricing_modules.json").read_text())
        for module in registry["modules"].values():
            with self.subTest(module=module):
                self.assertTrue((ROOT / "src" / (module.replace(".", "/") + ".py")).is_file())
        self.assertEqual(paths.PROJECT_ROOT, ROOT)
        self.assertEqual(paths.CLEAN_PHONE_DATA, ROOT / "data/processed/phone_clean.csv")
        self.assertEqual(paths.MODEL_DIR, ROOT / "artifacts/models")

    def test_dispatcher_forwards_training_help_without_training(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "src/run.py"), "phonetrain", "--help"],
            cwd=ROOT.parent,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--with-quantiles", result.stdout)

    def test_module_entrypoint_lists_commands_and_rejects_unknown_command(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "phone_pricing", "--list"],
            cwd=ROOT / "src",
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data_preprocessing", result.stdout)
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "src/run.py"), "missing_command"],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown command", result.stderr)


if __name__ == "__main__":
    unittest.main()
