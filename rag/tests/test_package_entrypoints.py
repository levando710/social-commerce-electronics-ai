"""Exercise package imports and the public command boundary after source moves."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
layout = importlib.import_module("phone_assistant.common.project_layout")


class PackageEntrypointTests(unittest.TestCase):
    def test_registered_modules_and_portable_source_closure_exist(self):
        manifest = json.loads((ROOT.parent / "config/python_modules.json").read_text())
        for name in manifest["modules"]:
            with self.subTest(name=name):
                self.assertTrue(layout.source_path(name).is_file())
        bundle = layout.package_sources()
        for name in (
            "config/layout.json",
            "config/python_modules.json",
            "rag/src/run.py",
            "rag/src/phone_assistant/__init__.py",
            "rag/src/phone_assistant/__main__.py",
            "rag/scripts/run_handset_offline.ps1",
        ):
            self.assertIn(name, bundle)
        self.assertTrue(all(path.is_file() for path in bundle.values()))

    def test_command_help_is_forwarded_to_application(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "src/run.py"), "chat_app", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--interaction-db", result.stdout)
        self.assertIn("--handsets", result.stdout)

    def test_package_module_entrypoint_and_unknown_command(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "phone_assistant", "--list"],
            cwd=ROOT / "src",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("apps.chat_app", result.stdout)
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "src/run.py"), "missing_command"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown command", result.stderr)


if __name__ == "__main__":
    unittest.main()
