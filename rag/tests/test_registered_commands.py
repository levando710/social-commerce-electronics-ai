"""Every module registered in config/python_modules.json must exist and parse, and no module may be left unregistered."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rag" / "src"


class RegisteredCommandsTest(unittest.TestCase):
    def test_every_registered_module_exists_and_parses(self):
        registry = json.loads((ROOT / "config" / "python_modules.json").read_text(encoding="utf-8"))
        self.assertGreater(len(registry["modules"]), 100)
        problems = []
        for command, dotted in sorted(registry["modules"].items()):
            path = SOURCE / (dotted.replace(".", "/") + ".py")
            if not path.is_file():
                problems.append(f"{command}: missing file {path.relative_to(ROOT)}")
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as error:
                problems.append(f"{command}: {error}")
        self.assertEqual(problems, [])

    def test_no_python_file_is_left_unregistered(self):
        registry = json.loads((ROOT / "config" / "python_modules.json").read_text(encoding="utf-8"))
        registered = {dotted for dotted in registry["modules"].values()}
        unregistered = []
        for path in (SOURCE / "phone_assistant").rglob("*.py"):
            if path.name in ("__init__.py", "__main__.py"):
                continue
            dotted = ".".join(path.relative_to(SOURCE).with_suffix("").parts)
            if dotted not in registered:
                unregistered.append(dotted)
        self.assertEqual(unregistered, [])


if __name__ == "__main__":
    unittest.main()
