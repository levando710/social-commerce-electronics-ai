"""Guard against automated refactors that replace plain string literals with module paths."""

import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = re.compile(r"^phone_(assistant|pricing)\.[A-Za-z0-9_.]+$")
ALLOWED_CALLS = {"import_module", "patch"}


def call_name(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def stray_module_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and call_name(node) in ALLOWED_CALLS:
            allowed.update(id(child) for child in ast.walk(node))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and MODULE_PATH.match(node.value)
        and id(node) not in allowed
    ]


class SourceHygieneTests(unittest.TestCase):
    def test_no_plain_literal_was_replaced_by_a_module_path(self):
        problems = []
        for folder in ("rag/src", "pricing/src", "rag/tests", "pricing/tests", "tools", "tests"):
            for path in (ROOT / folder).rglob("*.py"):
                problems += [
                    f"{path.relative_to(ROOT)}:{line}: {value}"
                    for line, value in stray_module_literals(path)
                ]
        self.assertEqual(problems, [])

    def test_detector_flags_a_replaced_dictionary_key(self):
        sample = ROOT / "tests" / "_hygiene_sample.py"
        sample.write_text("x = {'phone_assistant.retrieval.search.bm25': 1}\n", encoding="utf-8")
        try:
            self.assertEqual(len(stray_module_literals(sample)), 1)
        finally:
            sample.unlink()


if __name__ == "__main__":
    unittest.main()
