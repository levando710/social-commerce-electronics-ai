"""Check repository structure and Python syntax without importing application code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

try:
    from .project_layout import rag_reference
except ImportError:
    from project_layout import rag_reference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = (
    "tools",
    "tests",
    "pricing/src",
    "pricing/apps",
    "pricing/scripts",
    "pricing/tests",
    "rag/src",
    "rag/tests",
)
REQUIRED_FILES = (
    "README.md",
    ".editorconfig",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    ".env.example",
    "requirements/rag.txt",
    "requirements/pricing.txt",
    "requirements/dev.txt",
    "pricing/README.md",
    "pricing/requirements/requirements.txt",
    "rag/README.md",
    "docs/ARCHITECTURE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md",
    "docs/requirements/De_cuong_Mang_xa_hoi_thuong_mai_do_dien_tu.md",
    "tools/Run-Pricing.ps1",
    "config/layout.json",
    "config/python_modules.json",
    "rag/src/run.py",
    "rag/src/phone_assistant/__main__.py",
    "config/pricing_modules.json",
    "pricing/src/run.py",
    "pricing/src/phone_pricing/__main__.py",
    "pricing/src/phone_pricing/apps/app_demo.py",
)
LEGACY_ROOT_PATHS = (
    "model",
    "BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md",
    "De_cuong_Mang_xa_hoi_thuong_mai_do_dien_tu.md",
    "BAN_GIAO_DU_AN.md",
)
CHAPTER_SOURCES = "rag/docs/CHUONG_1_2_RESULT_SOURCES.json"
REQUIREMENT_INCLUDE = re.compile(r"^-r\s+(.+)$")
MODULE_DIRECTORIES = {
    "rag": frozenset(
        ("src", "scripts", "tests", "requirements", "docs", "data", "artifacts", "__pycache__")
    ),
    "pricing": frozenset(
        (
            "src",
            "apps",
            "scripts",
            "tests",
            "requirements",
            "docs",
            "data",
            "artifacts",
            ".streamlit",
            "__pycache__",
        )
    ),
}


def contained_file(root: Path, relative: str) -> Path:
    """Resolve a relative file while forbidding escapes from the repository."""
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    if not candidate.is_file():
        raise FileNotFoundError(relative)
    return candidate


def check(root: Path, *, require_evidence: bool = False) -> dict:
    root = root.resolve()
    for relative in REQUIRED_FILES:
        contained_file(root, relative)
    for relative in LEGACY_ROOT_PATHS:
        if (root / relative).exists():
            raise ValueError(f"Legacy root entry remains after migration: {relative}")
    for component, allowed in MODULE_DIRECTORIES.items():
        unexpected = [
            path.name
            for path in (root / component).iterdir()
            if path.is_dir() and path.name not in allowed
        ]
        if unexpected:
            raise ValueError(f"Unclassified folders in {component}: {unexpected}")
    registry = json.loads(contained_file(root, "config/python_modules.json").read_text())
    modules = registry["modules"]
    for name, module in modules.items():
        if not re.fullmatch(r"phone_assistant(?:\.[a-z_][a-z_0-9]*)+", module):
            raise ValueError(f"Invalid registered module: {module}")
        if name != module.rsplit(".", 1)[-1] + ".py":
            raise ValueError(f"Registered source basename differs: {name}")
        contained_file(root, "rag/src/" + module.replace(".", "/") + ".py")
    flat_sources = sorted(path.name for path in (root / "rag/src").glob("*.py"))
    if flat_sources != ["run.py"]:
        raise ValueError(f"Unclassified flat Python sources: {flat_sources}")
    pricing_registry = json.loads(contained_file(root, "config/pricing_modules.json").read_text())
    pricing_modules = pricing_registry["modules"]
    for name, module in pricing_modules.items():
        if not re.fullmatch(r"phone_pricing(?:\.[a-z_][a-z_0-9]*)+", module):
            raise ValueError(f"Invalid registered pricing module: {module}")
        if name != module.rsplit(".", 1)[-1] + ".py":
            raise ValueError(f"Registered pricing basename differs: {name}")
        contained_file(root, "pricing/src/" + module.replace(".", "/") + ".py")
    for command in pricing_registry["commands"]:
        if command + ".py" not in pricing_modules:
            raise ValueError(f"Missing registered pricing command: {command}")
    pricing_flat_sources = sorted(path.name for path in (root / "pricing/src").glob("*.py"))
    if pricing_flat_sources != ["run.py"]:
        raise ValueError(f"Unclassified flat pricing sources: {pricing_flat_sources}")
    source_count = 0
    for relative in SOURCE_DIRECTORIES:
        directory = root / relative
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            contained_file(root, path.relative_to(root).as_posix())
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            if relative == "rag/src":
                for node in ast.walk(tree):
                    targets = []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        targets = [node.module]
                    elif isinstance(node, ast.Import):
                        targets = [alias.name for alias in node.names]
                    for target in targets:
                        if target + ".py" in modules:
                            raise ValueError(f"Legacy flat import in {path}: {target}")
                        if target.startswith("phone_assistant."):
                            imported = root / "rag/src" / target.replace(".", "/")
                            if (
                                not imported.with_suffix(".py").is_file()
                                and not (imported / "__init__.py").is_file()
                            ):
                                raise ValueError(f"Missing package import in {path}: {target}")
            if relative == "pricing/src":
                for node in ast.walk(tree):
                    targets = []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        targets = [node.module]
                    elif isinstance(node, ast.Import):
                        targets = [alias.name for alias in node.names]
                    for target in targets:
                        if target.startswith("src.") or target + ".py" in pricing_modules:
                            raise ValueError(f"Legacy pricing import in {path}: {target}")
                        if target.startswith("phone_pricing."):
                            imported = root / "pricing/src" / target.replace(".", "/")
                            if (
                                not imported.with_suffix(".py").is_file()
                                and not (imported / "__init__.py").is_file()
                            ):
                                raise ValueError(f"Missing pricing package import: {target}")
            source_count += 1
    requirement_count = 0
    requirement_paths = list((root / "requirements").glob("*.txt"))
    requirement_paths += list((root / "pricing/requirements").glob("*.txt"))
    requirement_paths += list((root / "rag/requirements").glob("*.txt"))
    for path in sorted(requirement_paths):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = REQUIREMENT_INCLUDE.match(line.strip())
            if match:
                target = (path.parent / match.group(1)).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    raise ValueError(f"Broken requirements include: {path.relative_to(root)}")
        requirement_count += 1
    verified_sources = 0
    unavailable_sources = []
    if (root / CHAPTER_SOURCES).is_file():
        manifest = json.loads((root / CHAPTER_SOURCES).read_text(encoding="utf-8"))
        for relative, expected in manifest.items():
            normalized = rag_reference(root, relative)
            if not (root / normalized).is_file():
                if require_evidence:
                    raise FileNotFoundError(normalized)
                unavailable_sources.append(normalized)
                continue
            path = contained_file(root, normalized)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Research source hash mismatch: {relative}")
            verified_sources += 1
    return {
        "status": "passed",
        "python_sources_parsed": source_count,
        "registered_python_modules_verified": len(modules),
        "registered_pricing_modules_verified": len(pricing_modules),
        "requirements_files_checked": requirement_count,
        "frozen_report_inputs_verified": verified_sources,
        "frozen_report_inputs_unavailable": unavailable_sources,
        "api_calls": 0,
        "training_runs": 0,
        "scope": "Structure, Python syntax, requirements includes and frozen report input hashes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--require-evidence", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.root, require_evidence=args.require_evidence), indent=2))


if __name__ == "__main__":
    main()
