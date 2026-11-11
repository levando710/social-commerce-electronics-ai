"""Resolve stored paths without rewriting frozen experiment data or source snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

MODULE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = MODULE_ROOT.parent
LAYOUT = json.loads((PROJECT_ROOT / "config/layout.json").read_text(encoding="utf-8"))["rag"]
SNAPSHOT_ROOT = MODULE_ROOT / "artifacts/provenance/pre_layout/src"
PYTHON_MODULES = json.loads(
    (PROJECT_ROOT / "config/python_modules.json").read_text(encoding="utf-8")
)["modules"]


def source_path(name: str) -> Path:
    """Resolve a recorded source basename to its current functional package."""
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Source reference must be a filename")
    if name.endswith((".ps1", ".vbs")):
        return MODULE_ROOT / "scripts" / name
    module = PYTHON_MODULES.get(name)
    if module is None:
        return MODULE_ROOT / "src" / name
    return MODULE_ROOT / "src" / (module.replace(".", "/") + ".py")


def package_sources() -> dict[str, Path]:
    """Portable source closure, including launchers and layout configuration."""
    package = MODULE_ROOT / "src/phone_assistant"
    paths = list(package.rglob("*.py"))
    paths += [MODULE_ROOT / "src/run.py"]
    paths += list((MODULE_ROOT / "scripts").glob("*.ps1"))
    paths += list((MODULE_ROOT / "scripts").glob("*.vbs"))
    paths += [PROJECT_ROOT / "config/layout.json", PROJECT_ROOT / "config/python_modules.json"]
    return {path.relative_to(PROJECT_ROOT).as_posix(): path for path in paths}


def relative_path(relative: str) -> Path:
    """Resolve a documented pre-migration path against the current RAG layout."""
    normalized = relative.replace("\\", "/")
    head, separator, tail = normalized.partition("/")
    if head in ("src", "scripts") and tail in PYTHON_MODULES:
        return source_path(tail)
    return MODULE_ROOT / (LAYOUT.get(head, head) + (separator + tail if separator else ""))


def rebind_paths(value):
    """Rebase absolute project asset paths at read time; leave stored bytes unchanged."""
    if isinstance(value, list):
        return [rebind_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: rebind_paths(item) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    if not (normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized)):
        return value
    marker = "/rag/"
    position = normalized.lower().rfind(marker)
    if position < 0:
        return value
    tail = normalized[position + len(marker) :]
    head = tail.partition("/")[0]
    if head not in LAYOUT:
        return value
    return str(relative_path(tail))


def recorded_source(name: str, expected: str) -> Path:
    """Find the exact recorded source, never claim migrated code has the old hash."""
    if Path(name).name != name:
        raise ValueError("Recorded source must be a filename")
    candidates = [source_path(name), SNAPSHOT_ROOT / name]
    for path in candidates:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path
    for path in (MODULE_ROOT / "artifacts/evaluation").glob(f"*/source/{name}"):
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return path
    raise ValueError(f"No exact recorded source found for {name}")


def asset_path(value) -> Path:
    """Locate an asset referenced by an unchanged historical catalog record."""
    return Path(rebind_paths(str(value)))
