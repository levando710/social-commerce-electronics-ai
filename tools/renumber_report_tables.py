"""Renumber table and figure captions of the main chapters in order of appearance and update every reference.

Captions look like ``**Bảng 2.7b. ...`` or ``**Hình 2.1b. ...``.  Numbers are reassigned per chapter and per
kind as 2.1, 2.2, ... in the order the captions appear; references such as ``Bảng 2.7b`` or ``Hình 2.1b``
anywhere in the text follow the same mapping.  Appendix items (``Bảng C.1``) are left alone.  Default is a dry
run; ``--apply`` rewrites the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md"
KINDS = ("Bảng", "Hình")


def caption_pattern(kind):
    return re.compile(rf"^\*\*{kind} (\d+)\.(\d+[a-z]?)\.", re.M)


def reference_pattern(kind):
    return re.compile(rf"{kind} (\d+)\.(\d+[a-z]?)(?![\w])")


def build_mapping(text, kind="Bảng"):
    """old label -> new label, numbering each chapter's captions sequentially by appearance."""
    counters, mapping = {}, {}
    for match in caption_pattern(kind).finditer(text):
        chapter, old = match.group(1), f"{match.group(1)}.{match.group(2)}"
        if old in mapping:
            raise ValueError(f"Duplicate {kind} caption number {old}")
        counters[chapter] = counters.get(chapter, 0) + 1
        mapping[old] = f"{chapter}.{counters[chapter]}"
    return mapping


def apply_mapping(text, mapping, kind="Bảng"):
    pattern = reference_pattern(kind)
    unknown = sorted({f"{m.group(1)}.{m.group(2)}" for m in pattern.finditer(text)} - set(mapping))
    if unknown:
        raise ValueError(f"{kind} references without a caption: {unknown}")
    return pattern.sub(lambda m: f"{kind} " + mapping[f"{m.group(1)}.{m.group(2)}"], text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    text = args.report.read_text(encoding="utf-8")
    updated = text
    for kind in KINDS:
        mapping = build_mapping(text, kind)
        changes = {old: new for old, new in mapping.items() if old != new}
        for old, new in changes.items():
            print(f"{kind} {old} -> {kind} {new}")
        print(f"{kind}: {len(mapping)} captions, {len(changes)} renumbered")
        updated = apply_mapping(updated, mapping, kind)
    if args.apply:
        args.report.write_text(updated, encoding="utf-8")
        print("applied")


if __name__ == "__main__":
    main()
