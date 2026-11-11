"""Check that decimal numbers in the report's result tables exist in the frozen result files.

This confirms a value is present in the cited sources (including negative bounds, percentages and
millisecond conversions).  It does not verify that a number sits in the correct row or column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md"
SOURCES = ROOT / "rag/docs/CHUONG_1_2_RESULT_SOURCES.json"
SECTION_START, SECTION_END = "### 2.6.3.", "## Thảo luận kết quả"
EXTRA_GLOBS = (
    "rag/artifacts/evaluation/results_electronics_latency_v1/*.json",
    "rag/artifacts/evaluation/results_rag_generation_expansion_openai_analysis_v1/summary.json",
    "rag/artifacts/evaluation/results_amazon2023_baby_v1/experiment.json",
    "rag/artifacts/evaluation/results_spec_extraction_test_v1/summary.json",
    "rag/artifacts/evaluation/results_answer_relevance_openai_v1/summary.json",
    "rag/artifacts/evaluation/results_mind_small_text_v1/experiment.json",
    "rag/artifacts/evaluation/results_mind_small_text_analysis_v1/analysis.json",
)
TOLERANCE = 6e-7
DECIMAL = re.compile(r"(?<![\d.])\d+\.\d{3,}")
SCALES = (1, 100, 1000)
DIGITS = (6, 5, 4, 3)


def candidate_paths(name):
    relative = name.replace("\\", "/")
    yield ROOT / "rag/artifacts" / relative
    yield ROOT / "rag/data" / relative
    yield ROOT / "rag/data" / relative.replace("external_datasets", "external", 1)


def collect_values(node, values):
    if isinstance(node, dict):
        for child in node.values():
            collect_values(child, values)
    elif isinstance(node, list):
        for child in node:
            collect_values(child, values)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        for scale in SCALES:
            for digits in DIGITS:
                values.add(round(abs(float(node)) * scale, digits))


def load_values(sources=SOURCES):
    values, missing = set(), []
    paths = []
    for name in json.loads(Path(sources).read_text(encoding="utf-8")):
        found = next((path for path in candidate_paths(name) if path.is_file()), None)
        (paths.append(found) if found else missing.append(name))
    for pattern in EXTRA_GLOBS:
        paths.extend(ROOT.glob(pattern))
    for path in paths:
        collect_values(json.loads(path.read_text(encoding="utf-8")), values)
    return values, missing


def table_numbers(text, start=SECTION_START, end=SECTION_END):
    section = text[text.index(start) : text.index(end)]
    return [
        (line[:46], float(match))
        for line in section.splitlines()
        if line.startswith("|")
        for cell in line.strip("|").split("|")
        for match in DECIMAL.findall(cell)
    ]


def unmatched(text, values):
    return [
        (row, number)
        for row, number in table_numbers(text)
        if not any(abs(number - value) < TOLERANCE for value in values)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    args = parser.parse_args()
    values, missing = load_values(args.sources)
    text = args.report.read_text(encoding="utf-8")
    checked = len(table_numbers(text))
    bad = unmatched(text, values)
    print(
        json.dumps(
            {
                "numbers_checked": checked,
                "not_found": len(bad),
                "source_files_missing": missing,
                "scope": "presence of each value in the cited sources, not correct row or column",
            },
            indent=2,
        )
    )
    for row, number in bad:
        print(f"NOT FOUND {number}  in  {row}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
