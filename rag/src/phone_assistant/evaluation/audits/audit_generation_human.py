"""Validate actual human edits, keeping blank/partial reviews separate from quality scores."""
import argparse
import csv
import json
import math
from pathlib import Path

from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256
from phone_assistant.training.train_recsys_portable import atomic_json

SOURCE = ROOT / 'artifacts/evaluation/results_rag_generation_pilot_v1'
OUTPUT = ROOT / 'artifacts/evaluation/results_rag_generation_human_v1'
FIELDS = ("human_faithfulness", "human_relevance_0_1_2", "human_visual_support")
RELEVANCE_LEVELS = (0, 1, 2)
VISUAL_LABELS = ("supported", "unsupported", "uncertain", "not_applicable")
MINIMUM_FAITHFULNESS = 0.0
MAXIMUM_FAITHFULNESS = 1.0


def validate_review(row, case):
    values = {name: str(row.get(name) or "").strip() for name in (*FIELDS, "reviewer", "notes")}
    if not any(values[name] for name in FIELDS):
        return {"case_id": row["case_id"], "status": "unreviewed"}
    if not case.get("answer"):
        raise ValueError("Failed/missing generation cannot receive a quality score")
    if not values["reviewer"]:
        raise ValueError("A human review needs a reviewer name/code")
    faithfulness = None
    if values["human_faithfulness"]:
        try:
            faithfulness = float(values["human_faithfulness"])
        except ValueError as exc:
            raise ValueError("Human faithfulness must be a finite fraction from 0 to 1") from exc
        if not math.isfinite(faithfulness) or not MINIMUM_FAITHFULNESS <= faithfulness <= MAXIMUM_FAITHFULNESS:
            raise ValueError("Human faithfulness must be a finite fraction from 0 to 1")
        if faithfulness < MAXIMUM_FAITHFULNESS and not values["notes"]:
            raise ValueError("Explain unsupported claims/evidence in notes when faithfulness is below 1")
    relevance = None
    if values["human_relevance_0_1_2"]:
        try:
            relevance = int(values["human_relevance_0_1_2"])
        except ValueError as exc:
            raise ValueError("Human relevance must be 0, 1 or 2") from exc
        if relevance not in RELEVANCE_LEVELS:
            raise ValueError("Human relevance must be 0, 1 or 2")
    visual = values["human_visual_support"] or None
    if visual is not None and visual not in VISUAL_LABELS:
        raise ValueError("Unknown visual support label")
    return {"case_id": row["case_id"], "status": "complete" if all(values[name] for name in FIELDS) else "partial",
        "faithfulness": faithfulness, "relevance": relevance, "visual_support": visual,
        "reviewer": values["reviewer"], "notes": values["notes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cases = {row["case_id"]: row for row in read_jsonl(SOURCE / "records.jsonl")}
    sheet = SOURCE / "manual_answer_labels.csv"
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(cases) or {row["case_id"] for row in rows} != set(cases):
        raise ValueError("Human sheet must contain each fixed case exactly once")
    reviews = []
    for row in rows:
        case = cases[row["case_id"]]
        if row["question"] != case["question"] or row["answer"] != (case["answer"] or ""):
            raise ValueError("Only label/reviewer/notes fields may be changed in the human sheet")
        reviews.append(validate_review(row, case))
    decided = [r for r in reviews if r["status"] == "complete"]
    result = {"sheet_sha256": sha256(sheet), "records_sha256": sha256(SOURCE / "records.jsonl"),
        "source_sha256": sha256(Path(__file__)), "planned": len(cases),
        "generated": sum(bool(case["answer"]) for case in cases.values()),
        "complete": len(decided), "partial": sum(r["status"] == "partial" for r in reviews),
        "unreviewed": sum(r["status"] == "unreviewed" for r in reviews), "reviews": reviews,
        "scope": "Values entered by named reviewers; no automatic substitution for missing human scores."}
    if not decided:
        print(json.dumps({key: result[key] for key in ("planned", "generated", "complete", "partial", "unreviewed")}, indent=2))
        print("No complete human review yet; no quality metric or frozen report was created.")
        return
    if args.output.exists():
        raise FileExistsError("Keep previous human audit; use a new output for later edits")
    result["mean_faithfulness_complete_reviews"] = sum(r["faithfulness"] for r in decided) / len(decided)
    result["mean_relevance_complete_reviews"] = sum(r["relevance"] for r in decided) / len(decided)
    args.output.mkdir(parents=True); atomic_json(args.output / "analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "reviews"}, indent=2))


if __name__ == "__main__":
    main()
