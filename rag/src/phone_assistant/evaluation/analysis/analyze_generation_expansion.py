"""Summarize the generation-expansion run as designed-case diagnostics, not human truth or user-study evidence."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from phone_assistant.common.project_layout import asset_path
from phone_assistant.evaluation.analysis.analyze_generation_pilot import aggregate
from phone_assistant.evaluation.metrics.run_generation_expansion import OUTPUT as RUN_OUTPUT, usage_cost_usd
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256

ANALYSIS_OUTPUT = ROOT / "artifacts/evaluation/results_rag_generation_expansion_analysis_v1"
ARMS = ("text_only", "text_image")
CITATION = re.compile(r"\[([A-Za-z0-9_]{6,})\]")
LIMITATIONS = (
    "Researcher-designed prompts over already-inspected phone contexts; not independent users or a new test set.",
    "Same-provider judge: quote/ID checks validate references only, not semantic truth; a 'supported' vote can still be wrong.",
    "Answers without any definite claim have undefined faithfulness and are excluded from macro averages, never counted as 1.",
    "Group-level descriptive differences only; no confidence interval is claimed for user-level effects.",
    "Abstention groups have empty context, so their image arm is a negative control and not evidence about images.",
    "Not a verified hallucination rate; human labels remain blank until a person fills them.",
)


def citation_audit(rows):
    cited = unknown = answered = uncited = 0
    for row in rows:
        if not row["answer"]:
            continue
        answered += 1
        ids = CITATION.findall(row["answer"])
        known = {context["doc_id"] for context in row["contexts"]}
        cited += bool(ids)
        uncited += not ids and bool(known)
        unknown += sum(identifier not in known for identifier in ids)
    return {"answered": answered, "with_any_citation": cited, "no_citation_despite_context": uncited,
            "citations_to_ids_outside_context": unknown,
            "unresolved_numeric_references": sum(len(row["citation_audit"]["unresolved_numeric"]) for row in rows if row["answer"])}


def group_macro(rows):
    """Mean answer faithfulness per source group, then across groups, so repeated arms/questions do not dominate."""
    per_group = defaultdict(list)
    for row in rows:
        metrics = row["judgment"].get("metrics") if row["judgment"]["status"] == "judged" else None
        if metrics and metrics["faithfulness"] is not None:
            per_group[row["cohort"], row["intent_id"]].append(metrics["faithfulness"])
    means = [float(np.mean(values)) for values in per_group.values()]
    return {"groups_with_faithfulness": len(means), "macro_over_groups": float(np.mean(means)) if means else None}


def paired_differences(rows):
    """Text+image minus text-only on matched questions where both answers have defined faithfulness."""
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]][row["arm"]] = row
    result = defaultdict(lambda: {"pairs": 0, "image_higher": 0, "image_lower": 0, "equal": 0, "mean_difference": None, "_d": []})
    for arms in pairs.values():
        if set(arms) != set(ARMS):
            continue
        values = []
        for arm in ARMS:
            judgment = arms[arm]["judgment"]
            values.append(judgment["metrics"]["faithfulness"] if judgment["status"] == "judged" else None)
        if None in values or arms["text_only"]["source_kind"] != "retrieval":
            continue
        entry = result[arms["text_only"]["question_type"]]
        difference = values[1] - values[0]
        entry["pairs"] += 1
        entry["image_higher"] += difference > 0
        entry["image_lower"] += difference < 0
        entry["equal"] += difference == 0
        entry["_d"].append(difference)
    for entry in result.values():
        entry["mean_difference"] = float(np.mean(entry.pop("_d")))
    return dict(result)


def usage_report(run, plan):
    latencies, tokens, cost, missing_usage = [], {"prompt": 0, "output_incl_thinking": 0}, 0.0, 0
    for event in read_jsonl(run / "requests.jsonl"):
        if event["event"] != "finished":
            continue
        latencies.append(event["latency_seconds"])
        usage = event.get("usage")
        if not usage:
            missing_usage += 1
            continue
        tokens["prompt"] += int(usage.get("promptTokenCount") or 0)
        tokens["output_incl_thinking"] += int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
        cost += usage_cost_usd(usage, plan["pricing"])
    return {"requests_finished": len(latencies), "requests_without_usage": missing_usage, "tokens": tokens,
            "billable_cost_usd_at_planned_prices": round(cost, 6),
            "latency_seconds_median": float(np.median(latencies)) if latencies else None,
            "latency_seconds_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "pricing": plan["pricing"], "scope": "Provider-reported usage; prices are those in the protocol, not an invoice."}


def analyze(run, output):
    if output.exists():
        raise FileExistsError("Preserve the first analysis; use a new output for a revised one")
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if sha256(run / "records.jsonl") != status["records_sha256"]:
        raise ValueError("Saved run records changed after the runner finished")
    records = read_jsonl(run / "records.jsonl")
    for row in records:
        for image in row["images"]:
            if sha256(asset_path(image["path"])) != image["sha256"]:
                raise ValueError("Image evidence changed")
    def by(key):
        return {value: {arm: aggregate([r for r in records if r[key] == value and r["arm"] == arm]) for arm in ARMS}
                for value in sorted({r[key] for r in records})}

    statuses = defaultdict(int)
    for row in records:
        statuses[row["judgment"]["status"]] += 1
    flagged = [{"case_id": row["case_id"], "claim": claim} for row in records if row["judgment"]["status"] == "judged"
               for claim in row["judgment"]["claims"] if claim["supported"] is not True]
    result = {
        "run_status": status, "records_sha256": status["records_sha256"], "source_sha256": sha256(Path(__file__)),
        "coverage": {"planned": plan["planned_answers"], "recorded": len(records),
                     "not_recorded": plan["planned_answers"] - len(records), "judgment_status_counts": dict(statuses)},
        "overall": aggregate(records),
        "by_arm": {arm: aggregate([r for r in records if r["arm"] == arm]) for arm in ARMS},
        "by_question_type": by("question_type"), "by_source_kind": by("source_kind"), "by_cohort": by("cohort"),
        "source_group_macro_faithfulness": {arm: group_macro([r for r in records if r["arm"] == arm]) for arm in ARMS},
        "image_vs_text_only_paired_descriptive": paired_differences(records),
        "citation_audit": {arm: citation_audit([r for r in records if r["arm"] == arm]) for arm in ARMS},
        "unsupported_or_unknown_claims_for_human_review": flagged,
        "usage": usage_report(run, plan), "human_labels_filled": 0, "limitations": list(LIMITATIONS)}
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN_OUTPUT)
    parser.add_argument("--output", type=Path, default=ANALYSIS_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.run, args.output)
    print(json.dumps({key: result[key] for key in ("coverage", "overall", "usage")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
