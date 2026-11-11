"""Retry only a failed judgment of an unchanged saved answer, within six total calls."""
import csv
import json
from pathlib import Path
import time

from phone_assistant.evaluation.metrics.answer_evaluation import judge_answer
from phone_assistant.retrieval.search.dense_search import load_encoder
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    source = ROOT / 'artifacts/evaluation/results_handset_answers_v1'
    output = ROOT / 'artifacts/evaluation/results_handset_answer_rejudge_v1'
    if output.exists():
        raise FileExistsError("Keep every judgment attempt")
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    budget = min(2, 6 - original["api_calls_attempted"])
    eligible = [r for r in read_jsonl(source / "records.jsonl") if r.get("answer") and r["judgment"]["status"] == "judge_failed"]
    if budget < 1 or not eligible:
        raise ValueError("No remaining call budget or failed judgment")
    case = eligible[0]
    output.mkdir(parents=True)
    plan = {"saved_records_sha256": sha256(source / "records.jsonl"), "query_id": case["query_id"],
        "original_calls": original["api_calls_attempted"], "max_additional_calls": budget,
        "selection": "First failed judgment only; unchanged original answer and contexts; stop at first valid judgment regardless of score.",
        "source_sha256": sha256(Path(__file__)), "judge_source_sha256": sha256(ROOT / 'src/phone_assistant/evaluation/metrics/answer_evaluation.py')}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    import torch
    torch.set_num_threads(2)
    config = json.loads((ROOT / 'artifacts/cache/handset_features_v1/manifest.json').read_text(encoding="utf-8"))["text_encoder"]
    encoder = load_encoder(config["model"], revision=config["revision"])
    attempts = []
    for index in range(budget):
        judgment = judge_answer(case["query"], case["answer"], case["contexts"], encoder)
        attempts.append(judgment)
        with (output / "attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(judgment, ensure_ascii=False) + "\n")
        print(f"Judge attempt {index + 1}: {judgment['status']}", flush=True)
        if judgment["status"] == "judged":
            break
        if index + 1 < budget:
            time.sleep(5)
    final = attempts[-1]
    record = {**plan, "additional_calls": len(attempts), "total_calls": original["api_calls_attempted"] + len(attempts),
        "status": final["status"], "judgment": final,
        "scope": "Diagnostic on one saved pre-fix answer; not validation of the corrected generator or human truth."}
    (output / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (output / "manual_claim_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "claim", "doc_id", "quote", "auto_supported", "manual_supported", "notes"])
        for claim in final.get("claims", []):
            writer.writerow([case["query_id"], claim["text"], claim.get("doc_id"), claim.get("quote"), claim.get("supported"), "", ""])
    print(json.dumps({"total_calls": record["total_calls"], "status": final["status"], "metrics": final.get("metrics")}, indent=2))


if __name__ == "__main__":
    main()
