"""Three preselected development queries, at most 6 API calls, no retries or test tuning."""
from phone_assistant.common.project_layout import source_path
import csv
import argparse
import json
from pathlib import Path

from phone_assistant.evaluation.metrics.answer_evaluation import judge_answer, summarize_claims
from phone_assistant.retrieval.search.dense_search import load_encoder
from phone_assistant.llm.gemini_client import generate_answer, get_secret
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256

QUERY_IDS = ("samsung128_vi_0", "apple128_vi_0", "battery4000_vi_0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_handset_answers_v2')
    parser.add_argument("--contexts", type=Path, default=ROOT / 'artifacts/evaluation/results_handset_rag_dev_v2/cases.jsonl')
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError("Keep saved answers; do not regenerate to cherry-pick a result")
    source = args.contexts
    cases = {r["query_id"]: r for r in read_jsonl(source)}
    output.mkdir(parents=True)
    plan = {"query_ids": QUERY_IDS, "max_api_calls": 6, "generator_retries": 0, "judge_retries": 0,
        "contexts_source_sha256": sha256(source), "source_sha256": sha256(Path(__file__)),
        "dependency_sha256": {name: sha256(source_path(name)) for name in ("gemini_client.py", "answer_evaluation.py", "handset_dialogue.py")},
        "scope": "Small same-provider generation/judge development diagnostic; not human labels, official Ragas or test accuracy."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (output / "source").mkdir()
    for name in (Path(__file__).name, *plan["dependency_sha256"]):
        (output / "source" / name).write_bytes((source_path(name)).read_bytes())
    available = bool(get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key", "gemini_api_key"))
    encoder = None
    records, attempted = [], 0
    for query_id in QUERY_IDS:
        case = cases[query_id]
        row = {"query_id": query_id, "query": case["query"], "contexts": case["contexts"]}
        if not available:
            row.update(answer=None, judgment={"status": "blocked_missing_api_key"})
        elif not case["contexts"]:
            row.update(answer=None, judgment={"status": "abstained_no_verified_evidence"})
        else:
            attempted += 1
            result = generate_answer(case["query"], case["contexts"], include_images=True, timeout=30, retries=0)
            row.update(answer=result.get("answer"), raw_answer=result.get("raw_answer"), citation_audit=result.get("citation_audit"),
                       generator_model=result.get("model"), generator_usage=result.get("usage"), generation_error=result.get("error"))
            # Save the original generation before requesting a judgment.
            (output / f"{query_id}_generation.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            if result.get("answer"):
                if encoder is None:
                    import torch
                    torch.set_num_threads(2)
                    config = json.loads((ROOT / 'artifacts/cache/handset_features_v1/manifest.json').read_text(encoding="utf-8"))["text_encoder"]
                    encoder = load_encoder(config["model"], revision=config["revision"])
                attempted += 1
                row["judgment"] = judge_answer(case["query"], result["answer"], case["contexts"], encoder)
            else:
                row["judgment"] = {"status": "generation_failed", "error": result.get("error")}
        records.append(row)
        with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(query_id, row["judgment"]["status"], flush=True)
    judged = [r for r in records if r["judgment"]["status"] == "judged"]
    claims = [c for r in judged for c in r["judgment"]["claims"]]
    summary = {**plan, "api_calls_attempted": attempted, "requested_answers": len(QUERY_IDS),
        "generated": sum(bool(r.get("answer")) for r in records), "judged": len(judged),
        "statuses": {r["query_id"]: r["judgment"]["status"] for r in records},
        "claim_metrics": summarize_claims(claims),
        "records_sha256": sha256(output / "records.jsonl"),
        "limitations": ["Three development examples only; same-provider judge may be biased.",
            "Exact quote validation does not replace human semantic checking.",
            "Errors/unknown claims are retained; no zero-hallucination claim if generation/judgment is missing.",
            "No multi-turn generation, user satisfaction or robustness rate is measured."]}
    for metric in ("answer_relevance_reverse_question_cosine", "judge_coverage"):
        values = [r["judgment"]["metrics"][metric] for r in judged if r["judgment"]["metrics"][metric] is not None]
        summary[metric] = float(sum(values) / len(values)) if values else None
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "manual_claim_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "claim", "doc_id", "quote", "auto_supported", "manual_supported", "reviewer", "notes"])
        for row in judged:
            for claim in row["judgment"]["claims"]:
                writer.writerow([row["query_id"], claim["text"], claim.get("doc_id"), claim.get("quote"), claim.get("supported"), "", "", ""])
    (output / "report.md").write_text("# Handset answer diagnostic\n\n" + plan["scope"] + "\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"generated": summary["generated"], "judged": len(judged), "api_calls_attempted": attempted, "claim_metrics": summary["claim_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
