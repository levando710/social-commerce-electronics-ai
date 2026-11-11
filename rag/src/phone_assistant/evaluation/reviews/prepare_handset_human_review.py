"""Prepare blank, outcome-independent human review forms; do not invent labels."""
import csv
import json

from phone_assistant.llm.gemini_client import normalize_answer_citations
from phone_assistant.conversation.handset_dialogue import HandsetFactIndex
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    output = ROOT / 'artifacts/evaluation/handset_human_review_v1'
    if output.exists():
        raise FileExistsError("Preserve human edits")
    output.mkdir(parents=True)
    runtime = HandsetRuntime()
    docs = {d["doc_id"]: d for d in read_jsonl(runtime.corpus)}
    experiment_path = ROOT / 'artifacts/evaluation/results_handset_rag_test_v1/experiment.json'
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    pairs = {}
    for method in ("bm25", "rewrite_guard"):
        for row in experiment["rows"][method]:
            if row["kind"] != "retrieval" or not row["id"].endswith("_0"):
                continue
            for item in row["ids"][:5]:
                pairs[(row["id"], item)] = row["query"]
    with (output / "retrieval_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "query", "item_id", "title", "details", "manual_relevance", "reviewer", "notes"])
        for (query_id, item), query in sorted(pairs.items()):
            writer.writerow([query_id, query, item, docs[item]["title"], json.dumps(docs[item]["details"], ensure_ascii=False), "", "", ""])
    answers_source = ROOT / 'artifacts/evaluation/results_handset_answers_v1/records.jsonl'
    answers = read_jsonl(answers_source)
    with (output / "answer_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "query", "raw_answer", "generation_status", "manual_faithfulness", "manual_relevance", "manual_citation_correctness", "reviewer", "notes"])
        for row in answers:
            writer.writerow([row["query_id"], row["query"], row.get("answer"), row["judgment"]["status"], "", "", "", "", ""])
    facts = HandsetFactIndex(runtime.split)
    observations = []
    for row in answers:
        if not row.get("answer"):
            continue
        normalized, citations = normalize_answer_citations(row["answer"], row["contexts"])
        (output / f"{row['query_id']}_citation_preview.txt").write_text(normalized, encoding="utf-8")
        for context in row["contexts"]:
            current = facts(docs[context["doc_id"]])
            if current["fact_warnings"]:
                observations.append({"query_id": row["query_id"], "item_id": context["doc_id"],
                    "old_ram_gb": context["catalog_facts"].get("ram_gb"), "current_ram_gb": current["ram_gb"],
                    "raw_evidence": current["fact_evidence"].get("ram_gb"), "warnings": current["fact_warnings"],
                    "citation_normalization_preview": citations})
    manifest = {"retrieval_experiment_sha256": sha256(experiment_path), "answers_source_sha256": sha256(answers_source),
        "retrieval_pairs": len(pairs), "answer_rows": len(answers), "human_labels_filled": 0,
        "ai_diagnostic_observations": observations,
        "instructions": ["manual_relevance: 2 relevant, 1 partial, 0 irrelevant, uncertain if insufficient evidence; blanks are unreviewed.",
            "Judge independently; automatic qrels/scores and contributing method names are not shown in the retrieval sheet.",
            "Answer failures are not correct answers. No automatic or AI review is a human label.",
            "RAM warnings are conservative suspected metadata issues, not confirmed corrections. Do not guess replacement values.",
            "Citation previews only normalize indexes to supplied doc IDs; they are not regenerated answers or evidence of factual support."]}
    (output / "instructions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"retrieval_pairs": len(pairs), "answer_rows": len(answers), "source_warning_examples": len(observations)}, indent=2))


if __name__ == "__main__":
    main()
