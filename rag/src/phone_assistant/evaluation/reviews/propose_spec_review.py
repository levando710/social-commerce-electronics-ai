"""Read-only cross-catalog RAM review suggestions; never corrections or human labels."""
from collections import Counter, defaultdict
import csv
import json

from phone_assistant.conversation.handset_dialogue import HandsetFactIndex
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.retrieval.queries.product_constraints import model_names
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    output = ROOT / 'data/discovery/spec_review_proposals_v1'
    if output.exists():
        raise FileExistsError("Keep existing proposals")
    runtime = HandsetRuntime()
    facts = HandsetFactIndex(runtime.split)
    docs = read_jsonl(runtime.corpus)
    groups = defaultdict(list)
    by_id = {}
    for doc in docs:
        record = {"doc": doc, "facts": facts(doc)}
        by_id[doc["doc_id"]] = record
        models = model_names(doc["title"])
        if len(models) == 1:
            groups[models[0]].append(record)
    proposals = []
    for item, record in by_id.items():
        if "ram_capacity_requires_review" not in record["facts"]["fact_warnings"]:
            continue
        models = model_names(record["doc"]["title"])
        family = models[0] if len(models) == 1 else None
        candidates = [r for r in groups.get(family, []) if r["doc"]["doc_id"] != item and r["facts"].get("ram_gb") is not None
                      and r["facts"].get("fact_evidence", {}).get("ram_gb")]
        counts = Counter(r["facts"]["ram_gb"] for r in candidates)
        proposals.append({"item_id": item, "title": record["doc"]["title"], "parsed_family": family,
            "raw_evidence": record["facts"]["fact_evidence"].get("ram_gb"),
            "candidate_counts": dict(sorted(counts.items())),
            "candidate_source_items": [{"item_id": r["doc"]["doc_id"], "ram_gb": r["facts"]["ram_gb"],
                "quote": r["facts"]["fact_evidence"]["ram_gb"]} for r in candidates],
            "status": "needs_variant_and_independent_source_review" if counts else "no_local_comparison_evidence"})
    output.mkdir(parents=True)
    (output / "proposals.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in proposals), encoding="utf-8")
    with (output / "review_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "title", "parsed_family", "raw_evidence", "candidate_counts_from_other_catalog_items", "candidate_source_items", "status"])
        for p in proposals:
            writer.writerow([p["item_id"], p["title"], p["parsed_family"], json.dumps(p["raw_evidence"]),
                json.dumps(p["candidate_counts"]), "; ".join(r["item_id"] for r in p["candidate_source_items"]), p["status"]])
    summary = {"source_sha256": sha256(ROOT / 'src/phone_assistant/evaluation/reviews/propose_spec_review.py'),
        "split_manifest_sha256": sha256(runtime.split / "manifest.json"), "suspect_items": len(proposals),
        "with_local_candidates": sum(bool(p["candidate_counts"]) for p in proposals),
        "conflicting_candidate_values": sum(len(p["candidate_counts"]) > 1 for p in proposals),
        "automatic_corrections_applied": 0,
        "limitations": ["Parsed model family is incomplete and may merge variants with different RAM; candidate values may also be wrong.",
            "Agreement across Amazon records is not independent manufacturer verification. Do not majority-vote these suggestions into truth.",
            "No raw catalog, features, reviewer labels or app facts were modified."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
