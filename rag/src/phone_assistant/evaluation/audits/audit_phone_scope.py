"""Audit handset candidates and frozen split coverage without changing historical datasets."""
from phone_assistant.common.project_layout import source_path
from collections import Counter
import argparse
import csv
import json
from pathlib import Path

from phone_assistant.data.phone_scope import handset_decision
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'data/discovery/phone_scope_v3')
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError("Keep the audit; use a new version for a changed scope rule")
    output.mkdir(parents=True)
    report = {"scope": "Mobile handsets only, including iPhone, Android and feature phones; accessories/tablets/laptops excluded.",
        "policy": "Conservative automatic title/category screening; candidate list needs review, not ground truth device labels.",
        "source_sha256": {name: sha256(source_path(name)) for name in ("phone_scope.py", "product_constraints.py", "audit_phone_scope.py")},
        "catalogs": {}}
    for name, source in (("connected", ROOT / 'artifacts/cache/connected_runtime_v1/documents.jsonl'),
                         ("legacy", ROOT / 'artifacts/runtime/catalog/documents.jsonl')):
        docs = read_jsonl(source)
        decisions = [(d, *handset_decision(d)) for d in docs]
        allowed = {d["doc_id"] for d, accepted, _ in decisions if accepted}
        report["catalogs"][name] = {"source_sha256": sha256(source), "total": len(docs), "handset_candidates": len(allowed),
            "excluded": len(docs) - len(allowed), "reasons": dict(Counter(reason for _, _, reason in decisions))}
        with (output / f"{name}_catalog_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["doc_id", "title", "categories", "automatic_handset_candidate", "reason", "human_handset_label", "notes"])
            writer.writerows((d["doc_id"], d.get("title"), json.dumps(d.get("categories", []), ensure_ascii=False), accepted, reason, "", "")
                for d, accepted, reason in decisions)
        if name == "connected":
            split = ROOT / 'data/external/benchmarks/connected_phones_v1'
            report["historical_split_manifest_sha256"] = sha256(split / "manifest.json")
            report["phone_only_projection_diagnostic_not_new_split"] = {}
            for part in ("train", "validation", "test"):
                rows = read_jsonl(split / f"{part}.jsonl")
                retained = [r for r in rows if r["item_id"] in allowed]
                report["phone_only_projection_diagnostic_not_new_split"][part] = {"events": len(retained), "users": len({r["user_id"] for r in retained})}
            labels = {r["user_id"]: r for r in read_jsonl(split / "evaluator_labels.jsonl")}
            queries = read_jsonl(split / "agent_queries.jsonl")
            report["historical_queries_with_handset_target"] = {part: {"queries": sum(q["split"] == part and labels[q["user_id"]]["item_id"] in allowed for q in queries),
                "users": len({q["user_id"] for q in queries if q["split"] == part and labels[q["user_id"]]["item_id"] in allowed})}
                for part in ("train", "validation", "test")}
    report["limitations"] = ["The old catalog, user histories, model weights and experiments remain mixed-catalog artifacts.",
        "A handset-only serving mask is not handset-only model retraining or a valid rebuilt evaluation split.",
        "Do not reuse previous 4,435-item metrics as handset-only results or silently remove rejected test cases.",
        "Conservative screening can exclude real handset bundles or unusual names; CSV human labels remain blank."]
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
