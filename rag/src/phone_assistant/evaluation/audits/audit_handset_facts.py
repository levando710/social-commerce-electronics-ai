"""Auditable explicit-field normalization and chip-name extraction, no dataset mutation."""
from collections import Counter
import argparse
import json
from pathlib import Path

from phone_assistant.conversation.handset_dialogue import HandsetFactIndex
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.retrieval.queries.product_constraints import connected_facts
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'data/discovery/handset_fact_audit_v2')
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError("Keep the existing audit")
    runtime = HandsetRuntime()
    index = HandsetFactIndex(runtime.split)
    records = []
    before, after = Counter(), Counter()
    for doc in read_jsonl(runtime.corpus):
        old, new = connected_facts(doc), index(doc)
        for key in ("storage_gb", "ram_gb", "battery_mah", "processor_claims"):
            before[key] += old.get(key) is not None if key != "processor_claims" else bool(old.get(key))
            after[key] += new.get(key) is not None if key != "processor_claims" else bool(new.get(key))
        records.append({"item_id": doc["doc_id"], "title": doc["title"],
            "changes": {k: {"before": old.get(k), "after": new.get(k)} for k in ("storage_gb", "ram_gb", "battery_mah", "processor_claims")
                        if (old.get(k) or None) != (new.get(k) or None)},
            "fact_evidence": new["fact_evidence"]})
    output.mkdir(parents=True)
    (output / "facts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = {"items": len(records), "before_nonmissing": dict(before), "after_nonmissing": dict(after),
        "changed_items": sum(bool(r["changes"]) for r in records), "source_sha256": sha256(ROOT / 'src/phone_assistant/conversation/handset_dialogue.py'),
        "policy": "Runtime-only explicit metadata normalization/extraction. Frozen catalog, features and training metrics unchanged. Not human verification or external spec completion."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
