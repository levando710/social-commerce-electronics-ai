"""Exercise offline catalog answers against stored cases; no LLM quality claims."""
import json
import re

from phone_assistant.conversation.catalog_answer import catalog_answer
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    output = ROOT / 'artifacts/evaluation/results_catalog_answers_v1'
    if output.exists():
        raise FileExistsError("Keep prior offline outputs")
    records, sources = [], {}
    for cohort in ("dev", "test"):
        source = ROOT / 'artifacts/evaluation' / f"results_handset_rag_{cohort}_v2/cases.jsonl"
        sources[cohort] = sha256(source)
        for case in read_jsonl(source):
            answer = catalog_answer(case["contexts"], case["query_plan"])
            ids = set(re.findall(r"\[([A-Z0-9]{10})\]", answer))
            allowed = {r["doc_id"] for r in case["contexts"][:3]}
            if ids != allowed:
                raise ValueError("Offline summary lost or invented a product citation")
            records.append({"query_id": case["query_id"], "cohort": cohort, "query": case["query"],
                "answer": answer, "answer_mode": "catalog_template", "llm_generated": False,
                "source_ids": sorted(ids), "abstention": not case["contexts"]})
    output.mkdir(parents=True)
    (output / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    record = {"cases": len(records), "cohort_source_sha256": sources, "source_sha256": sha256(ROOT / 'src/phone_assistant/conversation/catalog_answer.py'),
        "api_calls": 0, "llm_generated_answers": 0, "citation_identity_checks_passed": len(records),
        "abstentions": sum(r["abstention"] for r in records),
        "scope": "Template output and citation-identity checks only; not an LLM faithfulness benchmark, human evaluation or replacement for the blocked generation study."}
    (output / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "catalog_answer_source.py").write_bytes((ROOT / 'src/phone_assistant/conversation/catalog_answer.py').read_bytes())
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
