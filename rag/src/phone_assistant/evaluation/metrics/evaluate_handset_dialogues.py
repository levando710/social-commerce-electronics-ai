"""Researcher-authored multi-turn functional cases; not a user study or training set."""
from phone_assistant.common.project_layout import source_path
import json
import argparse
from pathlib import Path

from phone_assistant.conversation.co_guiding import CoGuidingEngine
from phone_assistant.conversation.handset_dialogue import dialogue_options
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_handset_dialogues_v2')
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError("Preserve existing functional outcomes")
    runtime = HandsetRuntime()
    docs = read_jsonl(runtime.corpus)
    engine = CoGuidingEngine(docs, json.loads(runtime.index.read_text(encoding="utf-8")), runtime.recommender(),
        method="legacy", **dialogue_options(runtime, docs))
    dialogues = [
        [("Samsung phone 128GB để chơi game", {"brands": ["samsung"], "storage_gb": 128}, [], ["gaming"]),
         ("còn 256GB thì sao", {"brands": ["samsung"], "storage_gb": 256}, [], ["gaming"]),
         ("8GB RAM nữa", {"brands": ["samsung"], "storage_gb": 256, "ram_gb": 8}, [], ["gaming"]),
         ("không chọn Samsung", {"brands": [], "avoid_brands": ["samsung"], "storage_gb": 256}, ["manufacturer_names"], ["gaming"]),
         ("bắt đầu lại Nokia", {"brands": ["nokia"]}, ["storage_gb", "ram_gb"], [])],
        [("iPhone 13 128GB", {"brands": ["apple"], "model_terms": "iphone 13", "storage_gb": 128}, [], []),
         ("còn 256GB", {"brands": ["apple"], "model_terms": "iphone 13", "storage_gb": 256}, [], []),
         ("Samsung phone", {"brands": ["samsung"], "storage_gb": 256}, ["model_terms"], [])],
        [("Samsung phone 128GB unlocked", {"network": "unlocked", "storage_gb": 128}, [], []),
         ("không phải Samsung", {"brands": [], "avoid_brands": ["samsung"], "network": "unlocked"}, ["manufacturer_names"], []),
         ("chọn Samsung", {"brands": ["samsung"], "avoid_brands": []}, [], [])],
        [("điện thoại pin ít nhất 4000 mAh", {"battery_min_mah": 4000}, [], []),
         ("pin 5000 mAh", {"battery_mah": 5000}, ["battery_min_mah", "battery_max_mah"], []),
         ("new search phone", {"product_type": "phone"}, ["battery_mah"], [])],
        [("điện thoại Samsung dưới 5 triệu", {"max_price": 5000000}, [], []),
         ("new search Samsung phone", {"brands": ["samsung"]}, ["max_price"], [])],
        [("DOOGEE phone", {"manufacturer_names": ["doogee"]}, [], []),
         ("Samsung phone", {"brands": ["samsung"]}, ["manufacturer_names"], [])],
    ]
    rows = []
    for dialogue_id, turns in enumerate(dialogues):
        previous = None
        for turn_index, (query, expected, absent, preferences) in enumerate(turns):
            result = engine.run(query, previous_plan=previous, mode="coupled", top_k=5)
            plan = result["query_plan"]
            errors = [f"state:{key}" for key, value in expected.items() if plan["constraints"].get(key) != value]
            errors += [f"unexpected:{key}" for key in absent if key in plan["constraints"]]
            if plan["preferences"] != preferences:
                errors.append("soft_preferences")
            if any(not r["constraint_check"]["eligible"] for r in result["results"]):
                errors.append("ineligible_result")
            if "max_price" in expected and not result["abstention"]:
                errors.append("unverified_price_not_abstained")
            rows.append({"dialogue_id": dialogue_id, "turn": turn_index, "query": query, "expected": expected,
                "expected_absent": absent, "expected_preferences": preferences, "plan": plan,
                "result_ids": [r["doc_id"] for r in result["results"]], "abstention": result["abstention"],
                "errors": errors, "passed": not errors})
            previous = plan
    # Resolve references against an actual result list, retaining the original identity.
    first = engine.run("Samsung phone", top_k=5)
    for query, count in (("máy đầu tiên", 1), ("so sánh hai máy", 2)):
        result = engine.run(query, previous_plan=first["query_plan"], top_k=5)
        expected_ids = first["query_plan"]["last_result_ids"][:count]
        actual = result["query_plan"]["focus_doc_ids"]
        passed = actual == expected_ids and bool(expected_ids)
        rows.append({"query": query, "expected_focus_ids": expected_ids, "actual_focus_ids": actual, "passed": passed,
            "errors": [] if passed else ["reference_resolution"]})
    output.mkdir(parents=True)
    record = {"scope": "20 researcher-authored state/guard/reference checks, no generator, not human dialogs or joint training data.",
        "runtime_manifest_sha256": sha256(runtime.path / "manifest.json"),
        "source_sha256": {name: sha256(source_path(name)) for name in ("evaluate_handset_dialogues.py", "handset_dialogue.py", "conversation_state.py", "co_guiding.py")},
        "cases": rows, "total": len(rows), "passed": sum(r["passed"] for r in rows), "llm_calls": 0}
    (output / "functional_cases.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "source").mkdir()
    for name in record["source_sha256"]:
        (output / "source" / name).write_bytes((source_path(name)).read_bytes())
    print(json.dumps({"total": len(rows), "passed": record["passed"], "failures": [r for r in rows if not r["passed"]]}, indent=2))


if __name__ == "__main__":
    main()
