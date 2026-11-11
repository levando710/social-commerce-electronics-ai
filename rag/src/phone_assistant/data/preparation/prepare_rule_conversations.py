"""Rule-generated phone dialogue fixtures conditioned on review targets, NOT observed conversations."""
from collections import Counter, defaultdict
import hashlib
import json

from phone_assistant.conversation.handset_dialogue import capacity_gb
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, read_jsonl, sha256


def examples(data, products):
    histories = defaultdict(list)
    for row in data.rows["train"]:
        histories[row["user_id"]].append(row)
    val = {r["user_id"]: r for r in data.rows["validation"]}
    test = {r["user_id"]: r for r in data.rows["test"]}
    evaluation_users = sorted(set(val) & set(test), key=lambda u: hashlib.sha256(("rule-dialogue-v1:" + u).encode()).hexdigest())
    cohorts = {u: "validation" if i % 2 == 0 else "test" for i, u in enumerate(evaluation_users)}
    queries, labels, profiles, rejected = [], [], [], Counter()
    for uid in sorted(histories):
        rows = sorted(histories[uid], key=lambda r: (r["timestamp"], r["item_id"]))
        if uid in cohorts:
            cohort = cohorts[uid]
            goal = val[uid] if cohort == "validation" else test[uid]
            prior = rows
        else:
            # Never recycle an evaluation user into the proxy training cohort.
            if uid in val or uid in test or len(rows) < 2:
                rejected["no_train_query_history_or_other_eval_user"] += 1
                continue
            cohort, goal, prior = "train", rows[-1], rows[:-1]
        if not prior or max(r["timestamp"] for r in prior) >= goal["timestamp"] or any(r["item_id"] == goal["item_id"] for r in prior):
            rejected["non_novel_or_not_strictly_after_history"] += 1
            continue
        product = products[goal["item_id"]]
        details = product.get("details") or {}
        brand = str(details.get("Brand") or details.get("Manufacturer") or "").strip()
        if brand.lower() in {"unknown", "generic", "none"} or len(brand) > 40:
            brand = ""
        storage = capacity_gb(details.get("Memory Storage Capacity", ""))
        storage = int(storage) if storage in {8, 16, 32, 64, 128, 256, 512, 1024} else None
        if not brand and storage is None:
            rejected["no_supported_explicit_query_attributes"] += 1
            continue
        for language in ("vi", "en"):
            first = (f"Tôi tìm điện thoại {brand}." if language == "vi" else f"I am looking for a {brand} phone.").strip()
            second = (f"Tôi cần bộ nhớ trong {storage}GB." if language == "vi" else f"I need {storage}GB storage.") if storage else (
                "Cho tôi xem các điện thoại phù hợp." if language == "vi" else "Show me matching phones.")
            queries.append({"dialogue_id": f"rule:{uid}:{language}", "user_id": uid, "split": cohort,
                "query": first + " " + second, "language": language,
                "visible_turns": [{"role": "user", "text": first}, {"role": "assistant", "text": "Bạn cần dung lượng nào?" if language == "vi" else "What storage do you need?"}, {"role": "user", "text": second}],
                "provenance": "rule-generated from target catalog attributes; query is concatenated visible user turns only"})
        labels.append({**goal, "cohort": cohort})
        profiles.append({"user_id": uid, "cohort": cohort, "history": prior, "goal_in_history": False})
    return queries, labels, profiles, dict(rejected)


def main():
    output = ROOT / 'artifacts/evaluation/rule_phone_conversations_v1'
    if output.exists():
        raise FileExistsError("Keep frozen proxy fixtures")
    split = ROOT / 'data/external/benchmarks/handsets_only_v2'
    data = InteractionData(split)
    products = {p["parent_asin"]: p for p in read_jsonl(split / "products.jsonl")}
    queries, labels, profiles, rejected = examples(data, products)
    counts = {s: {"queries": sum(q["split"] == s for q in queries), "users": sum(r["cohort"] == s for r in labels)} for s in ("train", "validation", "test")}
    if any(v["users"] == 0 for v in counts.values()):
        raise ValueError("Need nonempty disjoint cohorts")
    output.mkdir(parents=True)
    for name, values in (("queries", queries), ("labels", labels), ("profiles", profiles)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
    train_texts = {q["query"] for q in queries if q["split"] == "train"}
    manifest = {"split_manifest_sha256": sha256(split / "manifest.json"), "products_sha256": sha256(split / "products.jsonl"),
        "source_sha256": sha256(ROOT / 'src/phone_assistant/data/prepare_rule_conversations.py'), "counts": counts, "rejected": rejected,
        "evaluation_texts_also_in_train": sum(q["query"] in train_texts for q in queries if q["split"] != "train"),
        "scope": "Rule-generated target-metadata-conditioned proxy; not public observed dialogue, not human labels or evidence of real preference understanding.",
        "limitations": ["Two language variants per goal are correlated; user-goal cohorts are disjoint but all train histories are transductive graph context.",
            "Train query targets are already train-positive edges; query profiles exclude their goal, but graph training legitimately sees the train edge.",
            "One held-out goal per query is incomplete relevance: many catalog phones satisfy the same brand/storage template.",
            "Identical generic query texts may occur across cohorts with different target items; no unseen-language-generalization claim.",
            "No query ASIN/model title, target review, future assistant answer or held-out interaction is inserted into model inputs."],
        "artifacts_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "scope": manifest["scope"], "shared_eval_texts": manifest["evaluation_texts_also_in_train"]}, indent=2))


if __name__ == "__main__":
    main()
