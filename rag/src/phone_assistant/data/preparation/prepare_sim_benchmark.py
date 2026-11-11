"""Freeze a history-conditioned benchmark without exposing simulator goals to agents.

Uses observed Amazon review histories and LLM-simulated first user messages.
All cohorts' strictly past histories are training context (transductive users).
Only training-cohort targets enter training; validation/test target labels are hidden.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import pyarrow.parquet as pq

from phone_assistant.data.preparation.prepare_recsys import popularity_metrics, write_jsonl, k_core_size
from phone_assistant.recommendation.recsys_data import ROOT, metadata_text, sha256

DOMAINS = ("Cell_Phones_and_Accessories", "Electronics")


def user_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def clean_user(row, catalog, threshold=4.0, min_history=2):
    """Deduplicate before rating filtering; reject previously seen target items."""
    target = row["target"]
    item, timestamp, rating = target["item"], int(target["ts"]), float(target["rating"])
    audit = Counter()
    if item not in catalog or timestamp <= 0 or not math.isfinite(rating) or not threshold <= rating <= 5:
        return None, {"excluded_invalid_target": 1}
    # A repeat purchase is not a novel-item prediction, even when the old rating was low.
    if any(event["item"] == item and int(event["ts"]) < timestamp for event in row["interaction_history"]):
        return None, {"excluded_previously_seen_target": 1}
    latest = {}
    for event in row["interaction_history"]:
        event_ts, event_rating = int(event["ts"]), float(event["rating"])
        if event_ts <= 0 or event_ts >= timestamp or event["item"] == item:
            audit["removed_nonpast_or_target_history_events"] += 1
            continue
        if event["item"] not in catalog:
            audit["removed_unresolved_history_events"] += 1
            continue
        if not math.isfinite(event_rating) or not 1 <= event_rating <= 5:
            audit["removed_invalid_rating_events"] += 1
            continue
        old = latest.get(event["item"])
        if old is not None:
            audit["duplicate_past_item_events"] += 1
        if old is None or (event_ts, -event_rating) > (old["ts"], -old["rating"]):
            latest[event["item"]] = {"item": event["item"], "ts": event_ts, "rating": event_rating}
    uid = user_hash(row["user_id"])
    def event_record(event):
        return {"user_id": uid, "item_id": event["item"], "timestamp": int(event["ts"]),
                "rating": float(event["rating"]), "feedback_type": "positive_review"}
    history = [event_record(event) for event in latest.values() if event["rating"] >= threshold]
    audit["nonpositive_unique_past_items"] += sum(event["rating"] < threshold for event in latest.values())
    if len(history) < min_history:
        audit["excluded_insufficient_positive_history"] += 1
        return None, dict(audit)
    return {"user_id": uid, "history": sorted(history, key=lambda e: (e["timestamp"], e["item_id"])),
            "target": event_record(target)}, dict(audit)


def assign_cohorts(user_ids):
    ordered = sorted(user_ids)
    n = len(ordered)
    return {uid: "train" if index < int(n * .8) else "validation" if index < int(n * .9) else "test"
            for index, uid in enumerate(ordered)}


def initial_query(response):
    # Never use simulator prompt, hidden target review, persona or assistant future turns.
    match = re.match(r"\s*\[USR\]\s*:\s*(.*?)(?=\[(?:CRS|USR)\]\s*:|\Z)", response, re.S)
    return match.group(1).strip() if match else None


def safe_product(row):
    details = row.get("details") or {}
    if isinstance(details, str):
        details = json.loads(details)
    return {"parent_asin": row["parent_asin"], "title": row.get("title") or "",
            "store": row.get("store") or "", "features": row.get("features") or [],
            "description": row.get("description") or [], "categories": row.get("categories") or [],
            "details": details, "images": row.get("images") or []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / 'data/external/ecommerce_user_sim')
    parser.add_argument("--domain", choices=DOMAINS, default=DOMAINS[0])
    parser.add_argument("--output", type=Path, default=ROOT / 'data/external/benchmarks/phones_v1')
    parser.add_argument("--positive-rating", type=float, default=4.0)
    parser.add_argument("--min-history", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.positive_rating <= 5 or args.min_history < 1:
        parser.error("Invalid rating threshold or minimum history")
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve the frozen benchmark")
    paths = {kind: args.source / f"corpus/{kind}/{args.domain}.parquet" for kind in ("users", "items", "dialogues")}
    download_manifest = json.loads((args.source.parent / "download_manifest.json").read_text(encoding="utf-8"))
    expected = {record["path"]: record["sha256"] for record in download_manifest["ecommerce_user_sim"]["files"]}
    for path in paths.values():
        if sha256(path) != expected[path.relative_to(args.source).as_posix()]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
    raw = {kind: pq.read_table(path).to_pylist() for kind, path in paths.items()}
    products = sorted((safe_product(row) for row in raw["items"]), key=lambda row: row["parent_asin"])
    catalog = [row["parent_asin"] for row in products]
    if len(set(catalog)) != len(catalog):
        raise ValueError("Duplicate catalog IDs")
    users, audit = {}, Counter()
    for row in raw["users"]:
        user, reasons = clean_user(row, set(catalog), args.positive_rating, args.min_history)
        audit.update(reasons)
        if user is not None:
            if user["user_id"] in users:
                raise ValueError("Duplicate canonical user")
            users[user["user_id"]] = user
    cohorts = assign_cohorts(users)
    splits = {name: [] for name in ("train", "validation", "test")}
    contexts, labels = [], []
    for uid, user in sorted(users.items()):
        splits["train"].extend(user["history"])
        splits[cohorts[uid]].append(user["target"])
        contexts.append({"user_id": uid, "cohort": cohorts[uid], "history": user["history"]})
        labels.append({"user_id": uid, "cohort": cohorts[uid], **{k: v for k, v in user["target"].items() if k != "user_id"}})
    if not all(splits.values()):
        raise ValueError("Need nonempty train, validation and test")
    raw_ids = {row["user_id"] for row in raw["users"]}
    queries = []
    for index, row in enumerate(raw["dialogues"]):
        original = row["user_id"] if row["user_id"] in raw_ids else re.sub(r"_\d+$", "", row["user_id"])
        uid = user_hash(original)
        if uid not in users:
            continue
        query = initial_query(row["response"])
        if not query:
            audit["dialogues_without_parseable_first_user_message"] += 1
            continue
        queries.append({"dialogue_id": f"{args.domain}:{index}", "user_id": uid,
                        "split": cohorts[uid], "query": query})
    args.output.mkdir(parents=True)
    for name, rows in splits.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
    write_jsonl(args.output / "agent_profiles.jsonl", contexts)
    write_jsonl(args.output / "agent_queries.jsonl", queries)
    write_jsonl(args.output / "evaluator_labels.jsonl", labels)
    write_jsonl(args.output / "products.jsonl", products)
    write_jsonl(args.output / "documents.jsonl", [{"doc_id": p["parent_asin"], "title": p["title"], "text": metadata_text(p)} for p in products])
    write_jsonl(args.output / "train_graph_edges.jsonl", [{"source": r["user_id"], "target": r["item_id"], "relation": "positive_review"} for r in splits["train"]])
    maps = {"users": {uid: i for i, uid in enumerate(sorted(users))}, "items": {item: i for i, item in enumerate(catalog)}}
    (args.output / "id_maps.json").write_text(json.dumps(maps, indent=2), encoding="utf-8")
    pop = {"method": "Pop", "fit_split": "train_only", "ranking": "full catalog, stable item ID ties",
           "validation": popularity_metrics(splits["train"], splits["validation"], catalog),
           "test": popularity_metrics(splits["train"], splits["test"], catalog, exclude=splits["validation"])}
    (args.output / "popularity_metrics.json").write_text(json.dumps(pop, indent=2), encoding="utf-8")
    summary = {"domain": args.domain, "source_users": len(raw["users"]), "eligible_users": len(users),
               "catalog_items": len(catalog), "cohorts": dict(Counter(cohorts.values())),
               "split_events": {k: len(v) for k, v in splits.items()},
               "positive_history_events": sum(len(u["history"]) for u in users.values()),
               "queries": dict(Counter(q["split"] for q in queries)), "audit": dict(audit),
               "two_core": k_core_size(splits["train"]),
               "protocol": "80/10/10 canonical user goal cohorts; all prior histories are transductive train context; train goals only added for train cohort",
               "global_time_split": False,
               "limitations": ["Cell Phones and Accessories includes accessories, not only handsets.",
                   "First user messages are simulated; histories are review feedback, not verified purchases or clicks.",
                   "Goal-held-out users have history edges in training: this is not unseen-user generalization.",
                   "Per-user temporal context, not global chronology; other users can have later train events.",
                   "Static catalog metadata not verified available at each historical event time.",
                   "A single hidden target is incomplete relevance: other suitable items are not labeled.",
                   "No dialogue continuation or human user study is evaluated in this benchmark."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"schema_version": 1, "source_revision": download_manifest["ecommerce_user_sim"].get("revision"),
                "source_sha256": {k: sha256(v) for k, v in paths.items()},
                "config": {"positive_rating": args.positive_rating, "min_history": args.min_history},
                "artifacts_sha256": {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "popularity": pop}, indent=2))


if __name__ == "__main__":
    main()
