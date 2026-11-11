"""Build a bounded, metadata-joined RecSys split while preserving conversational holdouts.

Stream the official 5-core archive through SQLite. Never add source interactions
for any base-benchmark user: their frozen past context/goals remain unchanged.
Additional users get chronological leave-last-two-out on positive reviews.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from phone_assistant.data.preparation.prepare_recsys import k_core_size, popularity_metrics, temporal_split, write_jsonl
from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, sha256


def import_catalog_rows(connection, rows, catalog, protected_users):
    audit = Counter()
    batch = []
    for row in rows:
        audit["source_rows"] += 1
        item = row["parent_asin"]
        if item not in catalog:
            continue
        audit["catalog_rows"] += 1
        uid = hashlib.sha256(row["user_id"].encode("utf-8")).hexdigest()
        if uid in protected_users:
            audit["excluded_protected_user_rows"] += 1
            continue
        timestamp, rating = int(row["timestamp"]), float(row["rating"])
        if timestamp <= 0 or not 1 <= rating <= 5:
            raise ValueError("Invalid source timestamp/rating")
        batch.append((uid, item, timestamp, rating))
        if len(batch) >= 5000:
            upsert_events(connection, batch)
            batch.clear()
    upsert_events(connection, batch)
    return dict(audit)


def upsert_events(connection, rows):
    # Latest review wins; lower rating wins ties, before positive filtering.
    connection.executemany("""INSERT INTO events VALUES (?,?,?,?)
        ON CONFLICT(user_id,item_id) DO UPDATE SET timestamp=excluded.timestamp,rating=excluded.rating
        WHERE excluded.timestamp>events.timestamp OR
        (excluded.timestamp=events.timestamp AND excluded.rating<events.rating)""", rows)
    connection.commit()


def connect_cache(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE events (user_id TEXT,item_id TEXT,timestamp INTEGER,rating REAL,PRIMARY KEY(user_id,item_id)) WITHOUT ROWID")
    return connection


def choose_support(connection, *, threshold, min_items, max_users):
    user_ids = [row[0] for row in connection.execute("""SELECT user_id FROM events
        WHERE rating>=? GROUP BY user_id HAVING COUNT(*)>=? ORDER BY user_id LIMIT ?""",
        (threshold, min_items, max_users))]
    events = []
    for uid in user_ids:
        events.extend({"user_id": uid, "item_id": item, "timestamp": timestamp,
                       "rating": rating, "feedback_type": "positive_review"}
                      for item, timestamp, rating in connection.execute(
                          "SELECT item_id,timestamp,rating FROM events WHERE user_id=? AND rating>=? ORDER BY timestamp,item_id",
                          (uid, threshold)))
    return events, user_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / 'data/external/benchmarks/phones_v1')
    parser.add_argument("--source", type=Path, default=ROOT / 'data/external/amazon_5core/Cell_Phones_and_Accessories.csv.gz')
    parser.add_argument("--output", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--max-support-users", type=int, default=2000)
    parser.add_argument("--min-support-items", type=int, default=5)
    parser.add_argument("--positive-rating", type=float, default=4.0)
    args = parser.parse_args()
    if args.min_support_items < 3 or args.max_support_users < 1 or not 1 <= args.positive_rating <= 5:
        parser.error("Need >=3 support items, positive user limit and valid rating threshold")
    if args.output.exists():
        raise FileExistsError("Use a new output; never change an existing split")
    base = InteractionData(args.base)
    source_manifest_path = args.source.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sha256(args.source) != source_manifest["sha256"]:
        raise ValueError("Source archive checksum mismatch")
    base_summary = json.loads((args.base / "summary.json").read_text(encoding="utf-8"))
    threshold = base.manifest["config"]["positive_rating"]
    if args.positive_rating != threshold:
        raise ValueError("Rating threshold must match base benchmark")
    args.output.mkdir(parents=True)
    database = args.output / "selection.sqlite"
    connection = connect_cache(database)
    try:
        with gzip.open(args.source, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            audit = import_catalog_rows(connection, reader, set(base.item_ids), set(base.user_ids))
        audit["unique_catalog_pairs_after_dedup"] = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        audit["eligible_support_users"] = connection.execute("""SELECT COUNT(*) FROM
            (SELECT user_id FROM events WHERE rating>=? GROUP BY user_id HAVING COUNT(*)>=?)""",
            (args.positive_rating, args.min_support_items)).fetchone()[0]
        support, support_ids = choose_support(connection, threshold=args.positive_rating,
                                              min_items=args.min_support_items, max_users=args.max_support_users)
    finally:
        connection.close()
    extra_train, extra_validation, extra_test, tied = temporal_split(support)
    splits = {"train": base.rows["train"] + extra_train,
              "validation": base.rows["validation"] + extra_validation,
              "test": base.rows["test"] + extra_test}
    roles = {uid: "conversation_benchmark" for uid in base.user_ids}
    roles.update({uid: "additional_review_history" for uid in support_ids})
    for name, rows in splits.items():
        rows.sort(key=lambda r: (r["user_id"], r["timestamp"], r["item_id"]))
        write_jsonl(args.output / f"{name}.jsonl", rows)
    # Keep original queries/profiles/labels in separate unchanged files.
    for name in ("products.jsonl", "documents.jsonl", "agent_profiles.jsonl", "agent_queries.jsonl", "evaluator_labels.jsonl"):
        shutil.copyfile(args.base / name, args.output / name)
    write_jsonl(args.output / "train_graph_edges.jsonl", [{"source": r["user_id"], "target": r["item_id"], "relation": "positive_review"} for r in splits["train"]])
    maps = {"users": {uid: i for i, uid in enumerate(sorted(roles))}, "items": base.maps["items"]}
    (args.output / "id_maps.json").write_text(json.dumps(maps, indent=2), encoding="utf-8")
    (args.output / "user_roles.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")
    degrees = Counter(r["item_id"] for r in splits["train"])
    pop = {"method": "Pop", "fit_split": "train_only", "ranking": "full catalog; stable item ID ties"}
    for split in ("validation", "test"):
        pop[split] = popularity_metrics(splits["train"], splits[split], base.item_ids,
                                        exclude=splits["validation"] if split == "test" else ())
    (args.output / "popularity_metrics.json").write_text(json.dumps(pop, indent=2), encoding="utf-8")
    summary = {"domain": base_summary["domain"], "users": len(roles), "items": base.n_items,
               "base_users": base.n_users, "support_users": len(support_ids), "audit": audit,
               "split_events": {name: len(rows) for name, rows in splits.items()},
               "support_tied_users_train_only": tied,
               "train_two_core": k_core_size(splits["train"]),
               "test_groups": {role: {"users": sum(roles[r["user_id"]] == role for r in splits["test"]),
                   "cold_targets": sum(roles[r["user_id"]] == role and not degrees[r["item_id"]] for r in splits["test"])}
                   for role in sorted(set(roles.values()))},
               "base_test_cold_targets_before": sum(not base.item_degree[base.maps["items"][r["item_id"]]] for r in base.rows["test"]),
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "global_time_split": False,
               "limitations": ["Per-user leave-last-out, not global chronological deployment evaluation.",
                   "All original benchmark users excluded from additional archive rows, including their future reviews.",
                   "Other users may interact with a held-out item: warm-item learning is allowed; held-out user-item pairs are forbidden.",
                   "Support users selected by >=5 (configurable) positive distinct catalog items before temporal split: activity-conditioned cohort.",
                   "Fixed catalog from the simulated phone/accessory benchmark; not a representative sample of all Amazon.",
                   "Metadata/images are static transductive catalog, not a time-versioned store.",
                   "Conversational queries remain simulated; additional users have reviews, not dialogue labels."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"schema_version": 1, "base_manifest_sha256": sha256(args.base / "manifest.json"),
                "source_sha256": {"archive": sha256(args.source), "archive_manifest": sha256(source_manifest_path),
                                  "preparer": sha256(Path(__file__))},
                "config": {"max_support_users": args.max_support_users, "min_support_items": args.min_support_items,
                           "positive_rating": args.positive_rating},
                "artifacts_sha256": {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Reopen and audit the actual artifacts, rather than trusting construction alone.
    frozen = InteractionData(args.output)
    for split in ("train", "validation", "test"):
        original = {(r["user_id"], r["item_id"], r["timestamp"], r["rating"]) for r in base.rows[split]}
        actual = {(r["user_id"], r["item_id"], r["timestamp"], r["rating"]) for r in frozen.rows[split] if r["user_id"] in base.maps["users"]}
        if actual != original:
            raise ValueError("Protected benchmark context or labels changed")
    print(json.dumps({"summary": summary, "popularity": pop}, indent=2), flush=True)


if __name__ == "__main__":
    main()
