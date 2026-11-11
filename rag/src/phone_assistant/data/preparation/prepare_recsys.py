"""Audit review interactions and export a deterministic per-user temporal split.

Reviews are a proxy for feedback, not fabricated clicks/purchases/social edges.
Only training interactions are exported as graph edges and counted by Pop.
"""
from __future__ import annotations
from phone_assistant.common.project_layout import MODULE_ROOT

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = MODULE_ROOT


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def collect_interactions(reviews, item_ids, positive_rating=4.0):
    latest = {}
    audit = Counter()
    for row in reviews:
        user, item = row.get("user_id"), row.get("parent_asin")
        if not user or item not in item_ids:
            audit["missing_user_or_outside_catalog"] += 1
            continue
        try:
            timestamp = int(row["timestamp"])
            rating = float(row["rating"])
            if timestamp <= 0 or not math.isfinite(rating) or not 1 <= rating <= 5:
                raise ValueError("Invalid timestamp or rating")
        except (KeyError, TypeError, ValueError, OverflowError):
            audit["invalid_timestamp_or_rating"] += 1
            continue
        user_id = hashlib.sha256(str(user).encode("utf-8")).hexdigest()
        interaction = {"user_id": user_id, "item_id": item, "timestamp": timestamp,
                       "rating": rating, "feedback_type": "positive_review"}
        key = (user_id, item)
        if key in latest:
            audit["duplicate_user_item_rows"] += 1
        # Deterministic conservative tie break: retain the lower rating.
        if key not in latest or (timestamp, -rating) > (latest[key]["timestamp"], -latest[key]["rating"]):
            latest[key] = interaction
    positive = [row for row in latest.values() if row["rating"] >= positive_rating]
    audit["unique_pairs_before_rating_filter"] = len(latest)
    audit["nonpositive_pairs"] = len(latest) - len(positive)
    return sorted(positive, key=lambda row: (row["user_id"], row["timestamp"], row["item_id"])), dict(audit)


def temporal_split(interactions):
    by_user = defaultdict(list)
    for row in interactions:
        by_user[row["user_id"]].append(row)
    train, validation, test = [], [], []
    tied_users = 0
    for user in sorted(by_user):
        rows = sorted(by_user[user], key=lambda row: (row["timestamp"], row["item_id"]))
        if len(rows) < 2:
            train.extend(rows)
            continue
        # A held-out event must be strictly newer than this user's training data.
        if rows[-1]["timestamp"] == rows[-2]["timestamp"]:
            train.extend(rows)
            tied_users += 1
        elif len(rows) >= 3 and rows[-2]["timestamp"] > rows[-3]["timestamp"]:
            train.extend(rows[:-2])
            validation.append(rows[-2])
            test.append(rows[-1])
        else:
            train.extend(rows[:-1])
            test.append(rows[-1])
    return train, validation, test, tied_users


def k_core_size(interactions, k=2):
    rows = interactions
    while rows:
        users = Counter(row["user_id"] for row in rows)
        items = Counter(row["item_id"] for row in rows)
        retained = [row for row in rows if users[row["user_id"]] >= k and items[row["item_id"]] >= k]
        if len(retained) == len(rows):
            break
        rows = retained
    return {"interactions": len(rows), "users": len({r['user_id'] for r in rows}),
            "items": len({r['item_id'] for r in rows})}


def popularity_metrics(train, heldout, catalog, *, exclude=(), k_values=(5, 10, 20)):
    counts = Counter(row["item_id"] for row in train)
    ranking = sorted(catalog, key=lambda item: (-counts[item], item))
    seen = defaultdict(set)
    for row in list(train) + list(exclude):
        seen[row["user_id"]].add(row["item_id"])
    groups = {"all": [], "warm_item": [], "cold_item": []}
    for event in heldout:
        candidates = [item for item in ranking if item not in seen[event["user_id"]]]
        rank = candidates.index(event["item_id"]) + 1
        groups["all"].append(rank)
        groups["warm_item" if counts[event["item_id"]] else "cold_item"].append(rank)
    result = {}
    for name, ranks in groups.items():
        metrics = {"users": len(ranks)}
        for k in k_values:
            metrics[f"hr@{k}"] = sum(rank <= k for rank in ranks) / len(ranks) if ranks else None
            metrics[f"ndcg@{k}"] = (sum(1 / math.log2(rank + 1) for rank in ranks if rank <= k)
                                     / len(ranks)) if ranks else None
        result[name] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path,
                        default=ROOT / 'data/collections/phones_filtered/processed/reviews.jsonl')
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog/documents.jsonl')
    parser.add_argument("--output", type=Path, default=ROOT / 'data/fixtures/recommendation')
    parser.add_argument("--positive-rating", type=float, default=4.0)
    args = parser.parse_args()
    if not 1 <= args.positive_rating <= 5:
        parser.error("positive-rating must lie between 1 and 5")
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Split already frozen. Use a new --output for a different experiment.")
    documents = read_jsonl(args.corpus)
    catalog = {doc["doc_id"] for doc in documents}
    reviews = read_jsonl(args.reviews)
    interactions, audit = collect_interactions(reviews, catalog, args.positive_rating)
    train, validation, test, tied_users = temporal_split(interactions)
    counts = Counter(row["user_id"] for row in interactions)
    item_counts = Counter(row["item_id"] for row in interactions)
    train_users = sorted({row["user_id"] for row in train})
    catalog_list = sorted(catalog)
    user_map = {user: idx for idx, user in enumerate(train_users)}
    item_map = {item: idx for idx, item in enumerate(catalog_list)}
    graph_edges = [{"user_index": user_map[row["user_id"]],
                    "item_index": item_map[row["item_id"]], "edge_type": "positive_review"}
                   for row in train]
    metrics = {
        "method": "Pop", "fit_split": "train_only",
        "ranking": "full catalog; popularity descending, item ID ascending for ties",
        "validation": popularity_metrics(train, validation, catalog),
        "test": popularity_metrics(train, test, catalog, exclude=validation),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_rows": len(reviews), "catalog_items": len(catalog), "audit": audit,
        "positive_rating_threshold": args.positive_rating,
        "positive_interactions": len(interactions), "users": len(counts),
        "items_with_positive_interactions": len(item_counts),
        "users_with_multiple_items": sum(value >= 2 for value in counts.values()),
        "user_degree_histogram": dict(sorted(Counter(counts.values()).items())),
        "interaction_density": len(interactions) / (len(counts) * len(catalog)) if counts and catalog else 0,
        "two_core": k_core_size(interactions),
        "split": {"train": len(train), "validation": len(validation), "test": len(test),
                  "timestamp_tied_users_kept_train_only": tied_users},
        "protocol": "per-user chronological leave-last-out; validation for users with >=3 separable events",
        "global_time_split": False,
        "limitations": [
            "Reviews are feedback proxies, not observed clicks, follows or confirmed transactions.",
            "The collected product/review subset is not representative of Amazon or a social network.",
            "Per-user chronology does not enforce a shared global time cutoff across different users.",
            "Low-degree users remain train-only. Do not fabricate interactions to increase evaluation size.",
            "Cold-item results are included and also reported separately; no sampled negatives.",
            "Historical product text/reviews are not a temporally frozen feature store for future GNN evaluation.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("interactions", interactions), ("train", train),
                       ("validation", validation), ("test", test), ("train_graph_edges", graph_edges)):
        write_jsonl(args.output / f"{name}.jsonl", rows)
    (args.output / "id_maps.json").write_text(json.dumps({"users": user_map, "items": item_map}, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "popularity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["# Kiểm tra dữ liệu và baseline RecSys", "",
        f"- Review nguồn: {len(reviews)}; sản phẩm trong catalog: {len(catalog)}.",
        f"- Tương tác tích cực (rating >= {args.positive_rating:g}, sau gộp trùng): {len(interactions)}.",
        f"- Người dùng: {len(counts)}; người có ít nhất 2 sản phẩm: {summary['users_with_multiple_items']}.",
        f"- Train / validation / test: {len(train)} / {len(validation)} / {len(test)} tương tác.",
        f"- 2-core (user và item cùng bậc >= 2): {summary['two_core']}.", "",
        "## Pop: full-catalog ranking, chỉ học từ train", "",
        "| Split / nhóm | Số người | HR@5 | NDCG@5 | HR@10 | NDCG@10 |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        for group, values in metrics[split].items():
            cells = ["N/A" if values[key] is None else f"{values[key]:.4f}"
                     for key in ("hr@5", "ndcg@5", "hr@10", "ndcg@10")]
            lines.append(f"| {split} / {group} | {values['users']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Giới hạn diễn giải", ""] + [f"- {line}" for line in summary["limitations"]])
    lines.extend(["", "User ID được băm SHA-256; cạnh đồ thị chỉ lấy từ train.",
                  "Pop được fit bằng train; khi đánh giá test, loại các item đã thấy ở train và validation.",
                  "Chưa huấn luyện GNN/GCL. Cần mở rộng tương tác trước khi kết luận hiệu quả cá nhân hóa.", ""])
    (args.output / "recsys_data_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"schema_version": 1, "source_sha256": {
        "reviews": hashlib.sha256(args.reviews.read_bytes()).hexdigest(),
        "corpus": hashlib.sha256(args.corpus.read_bytes()).hexdigest()},
        "artifacts_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(args.output.iterdir()) if path.is_file()},
        "config": {"positive_rating": args.positive_rating, "protocol": summary["protocol"]}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "popularity": metrics}, indent=2))


if __name__ == "__main__":
    main()
