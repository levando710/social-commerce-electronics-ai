"""Build a deterministic RecSys split from one Amazon Reviews 2023 category (item-sampled 5-core, leave-last-out).

The official 5-core archive of a whole category is far larger than this CPU study can train with
images, so items are sampled uniformly with a fixed seed *before* any model is seen, then a user/item
k-core is re-applied.  Output has the same layout as the Electronics split, so the existing feature
builder and portable trainer run unchanged.  This is a RecSys-only benchmark: no conversational
queries exist for it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from phone_assistant.data.preparation.prepare_recsys import popularity_metrics
from phone_assistant.features.build_connected_features import image_urls
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256

SOURCE = ROOT / "data/external/amazon_2023_baby_products"
OUTPUT = ROOT / "data/external/benchmarks/amazon2023_baby_products_v1"
POSITIVE_RATING = 4.0
MIN_USER_ITEMS = 5
MIN_ITEM_USERS = 5
SAMPLED_ITEMS = 12000
SEED = 20261110
FEEDBACK_TYPE = "positive_review"


def read_positives(rating_csv, positive_rating=POSITIVE_RATING):
    """user -> item -> (timestamp, rating); a repeated pair keeps the newer event, then the lower rating."""
    result = defaultdict(dict)
    with open(rating_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rating, stamp = float(row["rating"]), int(row["timestamp"])
            if rating < positive_rating:
                continue
            current = result[row["user_id"]].get(row["parent_asin"])
            if current is None or (stamp, -rating) > (current[0], -current[1]):
                result[row["user_id"]][row["parent_asin"]] = (stamp, rating)
    return result


def items_with_metadata(meta_path):
    """Return {asin} for items that have a title and a usable catalog image URL (one streaming pass)."""
    usable = set()
    with open(meta_path, encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            if product.get("parent_asin") and str(product.get("title") or "").strip() and image_urls(product):
                usable.add(product["parent_asin"])
    return usable


def sample_items(candidates, count, seed=SEED):
    ordered = sorted(candidates)
    return set(random.Random(seed).sample(ordered, min(count, len(ordered))))


def k_core(interactions, min_user=MIN_USER_ITEMS, min_item=MIN_ITEM_USERS):
    data = {user: dict(items) for user, items in interactions.items()}
    while True:
        counts = Counter(item for items in data.values() for item in items)
        keep = {item for item, count in counts.items() if count >= min_item}
        reduced = {user: {i: v for i, v in items.items() if i in keep} for user, items in data.items()}
        reduced = {user: items for user, items in reduced.items() if len(items) >= min_user}
        if reduced == data:
            return data
        data = reduced


def leave_last_out(data):
    """Per user by time (ties by item ID): last event is test, the one before is validation, the rest train."""
    split = {"train": [], "validation": [], "test": []}
    for user in sorted(data):
        events = sorted(((stamp, item, rating) for item, (stamp, rating) in data[user].items()))
        for position, (stamp, item, rating) in enumerate(events):
            name = "test" if position == len(events) - 1 else "validation" if position == len(events) - 2 else "train"
            split[name].append({"user_id": user, "item_id": item, "timestamp": stamp, "rating": rating,
                                "feedback_type": FEEDBACK_TYPE})
    return split


def popularity_report(train, validation, test, catalog):
    """Popularity baseline fitted on train only, same protocol as the other splits (full catalog, stable ties)."""
    return {"method": "Pop", "fit_split": "train_only", "ranking": "full catalog; stable item ID ties",
            "validation": popularity_metrics(train, validation, catalog),
            "test": popularity_metrics(train, test, catalog, exclude=validation)}


def write_popularity_sidecar(split_dir):
    """Add ``popularity_metrics.json`` to an existing split without touching its manifest.

    The first Baby split was frozen before this file was part of the preparer; its manifest, and every
    feature set and checkpoint bound to that manifest, must stay unchanged.  The file is therefore not
    listed in that manifest; the aggregated experiment file carries its content.
    """
    split_dir = Path(split_dir)
    target = split_dir / "popularity_metrics.json"
    if target.exists():
        raise FileExistsError("popularity_metrics.json already exists; not overwriting")
    rows = {name: read_jsonl(split_dir / f"{name}.jsonl") for name in ("train", "validation", "test")}
    maps = json.loads((split_dir / "id_maps.json").read_text(encoding="utf-8"))
    catalog = sorted(maps["items"], key=maps["items"].get)
    target.write_text(json.dumps(popularity_report(rows["train"], rows["validation"], rows["test"], catalog), indent=2),
                      encoding="utf-8")
    return sha256(target)


def write_jsonl(path, rows):
    # ASCII escapes keep U+2028/U+2029 inside strings; the shared reader splits lines with str.splitlines().
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def prepare(rating_csv, meta_path, output, sampled_items=SAMPLED_ITEMS, seed=SEED):
    if output.exists():
        raise FileExistsError("Preserve the benchmark; use a new output to revise it")
    positives = read_positives(rating_csv)
    usable = items_with_metadata(meta_path)
    candidates = {item for items in positives.values() for item in items} & usable
    chosen = sample_items(candidates, sampled_items, seed)
    restricted = {user: {i: v for i, v in items.items() if i in chosen} for user, items in positives.items()}
    data = k_core({user: items for user, items in restricted.items() if len(items) >= MIN_USER_ITEMS})
    split = leave_last_out(data)
    users = sorted(data)
    items = sorted({item for events in data.values() for item in events})
    products = []
    wanted = set(items)
    with open(meta_path, encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            if product.get("parent_asin") in wanted:
                products.append(product)
                wanted.discard(product["parent_asin"])
    if wanted:
        raise ValueError("Catalog metadata missing for selected items")
    products.sort(key=lambda product: product["parent_asin"])
    output.mkdir(parents=True)
    write_jsonl(output / "products.jsonl", products)
    for name, rows in split.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    write_jsonl(output / "train_graph_edges.jsonl", [
        {"source": row["user_id"], "target": row["item_id"], "relation": FEEDBACK_TYPE} for row in split["train"]])
    (output / "id_maps.json").write_text(json.dumps({"users": {u: i for i, u in enumerate(users)},
                                                     "items": {p: i for i, p in enumerate(items)}}, indent=2), encoding="utf-8")
    (output / "popularity_metrics.json").write_text(
        json.dumps(popularity_report(split["train"], split["validation"], split["test"], items), indent=2), encoding="utf-8")
    train_items = {row["item_id"] for row in split["train"]}
    summary = {"users": len(users), "items": len(items), "train": len(split["train"]),
               "validation": len(split["validation"]), "test": len(split["test"]),
               "test_targets_without_train_edge": sum(row["item_id"] not in train_items for row in split["test"]),
               "candidate_items_before_sampling": len(candidates)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts = ("products.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl", "train_graph_edges.jsonl",
                 "id_maps.json", "popularity_metrics.json", "summary.json")
    manifest = {"schema_version": 1, "dataset": "Amazon Reviews 2023, Baby_Products (official 5-core, item-sampled)",
                "source_sha256": {"rating_csv": sha256(rating_csv), "meta": sha256(meta_path),
                                  "preparer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                "config": {"positive_rating": POSITIVE_RATING, "min_user_items": MIN_USER_ITEMS,
                           "min_item_users": MIN_ITEM_USERS, "sampled_items": sampled_items, "seed": seed,
                           "split": "per-user leave-last-out by timestamp; last=test, second last=validation"},
                "artifacts_sha256": {name: sha256(output / name) for name in artifacts},
                "limitations": ["Item-sampled subset of the official 5-core; not the full category.",
                                "RecSys-only: no conversational queries or relevance labels exist for this benchmark.",
                                "Cold targets (no train edge) are kept and reported, not removed.",
                                "Spec hierarchy covers few attributes outside consumer electronics."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rating-csv", type=Path, default=SOURCE / "Baby_Products.csv")
    parser.add_argument("--meta", type=Path, default=SOURCE / "meta_Baby_Products.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sampled-items", type=int, default=SAMPLED_ITEMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--popularity-sidecar", action="store_true",
                        help="only add popularity_metrics.json to the existing split at --output (manifest untouched)")
    args = parser.parse_args()
    if args.popularity_sidecar:
        print(json.dumps({"popularity_metrics_sha256": write_popularity_sidecar(args.output)}))
        return
    print(json.dumps(prepare(args.rating_csv, args.meta, args.output, args.sampled_items, args.seed), indent=2))


if __name__ == "__main__":
    main()
