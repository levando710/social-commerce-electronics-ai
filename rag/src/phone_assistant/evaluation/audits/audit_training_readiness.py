"""Validate frozen splits, per-user chronology, protected goals and feature coverage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, load_features, read_jsonl, sha256


def audit(split, base=None, features=None):
    data = InteractionData(split)
    by_user = {name: defaultdict(list) for name in data.rows}
    for name, rows in data.rows.items():
        for row in rows:
            by_user[name][row["user_id"]].append(row["timestamp"])
    for uid in data.user_ids:
        train = by_user["train"][uid]
        validation, test = by_user["validation"][uid], by_user["test"][uid]
        if not train:
            raise ValueError("User without training history")
        if len(validation) > 1 or len(test) > 1:
            raise ValueError("This protocol requires at most one validation/test target per user")
        if validation and max(train) >= min(validation):
            raise ValueError("Training is not strictly before validation")
        if test and max(train + validation) >= min(test):
            raise ValueError("Context is not strictly before test")
    products = read_jsonl(Path(split) / "products.jsonl")
    catalog = {row["parent_asin"]: row for row in products}
    if set(data.item_ids) - catalog.keys():
        raise ValueError("Catalog metadata is missing")
    if base:
        original = InteractionData(base)
        for name in data.rows:
            expected = sorted((r["user_id"], r["item_id"], r["timestamp"], r["rating"]) for r in original.rows[name])
            actual = sorted((r["user_id"], r["item_id"], r["timestamp"], r["rating"]) for r in data.rows[name] if r["user_id"] in original.maps["users"])
            if actual != expected:
                raise ValueError("Original benchmark user context/labels changed")
        for name in ("agent_queries.jsonl", "agent_profiles.jsonl", "evaluator_labels.jsonl"):
            if sha256(Path(split) / name) != sha256(Path(base) / name):
                raise ValueError("Original conversation benchmark artifact changed")
    degree = Counter(row["user_id"] for row in data.rows["train"])
    result = {"split_manifest_sha256": sha256(Path(split) / "manifest.json"),
              "passed": True, "users": data.n_users, "items": data.n_items,
              "events": {name: len(rows) for name, rows in data.rows.items()},
              "train_user_degree": {"min": min(degree.values()), "max": max(degree.values()),
                                    "mean": sum(degree.values()) / len(degree)},
              "items_with_image_urls": sum(bool(catalog[item].get("images")) for item in data.item_ids),
              "metadata_covered": data.n_items, "protected_base_verified": bool(base),
              "lightgcn_ready": True, "multimodal_features_ready": False,
              "checks": ["all frozen artifact hashes", "disjoint unique user-item split pairs",
                         "strict per-user train < validation < test where present", "metadata coverage",
                         "original benchmark context and labels unchanged" if base else "no base supplied"],
              "limitations": ["Chronology checks are per user, not global.",
                              "Image URLs do not imply downloaded/decoded images or CLIP feature readiness.",
                              "No peak memory or training duration guarantee from this audit."]}
    if features:
        _, manifest = load_features(features, data)
        result["multimodal_features_ready"] = True
        result["feature_manifest_sha256"] = sha256(Path(features) / "manifest.json")
        result["items_with_clip_images"] = manifest["items_with_images"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--base", type=Path, default=ROOT / 'data/external/benchmarks/phones_v1')
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / 'data/discovery/connected_training_audit.json')
    args = parser.parse_args()
    result = audit(args.split, args.base, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
