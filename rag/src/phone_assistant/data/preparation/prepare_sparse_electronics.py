"""Controlled training-feedback thinning; leave validation/test, catalog and query labels byte-identical."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil

import numpy as np

from phone_assistant.data.preparation.prepare_recsys import k_core_size, popularity_metrics, write_jsonl
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, sha256

RETAIN_FRACTION = 0.5
MASK_SEED = 20261108
MINIMUM_TRAIN_EVENTS_PER_USER = 1
REBUILT_FILES = frozenset(("manifest.json", "summary.json", "train.jsonl", "train_graph_edges.jsonl",
    "popularity_metrics.json", "selection.sqlite"))


def thin_training(rows, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Retention fraction must be strictly between zero and one")
    grouped = defaultdict(list)
    for row in rows: grouped[row["user_id"]].append(row)
    generator, selected = np.random.default_rng(seed), []
    for user in sorted(grouped):
        candidates = sorted(grouped[user], key=lambda r: (r["timestamp"], r["item_id"]))
        count = max(MINIMUM_TRAIN_EVENTS_PER_USER, math.ceil(len(candidates) * fraction))
        indices = generator.choice(len(candidates), count, replace=False)
        selected.extend(candidates[int(i)] for i in indices)
    return sorted(selected, key=lambda r: (r["user_id"], r["timestamp"], r["item_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / 'data/external/benchmarks/connected_electronics_v2')
    parser.add_argument("--output", type=Path, default=ROOT / 'data/external/benchmarks/electronics_sparse50_v1')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Keep frozen sparsity masks")
    data = InteractionData(args.source)
    train = thin_training(data.rows["train"], RETAIN_FRACTION, MASK_SEED)
    args.output.mkdir(parents=True)
    for name in data.manifest["artifacts_sha256"]:
        if name not in REBUILT_FILES: shutil.copyfile(args.source / name, args.output / name)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "train_graph_edges.jsonl", [{"source": r["user_id"], "target": r["item_id"], "relation": "positive_review"} for r in train])
    pop = {"method": "Pop", "validation": popularity_metrics(train, data.rows["validation"], data.item_ids),
        "test": popularity_metrics(train, data.rows["test"], data.item_ids, exclude=data.rows["validation"])}
    (args.output / "popularity_metrics.json").write_text(json.dumps(pop, indent=2), encoding="utf-8")
    summary = {"users": data.n_users, "items": data.n_items, "train_before": len(data.rows["train"]),
        "train_after": len(train), "actual_retain_fraction": len(train) / len(data.rows["train"]),
        "train_two_core": k_core_size(train), "target_retain_fraction": RETAIN_FRACTION,
        "minimum_train_events_per_user": MINIMUM_TRAIN_EVENTS_PER_USER,
        "limitations": ["Missing-feedback simulation, not measured social noise or a new test population.",
            "Minimum one event per user makes actual retention exceed target on short histories.",
            "Same fixed mask for all model seeds; intervals do not cover mask uncertainty."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"parent_manifest_sha256": sha256(args.source / "manifest.json"), "source_sha256": sha256(Path(__file__)),
        "config": {"retain_fraction": RETAIN_FRACTION, "mask_seed": MASK_SEED},
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = InteractionData(args.output)
    for name in ("validation.jsonl", "test.jsonl", "products.jsonl", "agent_queries.jsonl", "evaluator_labels.jsonl"):
        if sha256(args.source / name) != sha256(args.output / name): raise ValueError("Heldout/catalog/query bytes changed")
    assert result.item_ids == data.item_ids and result.user_ids == data.user_ids
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
