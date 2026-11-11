"""Prepare predeclared train-only feedback masks without changing frozen heldout bytes."""
from phone_assistant.common.project_layout import source_path
import argparse
import json
from pathlib import Path
import shutil

from phone_assistant.data.preparation.prepare_sparse_electronics import thin_training, REBUILT_FILES, MINIMUM_TRAIN_EVENTS_PER_USER
from phone_assistant.data.preparation.prepare_recsys import k_core_size, popularity_metrics, write_jsonl
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, sha256

RETAIN_FRACTIONS = (0.25, 0.5, 0.75)
MASK_SEEDS = (20261108, 20261109)
PERCENT_SCALE = 100


def prepare(source, output, fraction, seed):
    if fraction not in RETAIN_FRACTIONS or seed not in MASK_SEEDS:
        raise ValueError("Mask is outside the predeclared experiment")
    if output.exists():
        raise FileExistsError("Keep frozen masks; validate/reuse instead of replacing them")
    data = InteractionData(source)
    rows = thin_training(data.rows["train"], fraction, seed)
    output.mkdir(parents=True)
    for name in data.manifest["artifacts_sha256"]:
        if name not in REBUILT_FILES:
            shutil.copyfile(source / name, output / name)
    write_jsonl(output / "train.jsonl", rows)
    write_jsonl(output / "train_graph_edges.jsonl", [
        {"source": r["user_id"], "target": r["item_id"], "relation": "positive_review"} for r in rows])
    popularity = {"method": "Pop", "validation": popularity_metrics(rows, data.rows["validation"], data.item_ids),
        "test": popularity_metrics(rows, data.rows["test"], data.item_ids, exclude=data.rows["validation"])}
    (output / "popularity_metrics.json").write_text(json.dumps(popularity, indent=2), encoding="utf-8")
    summary = {"users": data.n_users, "items": data.n_items, "train_before": len(data.rows["train"]),
        "train_after": len(rows), "actual_retain_fraction": len(rows) / len(data.rows["train"]),
        "target_retain_fraction": fraction, "mask_seed": seed, "train_two_core": k_core_size(rows),
        "minimum_train_events_per_user": MINIMUM_TRAIN_EVENTS_PER_USER,
        "limitations": ["Train-only missing-feedback simulation, not measured social noise.",
            "Per-user ceiling increases actual retention above the nominal fraction.",
            "Catalog and heldout population remain fixed; masks do not create independent test populations."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"parent_manifest_sha256": sha256(source / "manifest.json"),
        "source_sha256": sha256(Path(__file__)), "thinning_source_sha256": sha256(source_path('prepare_sparse_electronics.py')),
        "config": {"retain_fraction": fraction, "mask_seed": seed},
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = InteractionData(output)
    for name in ("id_maps.json", "validation.jsonl", "test.jsonl", "products.jsonl", "agent_queries.jsonl", "evaluator_labels.jsonl"):
        if sha256(source / name) != sha256(output / name):
            raise ValueError("Frozen heldout/catalog/query bytes changed")
    if result.item_ids != data.item_ids or result.user_ids != data.user_ids:
        raise ValueError("Mask changed the ID universe")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / 'data/external/benchmarks/connected_electronics_v2')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, choices=RETAIN_FRACTIONS, required=True)
    parser.add_argument("--mask-seed", type=int, choices=MASK_SEEDS, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.fraction, args.mask_seed), indent=2))


if __name__ == "__main__":
    main()
