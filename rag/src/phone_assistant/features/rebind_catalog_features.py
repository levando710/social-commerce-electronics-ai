"""Reuse verified metadata-only vectors across splits with exactly identical ordered catalogs."""
import argparse
import json
from pathlib import Path
import shutil

from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256


def rebind(source_split, source_features, target_split, output):
    if output.exists():
        raise FileExistsError("Do not replace a frozen feature store")
    original, target = InteractionData(source_split), InteractionData(target_split)
    _, manifest = load_features(source_features, original)
    if original.item_ids != target.item_ids:
        raise ValueError("Ordered item IDs differ")
    if sha256(target_split / "products.jsonl") != manifest["source_metadata_sha256"]:
        raise ValueError("Target metadata differs; re-encode rather than reusing features")
    output.mkdir(parents=True)
    for name, digest in manifest["artifacts_sha256"].items():
        shutil.copyfile(source_features / name, output / name)
        if sha256(output / name) != digest:
            raise ValueError("Copied feature bytes changed")
    manifest = {**manifest, "split_manifest_sha256": sha256(target_split / "manifest.json"),
        "reuse_provenance": {"source_feature_manifest_sha256": sha256(source_features / "manifest.json"),
            "source_split_manifest_sha256": sha256(source_split / "manifest.json"),
            "target_split_manifest_sha256": sha256(target_split / "manifest.json"),
            "source_sha256": sha256(Path(__file__)),
            "policy": "Identical ordered items and product metadata; frozen metadata/image features contain no interaction labels; artifact bytes unchanged."}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_features(output, target)
    print(json.dumps({"items": target.n_items, "output": str(output), "reencoded": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, default=ROOT / 'data/external/benchmarks/electronics_v1')
    parser.add_argument("--source-features", type=Path, default=ROOT / 'artifacts/cache/electronics_catalog_features_v1')
    parser.add_argument("--target-split", type=Path, default=ROOT / 'data/external/benchmarks/connected_electronics_v2')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/cache/electronics_features_v1')
    args = parser.parse_args()
    rebind(args.source_split, args.source_features, args.target_split, args.output)


if __name__ == "__main__":
    main()
