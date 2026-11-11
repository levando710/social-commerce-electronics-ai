"""Build metadata-only serving documents and reuse verified vector stores without re-encoding."""
from phone_assistant.common.project_layout import source_path
import argparse
import json
from pathlib import Path

import numpy as np

from phone_assistant.retrieval.search.bm25 import build_index
from phone_assistant.common.catalog_schema import catalog_text
from phone_assistant.retrieval.search.dense_search import corpus_fingerprint
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--features", type=Path, default=ROOT / 'artifacts/cache/connected_features_v1')
    parser.add_argument("--images", type=Path, default=ROOT / 'artifacts/cache/connected_feature_work_v1/images')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/cache/connected_runtime_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Runtime exists; retain it and use a new output")
    data = InteractionData(args.split)
    features, feature_manifest = load_features(args.features, data)
    products = {p["parent_asin"]: p for p in read_jsonl(args.split / "products.jsonl")}
    sources = {r["item_id"]: r for r in json.loads((args.features / "image_sources.json").read_text(encoding="utf-8"))}
    documents = []
    for item in data.item_ids:
        product, image = products[item], sources[item]
        paths = []
        if image["status"] == "ok":
            path = args.images / image["file"]
            if sha256(path) != image["sha256"]:
                raise ValueError("Runtime image checksum mismatch")
            paths = [str(path.resolve())]
        documents.append({"doc_id": item, "parent_asin": item, "title": product["title"],
            "brand": product["details"].get("Brand"), "details": product["details"],
            "categories": product.get("categories", []), "text": catalog_text(product),
            "product_image_paths": paths, "review_image_paths": [], "reviews": [], "visual_evidence": []})
    args.output.mkdir(parents=True)
    (args.output / "documents.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents), encoding="utf-8")
    (args.output / "bm25_index.json").write_text(json.dumps(build_index(documents)), encoding="utf-8")
    dense = args.output / "dense_index"; dense.mkdir()
    np.savez_compressed(dense / "text.npz", vectors=features["text"], doc_ids=np.asarray(data.item_ids))
    mask = features["image_mask"]
    np.savez_compressed(dense / "clip.npz", vectors=features["image"][mask], doc_ids=np.asarray(data.item_ids)[mask])
    dense_manifest = {"schema_version": 1, "corpus_sha256": corpus_fingerprint(documents),
        "source_feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "channels": {channel: {**feature_manifest[key], "file": f"{channel}.npz",
                    "pooling": "normalized mean metadata chunks" if channel == "text" else "one image per item"}
                    for channel, key in (("text", "text_encoder"), ("clip", "image_encoder"))}}
    (dense / "manifest.json").write_text(json.dumps(dense_manifest, indent=2), encoding="utf-8")
    manifest = {"split_manifest_sha256": sha256(args.split / "manifest.json"),
        "feature_manifest_sha256": sha256(args.features / "manifest.json"), "documents": len(documents),
        "policy": "catalog metadata and verified image files; no evaluator goals, histories or target reviews",
        "source_sha256": {n: sha256(source_path(n)) for n in ("prepare_connected_runtime.py", "catalog_schema.py")},
        "artifacts_sha256": {p.relative_to(args.output).as_posix(): sha256(p) for p in args.output.rglob("*") if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"documents": len(documents), "output": str(args.output)}))


if __name__ == "__main__":
    main()
