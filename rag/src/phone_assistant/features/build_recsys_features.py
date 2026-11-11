"""Encode catalog metadata only; never reuse text vectors containing held-out reviews."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from phone_assistant.retrieval.search.dense_search import load_encoder, text_chunks
from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, metadata_text, read_jsonl, sha256, spec_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/fixtures/recommendation')
    parser.add_argument("--products", type=Path, default=ROOT / 'data/collections/phones_filtered/processed/products.jsonl')
    parser.add_argument("--dense-index", type=Path, default=ROOT / 'artifacts/runtime/catalog/dense_index')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/features/legacy')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch-size and threads must be positive")
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Feature store exists; use a new --output to retain provenance.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    data = InteractionData(args.split)
    products = {row["parent_asin"]: row for row in read_jsonl(args.products)}
    missing = set(data.item_ids) - products.keys()
    if missing:
        raise ValueError(f"Missing catalog metadata for {len(missing)} items")
    dense_manifest = json.loads((args.dense_index / "manifest.json").read_text(encoding="utf-8"))
    config = dense_manifest["channels"]["text"]
    encoder = load_encoder(config["model"], revision=config["revision"])
    chunks, owners = [], []
    for index, item_id in enumerate(data.item_ids):
        item_chunks = text_chunks({"text": metadata_text(products[item_id])})
        chunks.extend(item_chunks)
        owners.extend([index] * len(item_chunks))
    started = time.perf_counter()
    vectors = encoder.encode(chunks, batch_size=args.batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)
    text = np.zeros((data.n_items, vectors.shape[1]), dtype=np.float32)
    np.add.at(text, owners, vectors)
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    with np.load(args.dense_index / "clip.npz", allow_pickle=False) as clip:
        image = np.zeros((data.n_items, clip["vectors"].shape[1]), dtype=np.float32)
        counts = np.zeros(data.n_items, dtype=np.int64)
        for item_id, vector in zip(clip["doc_ids"], clip["vectors"]):
            if str(item_id) in data.maps["items"]:
                index = data.maps["items"][str(item_id)]
                image[index] += vector
                counts[index] += 1
    image /= np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    specs = {item_id: spec_tokens(products[item_id]) for item_id in data.item_ids}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "features.npz", item_ids=np.asarray(data.item_ids),
                        text=text, image=image, image_mask=counts > 0)
    (args.output / "specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_manifest_sha256": sha256(data.directory / "manifest.json"),
        "source_metadata_sha256": sha256(args.products),
        "source_clip_sha256": sha256(args.dense_index / "clip.npz"),
        "text_policy": "catalog_metadata_allowlist_no_reviews_no_ratings",
        "text_encoder": config, "image_encoder": dense_manifest["channels"]["clip"],
        "items": data.n_items, "text_chunks": len(chunks),
        "text_dimensions": text.shape[1], "image_dimensions": image.shape[1],
        "items_with_images": int((counts > 0).sum()),
        "spec_nodes": len({token for values in specs.values() for token in values}),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Catalog metadata/images are a static transductive snapshot, not verified at each event time.",
                        "Marketing descriptions may be noisy; no reviews, ratings or helpful-vote fields are included."],
        "artifacts_sha256": {name: sha256(args.output / name) for name in ("features.npz", "specs.json")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
