"""Controlled structured-RAM ablation: immutable raw catalog/graph/images, local encoder only."""
from phone_assistant.common.project_layout import source_path
import copy
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from phone_assistant.common.catalog_schema import catalog_text, make_hierarchy
from phone_assistant.retrieval.search.dense_search import load_encoder, text_chunks
from phone_assistant.conversation.handset_dialogue import handset_facts
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, read_jsonl, sha256


def quarantine_view(products, documents):
    view, affected = [], []
    docs = {d["doc_id"]: d for d in documents}
    for product in products:
        item = product["parent_asin"]
        transformed = copy.deepcopy(product)
        if "ram_capacity_requires_review" in handset_facts(docs[item], product)["fact_warnings"]:
            removed = {key: transformed["details"].pop(key) for key in ("RAM", "RAM Memory Installed Size", "Memory RAM") if key in transformed["details"]}
            if removed:
                affected.append({"item_id": item, "removed_structured_fields": removed})
        view.append(transformed)
    return view, affected


def main():
    split = ROOT / 'data/external/benchmarks/handsets_only_v2'
    original_dir = ROOT / 'artifacts/cache/handset_features_v1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/cache/handset_features_ram_quarantine_v2')
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError("Keep feature experiments; use a new version")
    data = InteractionData(split)
    original, manifest = load_features(original_dir, data)
    products = read_jsonl(split / "products.jsonl")
    view, affected = quarantine_view(products, read_jsonl(split / "documents.jsonl"))
    lookup = {p["parent_asin"]: p for p in view}
    changed = {a["item_id"] for a in affected}
    selected = [(i, lookup[item]) for i, item in enumerate(data.item_ids) if item in changed]
    text = original["text"].copy()
    import torch
    torch.set_num_threads(2)
    config = manifest["text_encoder"]
    encoder = load_encoder(config["model"], revision=config["revision"])
    started = time.perf_counter()
    for start in range(0, len(selected), 4):
        batch = selected[start:start + 4]
        chunks, owners = [], []
        for local, (_, product) in enumerate(batch):
            values = text_chunks({"text": catalog_text(product)})
            chunks.extend(values); owners.extend([local] * len(values))
        encoded = encoder.encode(chunks, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.zeros((len(batch), text.shape[1]), dtype=np.float32)
        np.add.at(vectors, owners, encoded)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        for (index, _), vector in zip(batch, vectors):
            text[index] = vector
    unchanged = np.asarray([item not in changed for item in data.item_ids])
    if not np.array_equal(text[unchanged], original["text"][unchanged]):
        raise ValueError("Unchanged item vectors changed")
    specs, hierarchy = make_hierarchy(view)
    output.mkdir(parents=True)
    np.savez_compressed(output / "features.npz", item_ids=np.asarray(data.item_ids), text=text,
        image=original["image"], image_mask=original["image_mask"])
    for name, value in (("specs.json", specs), ("spec_hierarchy.json", hierarchy), ("quarantine.json", affected)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    (output / "metadata_view.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in view), encoding="utf-8")
    (output / "image_sources.json").write_bytes((original_dir / "image_sources.json").read_bytes())
    result = {**manifest, "created_at_utc": datetime.now(timezone.utc).isoformat(), "processing_runs": [],
        "text_chunks": sum(len(text_chunks({"text": catalog_text(p)})) for p in view),
        "source_feature_manifest_sha256": sha256(original_dir / "manifest.json"),
        "source_metadata_sha256": sha256(output / "metadata_view.jsonl"),
        "source_sha256": {name: sha256(source_path(name)) for name in
            ("build_quarantined_features.py", "handset_dialogue.py", "catalog_schema.py", "dense_search.py")},
        "transform": "remove suspicious structured RAM only; preserve raw source, item IDs, interactions, images and nonaffected text vectors",
        "quarantined_items": len(affected), "text_reencoded_items": len(selected), "encoding_seconds": time.perf_counter() - started,
        "spec_nodes": len(hierarchy["value_to_attribute"]), "spec_attributes": len(hierarchy["attribute_to_group"]),
        "spec_groups": len(set(hierarchy["attribute_to_group"].values())),
        "artifacts_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    arrays, _ = load_features(output, data)
    if not np.array_equal(arrays["image"], original["image"]) or not np.array_equal(arrays["image_mask"], original["image_mask"]):
        raise ValueError("Image features changed in the RAM-only ablation")
    print(json.dumps({"changed_items": len(affected), "spec_nodes": result["spec_nodes"], "encoding_seconds": result["encoding_seconds"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
