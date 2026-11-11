"""Resumable, one-encoder-at-a-time features for the frozen connected catalog.

Run phases images -> text -> clip -> finalize. Each vector is checkpointed per
item and input-hashed. No reviews/ratings/generated descriptions enter text.
"""
from __future__ import annotations
from phone_assistant.common.project_layout import source_path

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import time
import threading
from urllib.parse import urlsplit
import warnings

import numpy as np

from phone_assistant.common.catalog_schema import catalog_text, make_hierarchy
from phone_assistant.retrieval.search.dense_search import text_chunks
from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, read_jsonl, sha256

HTTP = threading.local()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def image_urls(product):
    entries = sorted(product.get("images") or [], key=lambda r: r.get("variant") != "MAIN")
    urls = []
    for entry in entries:
        for key in ("hi_res", "large", "thumb"):
            url = entry.get(key)
            if url and url not in urls:
                parsed = urlsplit(url)
                host = parsed.hostname or ""
                if parsed.scheme == "https" and (host.endswith(".amazon.com") or host.endswith(".media-amazon.com")):
                    urls.append(url)
        if urls:
            break  # One view per item; fallbacks are resolutions of the same view.
    return urls


def fetch_image(product, directory, retry_failed=False):
    import requests
    from PIL import Image
    if not hasattr(HTTP, "session"):
        HTTP.session = requests.Session()
    item = product["parent_asin"]
    record_path = directory / f"{item}.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] == "ok":
            if sha256(directory / record["file"]) != record["sha256"]:
                raise ValueError(f"Cached image changed: {item}")
            return record
        if not retry_failed:
            return record
    errors = []
    for url in image_urls(product):
        try:
            content = bytearray()
            with HTTP.session.get(url, stream=True, timeout=(8, 20)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(65536):
                    content.extend(chunk)
                    if len(content) > 5_000_000:
                        raise ValueError("Image exceeds 5 MB limit")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    if image.width * image.height > 20_000_000:
                        raise ValueError("Image exceeds 20 million pixels")
                    width, height, image_format = image.width, image.height, image.format
                    image.verify()
            path = directory / f"{item}.img"
            temporary = path.with_suffix(".img.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
            record = {"item_id": item, "status": "ok", "url": url, "file": path.name,
                      "sha256": sha256(path), "bytes": len(content), "width": width, "height": height,
                      "format": image_format}
            atomic_json(record_path, record)
            return record
        except (requests.RequestException, OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            errors.append(type(exc).__name__ + ": " + str(exc)[:180])
    record = {"item_id": item, "status": "failed", "errors": errors or ["No supported image URL"]}
    atomic_json(record_path, record)
    return record


def vector_file(directory, item):
    return directory / f"{item}.npz"


def saved_vector(directory, item, input_hash, dimensions):
    path = vector_file(directory, item)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if str(stored["input_sha256"]) != input_hash:
            raise ValueError(f"Vector input changed: {item}")
        vector = stored["vector"].copy()
    if vector.shape != (dimensions,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
        raise ValueError(f"Invalid cached vector: {item}")
    return vector


def save_vector(directory, item, input_hash, vector):
    vector = np.asarray(vector, dtype=np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
        raise ValueError(f"Invalid encoded vector: {item}")
    path = vector_file(directory, item)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, vector=vector, input_sha256=np.asarray(input_hash))
    temporary.replace(path)


def text_hash(product):
    return hashlib.sha256(catalog_text(product).encode("utf-8")).hexdigest()


def encode_text(products, work, config, args):
    import torch
    from phone_assistant.retrieval.search.dense_search import load_encoder
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    directory = work / "text"
    directory.mkdir(exist_ok=True)
    pending = [p for p in products if saved_vector(directory, p["parent_asin"], text_hash(p), 384) is None]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("Text vectors already complete", flush=True)
        return
    encoder = load_encoder(config["model"], revision=config["revision"])
    started = time.perf_counter()
    for offset in range(0, len(pending), 8):
        group = pending[offset:offset + 8]
        chunks, lengths = [], []
        for product in group:
            values = text_chunks({"text": catalog_text(product)})
            lengths.append(len(values))
            chunks.extend(values)
        encoded = encoder.encode(chunks, batch_size=args.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        cursor = 0
        for product, length in zip(group, lengths):
            vector = np.mean(encoded[cursor:cursor + length], axis=0)
            cursor += length
            save_vector(directory, product["parent_asin"], text_hash(product), vector)
        completed = offset + len(group)
        atomic_json(work / "text_progress.json", {"completed_this_invocation": completed, "pending_at_start": len(pending),
                    "seconds": time.perf_counter() - started, "batch_size": args.batch_size, "threads": args.threads})
        if completed % 80 == 0 or completed == len(pending):
            print(f"Text {completed}/{len(pending)} pending items, {time.perf_counter()-started:.1f}s", flush=True)


def encode_clip(products, work, config, args):
    import torch
    from PIL import Image
    from phone_assistant.retrieval.search.dense_search import load_encoder
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    directory = work / "clip"
    directory.mkdir(exist_ok=True)
    pending = []
    for product in products:
        item = product["parent_asin"]
        record_path = work / "images" / f"{item}.json"
        if not record_path.exists():
            raise ValueError("Run the complete images phase before CLIP")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] != "ok":
            continue
        if sha256(work / "images" / record["file"]) != record["sha256"]:
            raise ValueError("Image checksum mismatch")
        if saved_vector(directory, item, record["sha256"], 512) is None:
            pending.append(record)
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("CLIP vectors already complete", flush=True)
        return
    encoder = load_encoder(config["model"], revision=config["revision"])
    started = time.perf_counter()
    for offset in range(0, len(pending), args.batch_size):
        group = pending[offset:offset + args.batch_size]
        images = []
        try:
            for record in group:
                with Image.open(work / "images" / record["file"]) as image:
                    images.append(image.convert("RGB"))
            vectors = encoder.encode(images, batch_size=args.batch_size, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
            for record, vector in zip(group, vectors):
                save_vector(directory, record["item_id"], record["sha256"], vector)
        finally:
            for image in images:
                image.close()
        completed = offset + len(group)
        atomic_json(work / "clip_progress.json", {"completed_this_invocation": completed, "pending_at_start": len(pending),
                    "seconds": time.perf_counter() - started, "batch_size": args.batch_size, "threads": args.threads})
        if completed % 100 == 0 or completed == len(pending):
            print(f"CLIP {completed}/{len(pending)} pending items, {time.perf_counter()-started:.1f}s", flush=True)


def finalize(products, data, work, output, contract):
    if output.exists():
        raise FileExistsError("Feature output exists; use a new path")
    text = np.zeros((len(products), 384), dtype=np.float32)
    image = np.zeros((len(products), 512), dtype=np.float32)
    mask = np.zeros(len(products), dtype=bool)
    image_records = []
    for index, product in enumerate(products):
        item = product["parent_asin"]
        vector = saved_vector(work / "text", item, text_hash(product), 384)
        if vector is None:
            raise ValueError("Text phase is incomplete")
        text[index] = vector
        record = json.loads((work / "images" / f"{item}.json").read_text(encoding="utf-8"))
        image_records.append(record)
        if record["status"] == "ok":
            if sha256(work / "images" / record["file"]) != record["sha256"]:
                raise ValueError("Image changed before finalization")
            vector = saved_vector(work / "clip", item, record["sha256"], 512)
            if vector is None:
                raise ValueError("CLIP phase is incomplete")
            image[index], mask[index] = vector, True
    specs, hierarchy = make_hierarchy(products)
    output.mkdir(parents=True)
    np.savez_compressed(output / "features.npz", item_ids=np.asarray(data.item_ids), text=text, image=image, image_mask=mask)
    atomic_json(output / "specs.json", specs)
    atomic_json(output / "spec_hierarchy.json", hierarchy)
    atomic_json(output / "image_sources.json", image_records)
    manifest = {**contract, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "processing_runs": [run for run in json.loads((work / "phase_runs.json").read_text(encoding="utf-8")) if run["phase"] != "finalize"] if (work / "phase_runs.json").exists() else [],
        "text_policy": "catalog_metadata_allowlist_no_reviews_no_ratings", "items": data.n_items,
        "text_dimensions": 384, "image_dimensions": 512, "items_with_images": int(mask.sum()),
        "text_chunks": sum(len(text_chunks({"text": catalog_text(p)})) for p in products),
        "spec_nodes": len(hierarchy["value_to_attribute"]), "spec_attributes": len(hierarchy["attribute_to_group"]),
        "spec_groups": len(set(hierarchy["attribute_to_group"].values())),
        "text_encoder": contract["encoders"]["text"], "image_encoder": contract["encoders"]["clip"],
        "pooling": "normalized mean of all normalized text chunks; one actual product image per item",
        "limitations": ["Static catalog text/images, not time-versioned per interaction.",
            "Images are catalog photos, not verified used-device condition photographs.",
            "Spec hierarchy encodes only explicit allowlisted attributes; no missing facts inferred.",
            "Failed image downloads are zero vectors with image_mask=False, never fabricated embeddings."],
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"items": data.n_items, "images": int(mask.sum()), "features": str(output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--work", type=Path, default=ROOT / 'artifacts/cache/connected_feature_work_v1')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/cache/connected_features_v1')
    parser.add_argument("--encoder-manifest", type=Path, default=ROOT / 'artifacts/runtime/catalog/dense_index/manifest.json')
    parser.add_argument("--phase", choices=("images", "text", "clip", "finalize"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Bound this invocation; resume by rerunning the phase without a limit")
    parser.add_argument("--retry-failed-images", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.workers) < 1 or args.limit < 0:
        parser.error("Invalid processing limits")
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Features are frozen; do not mutate their work cache")
    data = InteractionData(args.split)
    source = args.split / "products.jsonl"
    by_id = {p["parent_asin"]: p for p in read_jsonl(source)}
    products = [by_id[item] for item in data.item_ids]
    channels = json.loads(args.encoder_manifest.read_text(encoding="utf-8"))["channels"]
    contract = {"schema_version": 2, "split_manifest_sha256": sha256(args.split / "manifest.json"),
        "source_metadata_sha256": sha256(source), "encoders": {key: {field: channels[key][field] for field in ("model", "revision")} for key in ("text", "clip")},
        "source_sha256": {name: sha256(source_path(name)) for name in ("build_connected_features.py", "catalog_schema.py", "dense_search.py")},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "sentence-transformers", "transformers", "Pillow")}}
    args.work.mkdir(parents=True, exist_ok=True)
    path = args.work / "contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Work cache contract changed; use a new work/output or restore the original source")
    else:
        atomic_json(path, contract)
    log_path = args.work / "phase_runs.json"
    logs = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    logs.append({"phase": args.phase, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "batch_size": args.batch_size, "threads": args.threads, "workers": args.workers,
                 "limit": args.limit, "seed": 42, "device": "cpu"})
    atomic_json(log_path, logs)
    if args.phase == "images":
        directory = args.work / "images"
        directory.mkdir(exist_ok=True)
        selected = products[:args.limit] if args.limit else products
        ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(lambda p: fetch_image(p, directory, args.retry_failed_images), selected)
            for count, result in enumerate(results, 1):
                ok += result["status"] == "ok"
                if count % 200 == 0 or count == len(selected):
                    print(f"Images {count}/{len(selected)}, decoded={ok}", flush=True)
        atomic_json(args.work / "images_progress.json", {"checked": len(selected), "ok": ok})
    elif args.phase == "text":
        encode_text(products, args.work, contract["encoders"]["text"], args)
    elif args.phase == "clip":
        encode_clip(products, args.work, contract["encoders"]["clip"], args)
    else:
        finalize(products, data, args.work, args.output, contract)


if __name__ == "__main__":
    main()
