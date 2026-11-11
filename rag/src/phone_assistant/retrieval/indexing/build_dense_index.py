"""Build real multilingual text and CLIP image vectors without calling an LLM API."""
from __future__ import annotations
from phone_assistant.common.project_layout import asset_path

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from phone_assistant.retrieval.search.dense_search import CLIP_MODEL, ROOT, TEXT_MODEL, corpus_fingerprint, load_encoder, text_chunks


def resolved_revision(model, requested, model_name):
    for module in model:
        for name in ("auto_model", "model"):
            config = getattr(getattr(module, name, None), "config", None)
            commit = getattr(config, "_commit_hash", None)
            if commit:
                return commit
    # New Sentence Transformers modules need not expose .auto_model/.model.
    reference = ROOT / 'artifacts/cache/huggingface/hub' / ("models--" + model_name.replace("/", "--")) / "refs" / requested
    if reference.is_file():
        commit = reference.read_text(encoding="utf-8").strip()
        if len(commit) == 40 and all(char in "0123456789abcdef" for char in commit):
            return commit
    return requested


def save_channel(output, channel, vectors, doc_ids, **extras):
    import numpy as np

    path = output / f"{channel}.npz"
    temporary = output / f"{channel}.tmp.npz"
    np.savez_compressed(temporary, vectors=np.asarray(vectors, dtype=np.float32),
                        doc_ids=np.asarray(doc_ids, dtype=str), **extras)
    temporary.replace(path)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog/documents.jsonl')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/runtime/catalog/dense_index')
    parser.add_argument("--mode", choices=("text", "clip", "both"), default="both")
    parser.add_argument("--text-model", default=TEXT_MODEL)
    parser.add_argument("--clip-model", default=CLIP_MODEL)
    parser.add_argument("--text-revision", default="main")
    parser.add_argument("--clip-revision", default="main")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--images-per-doc", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.images_per_doc <= 0 or args.threads <= 0:
        parser.error("batch-size, images-per-doc and threads must be positive")

    import numpy as np
    import torch
    from PIL import Image

    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    documents = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    ids = [doc["doc_id"] for doc in documents]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Corpus must contain unique, nonempty document IDs.")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Index already exists. Use a new --output directory to preserve provenance.")
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": corpus_fingerprint(documents), "documents": len(documents),
        "device": args.device, "seed": 42, "channels": {},
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "torch", "sentence-transformers", "transformers", "Pillow")},
        "search": "exact cosine; max chunk/image score per document",
    }
    if args.mode in ("text", "both"):
        started = time.perf_counter()
        encoder = load_encoder(args.text_model, revision=args.text_revision,
                               device=args.device, allow_download=args.allow_download)
        texts, text_ids = [], []
        for doc in documents:
            chunks = text_chunks(doc)
            texts.extend(chunks)
            text_ids.extend([doc["doc_id"]] * len(chunks))
        vectors = encoder.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
        manifest["channels"]["text"] = {
            "file": save_channel(args.output, "text", vectors, text_ids),
            "model": args.text_model, "revision": resolved_revision(encoder, args.text_revision, args.text_model),
            "vectors": len(text_ids), "dimensions": int(vectors.shape[1]),
            "chunk_words": 80, "chunk_overlap": 16, "excludes_visual_tags": True,
            "build_seconds": time.perf_counter() - started,
        }
        del encoder, vectors
        print("Text vectors complete", flush=True)
    if args.mode in ("clip", "both"):
        started = time.perf_counter()
        encoder = load_encoder(args.clip_model, revision=args.clip_revision,
                               device=args.device, allow_download=args.allow_download)
        batches, image_ids, paths, hashes, errors = [], [], [], [], []
        pending, pending_ids, pending_paths, pending_hashes = [], [], [], []

        def flush():
            if not pending:
                return
            try:
                batches.append(encoder.encode(pending, batch_size=args.batch_size,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
                image_ids.extend(pending_ids)
                paths.extend(pending_paths)
                hashes.extend(pending_hashes)
            finally:
                for image in pending:
                    image.close()
                pending.clear()
                pending_ids.clear()
                pending_paths.clear()
                pending_hashes.clear()

        for doc in documents:
            for raw_path in (doc.get("product_image_paths") or [])[:args.images_per_doc]:
                path = asset_path(raw_path)
                try:
                    with Image.open(path) as image:
                        converted = image.convert("RGB")
                    image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                except (OSError, ValueError) as exc:
                    errors.append({"doc_id": doc["doc_id"], "path": raw_path, "error": str(exc)})
                    continue
                pending.append(converted)
                pending_ids.append(doc["doc_id"])
                pending_paths.append(str(path))
                pending_hashes.append(image_hash)
                if len(pending) >= args.batch_size:
                    flush()
                    print(f"CLIP images encoded: {len(image_ids)}", flush=True)
        flush()
        if not batches:
            raise ValueError("No valid product images available for CLIP indexing.")
        vectors = np.concatenate(batches)
        manifest["channels"]["clip"] = {
            "file": save_channel(args.output, "clip", vectors, image_ids,
                                 image_paths=np.asarray(paths), image_sha256=np.asarray(hashes)),
            "model": args.clip_model, "revision": resolved_revision(encoder, args.clip_revision, args.clip_model),
            "vectors": len(image_ids), "documents": len(set(image_ids)),
            "dimensions": int(vectors.shape[1]), "images_per_doc": args.images_per_doc,
            "image_errors": errors, "build_seconds": time.perf_counter() - started,
            "query_language_note": "CLIP text encoder is English-centric; report Vietnamese queries separately.",
        }
    temporary = args.output / "manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
