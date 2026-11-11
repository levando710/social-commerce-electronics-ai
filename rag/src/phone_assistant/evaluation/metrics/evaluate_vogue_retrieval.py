"""Fixed, no-training retrieval pilot on first seeker turns from published human VOGUE dialogues."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from phone_assistant.retrieval.search.bm25 import build_index, score
from phone_assistant.retrieval.search.dense_search import reciprocal_rank_fusion, text_chunks
from phone_assistant.recommendation.recsys_data import ROOT, sha256

CATALOG_FIRST_ITEM = 1
CATALOG_LAST_ITEM = 36
CPU_THREADS = 2
ENCODER_BATCH_SIZE = 4
RRF_CONSTANT = 60
RRF_WEIGHTS = (("bm25", 1.0), ("text", 1.0), ("clip", 0.5))
METRIC_CUTOFFS = (5, 10, 20)
CLIP_CHUNK_WORDS = 32
CLIP_CHUNK_OVERLAP_WORDS = 8
MINIMUM_VECTOR_NORM = 1e-12
SEEKER_ROLE = "Seeker"
SOURCE = ROOT / 'data/external/vogue'
OUTPUT = ROOT / 'artifacts/evaluation/results_vogue_ir_v1'
TEXT_FIELDS = ("product_name", "product_brand", "categories", "product_description", "about_product", "product_detail")


def observations(transcript):
    for turn in transcript["conversation_content"]:
        content = turn["content"]
        if content["role"] == SEEKER_ROLE:
            text = " ".join(content["utterances"]).strip()
            if text:
                return text
    raise ValueError("No first seeker query")


def inputs():
    documents, images, queries, labels, sources = [], [], [], [], {}
    manifest_path = ROOT / 'data/external/download_manifest.json'
    downloaded = json.loads(manifest_path.read_text(encoding="utf-8"))["vogue"]
    expected = {r["path"]: r["sha256"] for r in downloaded["files"]}
    def verified(path):
        relative = path.relative_to(SOURCE).as_posix()
        digest = sha256(path)
        if expected.get(relative) != digest:
            raise ValueError(f"Published source changed: {relative}")
        sources[relative] = digest
    for item in range(CATALOG_FIRST_ITEM, CATALOG_LAST_ITEM + 1):
        metadata = SOURCE / f"data/metadata/item_{item}.json"
        image = SOURCE / f"data/metadata/item_{item}.png"
        verified(metadata); verified(image)
        row = json.loads(metadata.read_text(encoding="utf-8"))
        allowed = {k: row.get(k) for k in TEXT_FIELDS}
        documents.append({"doc_id": str(item), "text": json.dumps(allowed, ensure_ascii=False)})
        images.append(image)
    catalog = {r["doc_id"] for r in documents}
    for path in sorted((SOURCE / "data/conversation_trials/transcripts").glob("*.json")):
        verified(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        targets = sorted({str(item) for item in row["gt_items"]})
        if not targets or not set(targets) <= catalog:
            raise ValueError("Missing/unresolved published selection labels")
        queries.append({"dialogue_id": str(row["conversation_id"]), "query": observations(row),
            "session_code": row["session_code"]})
        labels.append(targets)
    if len({q["dialogue_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate conversations")
    contract = {"source_revision": downloaded["revision"], "source_files_sha256": sources,
        "source_sha256": sha256(Path(__file__)), "query_policy": "First nonempty seeker turn only; no assistant proposals or later turns.",
        "metadata_policy": "Title/brand/categories/description/attributes only; no reviews, ratings, final labels or transcript text in item features."}
    return documents, images, queries, labels, contract


def metric_rows(rankings, labels, n_items):
    result = {}
    for cutoff in METRIC_CUTOFFS:
        values = defaultdict(list)
        for ranking, targets in zip(rankings, labels):
            hits = [float(item in targets) for item in ranking[:cutoff]]
            positions = [i + 1 for i, hit in enumerate(hits) if hit]
            values["precision"].append(sum(hits) / cutoff)
            values["recall"].append(sum(hits) / len(targets))
            ideal = sum(1 / np.log2(i + 2) for i in range(min(cutoff, len(targets))))
            values["ndcg"].append(sum(hit / np.log2(i + 2) for i, hit in enumerate(hits)) / ideal)
            values["mrr"].append(1 / positions[0] if positions else 0.)
        result.update({f"{key}@{cutoff}": float(np.mean(v)) for key, v in values.items()})
        result[f"coverage@{cutoff}"] = len({i for row in rankings for i in row[:cutoff]}) / n_items
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("text", "clip", "evaluate"), required=True)
    parser.add_argument("--work", type=Path, default=ROOT / 'artifacts/cache/vogue_ir_vectors_v1')
    args = parser.parse_args()
    documents, images, queries, labels, contract = inputs()
    args.work.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'artifacts/cache/handset_features_v1/manifest.json').read_text(encoding="utf-8"))
    if args.phase != "evaluate":
        import torch
        from phone_assistant.retrieval.search.dense_search import load_encoder
        torch.set_num_threads(CPU_THREADS)
        channel = "image_encoder" if args.phase == "clip" else "text_encoder"
        output = args.work / (args.phase + ".npz")
        if output.exists():
            raise FileExistsError("Keep existing vectors; use another work version")
        encoder = load_encoder(config[channel]["model"], revision=config[channel]["revision"])
        def encode_texts(texts):
            rows = []
            for text in texts:
                values = text_chunks({"text": text}, words=CLIP_CHUNK_WORDS, overlap=CLIP_CHUNK_OVERLAP_WORDS) if args.phase == "clip" else text_chunks({"text": text})
                vectors = encoder.encode(values, batch_size=ENCODER_BATCH_SIZE, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
                vector = np.mean(vectors, axis=0); vector /= max(float(np.linalg.norm(vector)), MINIMUM_VECTOR_NORM)
                rows.append(vector)
            return np.asarray(rows, dtype=np.float32)
        if args.phase == "text":
            items = encode_texts([d["text"] for d in documents])
        else:
            from PIL import Image
            opened = []
            try:
                for path in images:
                    with Image.open(path) as image:
                        opened.append(image.convert("RGB"))
                items = encoder.encode(opened, batch_size=ENCODER_BATCH_SIZE, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
            finally:
                for image in opened: image.close()
        query_vectors = encode_texts([q["query"] for q in queries])
        np.savez_compressed(output, items=items, queries=query_vectors)
        (args.work / (args.phase + ".json")).write_text(json.dumps({"contract": contract,
            "encoder": config[channel], "sha256": sha256(output)}, indent=2), encoding="utf-8")
        print(f"Encoded VOGUE {args.phase}: {len(items)} items, {len(queries)} first seeker queries", flush=True)
        return
    if OUTPUT.exists():
        raise FileExistsError("Keep recorded pilot; new version needed")
    channels = {}
    for name in ("text", "clip"):
        metadata = json.loads((args.work / (name + ".json")).read_text(encoding="utf-8"))
        if metadata["contract"] != contract or sha256(args.work / (name + ".npz")) != metadata["sha256"]:
            raise ValueError("Cached vectors or source changed")
        with np.load(args.work / (name + ".npz"), allow_pickle=False) as arrays:
            channels[name] = {key: arrays[key].copy() for key in arrays.files}
    ids = [d["doc_id"] for d in documents]
    index = build_index(documents)
    rankings = {key: [] for key in ("bm25", "dense_text", "clip", "hybrid")}
    for offset, query in enumerate(queries):
        sparse = score(index, query["query"])
        dense = {name: sorted(zip(ids, values["items"] @ values["queries"][offset]), key=lambda r: (-r[1], r[0]))
            for name, values in channels.items()}
        hybrid = reciprocal_rank_fusion({"bm25": sparse, **dense}, weights=dict(RRF_WEIGHTS),
            rank_constant=RRF_CONSTANT, top_k=len(ids))
        for name, rows in (("bm25", sparse), ("dense_text", dense["text"]), ("clip", dense["clip"]), ("hybrid", hybrid)):
            rankings[name].append([r[0] for r in rows])
    result = {**contract, "dialogues": len(queries), "catalog_items": len(ids), "api_calls": 0,
        "scope": "Fixed no-training pilot over all published first-seeker queries; not a held-out trained-model or interactive-dialogue benchmark.",
        "metrics": {name: metric_rows(rows, labels, len(ids)) for name, rows in rankings.items()},
        "rows": [{**query, "published_selections": target, "rankings": {name: rows[i] for name, rows in rankings.items()}}
            for i, (query, target) in enumerate(zip(queries, labels))],
        "limitations": ["Selections at dialogue end may reflect preferences not expressed in the first turn.",
            "Multiple dialogues share participants; no independent-user significance or unseen-participant claim.",
            "Incomplete final-selection qrels; unselected items are not verified irrelevant.",
            "No response generation or visual query input measured; CLIP compares query text to catalog pixels."]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "experiment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
