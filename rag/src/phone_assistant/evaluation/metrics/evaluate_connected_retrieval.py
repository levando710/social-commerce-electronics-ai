"""Paired first-query IR with metadata, real CLIP images and explicit single-target qrels.

Encode query phases separately to keep only one pretrained encoder in memory.
Do not interpret unjudged products as verified irrelevant or these scores as RAG quality.
"""
from __future__ import annotations
from phone_assistant.common.project_layout import source_path

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from phone_assistant.retrieval.search.bm25 import build_index, score
from phone_assistant.common.catalog_schema import catalog_text
from phone_assistant.retrieval.search.dense_search import reciprocal_rank_fusion, text_chunks
from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, load_features, read_jsonl, sha256


def ordered(scores, item_ids, mask=None):
    return sorted(((item, float(value)) for index, (item, value) in enumerate(zip(item_ids, scores))
                   if mask is None or mask[index]), key=lambda row: (-row[1], row[0]))


def query_chunks(query, channel):
    return text_chunks({"text": query}, words=32 if channel == "clip" else 80, overlap=8 if channel == "clip" else 16)


def encode_queries(queries, features_manifest, query_source, work, channel, batch_size, threads):
    import torch
    from phone_assistant.retrieval.search.dense_search import load_encoder
    torch.set_num_threads(threads)
    config = features_manifest["image_encoder" if channel == "clip" else "text_encoder"]
    contract = {"query_source_sha256": sha256(query_source), "encoder": config,
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "sentence-transformers", "transformers")},
                "source_sha256": sha256(Path(__file__)), "channel": channel,
                "pooling": "normalized mean of normalized query chunks", "batch_size": batch_size,
                "chunk_words": 32 if channel == "clip" else 80,
                "chunk_overlap": 8 if channel == "clip" else 16}
    output = work / f"{channel}.npz"
    manifest_path = work / f"{channel}_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["contract"] != contract or saved["sha256"] != sha256(output):
            raise ValueError("Query vector contract/checksum changed; use a new query work directory")
        print(f"{channel} query vectors already verified", flush=True)
        return
    encoder = load_encoder(config["model"], revision=config["revision"])
    chunks, owners = [], []
    for index, row in enumerate(queries):
        values = query_chunks(row["query"], channel)
        chunks.extend(values)
        owners.extend([index] * len(values))
    started = time.perf_counter()
    vectors = encoder.encode(chunks, batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)
    result = np.zeros((len(queries), vectors.shape[1]), dtype=np.float32)
    np.add.at(result, owners, vectors)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    work.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, dialogue_ids=np.asarray([q["dialogue_id"] for q in queries]), vectors=result)
    temporary.replace(output)
    manifest = {"contract": contract, "sha256": sha256(output), "queries": len(queries),
                "chunks": len(chunks), "seconds": time.perf_counter() - started}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


def load_queries(queries, source, work, channel, encoder):
    manifest = json.loads((work / f"{channel}_manifest.json").read_text(encoding="utf-8"))
    contract = manifest["contract"]
    if contract["query_source_sha256"] != sha256(source) or contract["encoder"] != encoder or contract["source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Query vectors do not match query source, encoder or code")
    path = work / f"{channel}.npz"
    if manifest["sha256"] != sha256(path):
        raise ValueError("Query vector checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        if list(stored["dialogue_ids"]) != [q["dialogue_id"] for q in queries]:
            raise ValueError("Query vector ordering mismatch")
        vectors = stored["vectors"].copy()
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite query vectors")
    return vectors


def visual_tags(record, image_dir):
    from PIL import Image, ImageStat
    if record["status"] != "ok":
        return ""
    path = image_dir / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Image for coarse-tag baseline changed")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        try:
            small = rgb.resize((16, 16))
            try:
                means = ImageStat.Stat(small).mean
            finally:
                small.close()
        finally:
            rgb.close()
    brightness = sum(means) / 3
    light = "dark" if brightness < 85 else "bright" if brightness > 170 else "medium brightness"
    color = "white" if min(means) > 200 else "black" if max(means) < 50 else (
        ("red", "green", "blue")[int(np.argmax(means))] if max(means) > 1.15 * (min(means) + 1) else "gray")
    aspect = "landscape" if record["width"] > 1.2 * record["height"] else "portrait" if record["height"] > 1.2 * record["width"] else "square"
    return " ".join((light, color, aspect))


def metrics(rows, n_items, k_values=(5, 10, 20)):
    users = defaultdict(list)
    for row in rows:
        users[row["user_id"]].append(row)
    result = {"users": len(users), "queries": len(rows)}
    for k in k_values:
        for name in ("precision", "recall", "ndcg", "mrr"):
            averages = []
            for group in users.values():
                values = []
                for row in group:
                    rank = row["rank"]
                    value = 0.0 if rank > k else (1 / k if name == "precision" else
                        1.0 if name == "recall" else 1 / np.log2(rank + 1) if name == "ndcg" else 1 / rank)
                    values.append(value)
                averages.append(float(np.mean(values)))
            result[f"{name}@{k}"] = float(np.mean(averages)) if averages else None
        result[f"coverage@{k}"] = len({item for row in rows for item in row["top_items"][:k]}) / n_items
    return result


def evaluate(data, features, manifest, queries, query_source, work, image_dir, output):
    if output.exists():
        raise FileExistsError("Use a new evaluation output; do not tune on this test")
    products = {p["parent_asin"]: p for p in read_jsonl(data.directory / "products.jsonl")}
    documents = [{"doc_id": item, "text": catalog_text(products[item])} for item in data.item_ids]
    records = json.loads((Path(manifest["_directory"]) / "image_sources.json").read_text(encoding="utf-8"))
    by_id = {r["item_id"]: r for r in records}
    tags = {item: visual_tags(by_id[item], image_dir) for item in data.item_ids}
    text_index = build_index(documents)
    tag_index = build_index([{**d, "text": d["text"] + "\n" + tags[d["doc_id"]]} for d in documents])
    text_queries = load_queries(queries, query_source, work, "text", manifest["text_encoder"])
    clip_queries = load_queries(queries, query_source, work, "clip", manifest["image_encoder"])
    labels = {r["user_id"]: r for r in read_jsonl(data.directory / "evaluator_labels.jsonl")}
    methods = ("bm25", "bm25_visual_tags", "dense_text", "dense_clip", "hybrid")
    rows = {split: {method: [] for method in methods} for split in ("validation", "test")}
    started = time.perf_counter()
    for index, query in enumerate(queries):
        split = query["split"]
        if split not in rows:
            continue
        target = labels[query["user_id"]]["item_id"]
        if labels[query["user_id"]]["cohort"] != split:
            raise ValueError("Query and target cohort mismatch")
        bm25 = score(text_index, query["query"])
        raw_tags = score(tag_index, query["query"])
        text = ordered(features["text"] @ text_queries[index], data.item_ids)
        clip = ordered(features["image"] @ clip_queries[index], data.item_ids, features["image_mask"])
        hybrid = reciprocal_rank_fusion({"bm25": bm25, "text": text, "clip": clip},
            weights={"bm25": 1., "text": 1., "clip": .5}, rank_constant=60, top_k=data.n_items)
        rankings = {"bm25": bm25, "bm25_visual_tags": raw_tags, "dense_text": text, "dense_clip": clip, "hybrid": hybrid}
        for method, ranking in rankings.items():
            ordered_ids = [r[0] for r in ranking]
            # Missing/zero-score sparse targets count as not retrieved.
            rank = ordered_ids.index(target) + 1 if target in ordered_ids else data.n_items + 1
            rows[split][method].append({"dialogue_id": query["dialogue_id"], "user_id": query["user_id"],
                "target_item": target, "rank": rank, "top_items": ordered_ids[:20]})
    result = {"split_manifest_sha256": sha256(data.directory / "manifest.json"),
        "feature_manifest_sha256": sha256(Path(manifest["_directory"]) / "manifest.json"),
        "source_sha256": {name: sha256(source_path(name)) for name in
                          ("evaluate_connected_retrieval.py", "catalog_schema.py", "bm25.py", "dense_search.py", "recsys_data.py")},
        "query_manifests_sha256": {channel: sha256(work / f"{channel}_manifest.json") for channel in ("text", "clip")},
        "config": {"rrf_weights": {"bm25": 1, "text": 1, "clip": .5}, "rrf_constant": 60,
                   "ranking": "full catalog, including previously seen items for evidence retrieval",
                   "qrels": "one released target per simulated first query; unjudged items are not verified negatives"},
        "metrics": {split: {method: metrics(values, data.n_items) for method, values in rows[split].items()} for split in rows},
        "rows": rows, "scoring_seconds_excluding_initialization": time.perf_counter() - started,
        "limitations": ["First user message from LLM simulation, not observed human conversation.",
            "Precision/recall/NDCG/MRR use incomplete single-target qrels; not human-judged overall relevance.",
            "Coverage is unique top-K item share of the whole catalog.",
            "Raw Pillow tags are only a baseline; learned CLIP vectors supply the proposed visual representation.",
            "No query understanding/fact-rerank or response generation evaluation in this isolated IR experiment.",
            "English queries; no Vietnamese cross-modal result claimed."]}
    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "visual_tags.json").write_text(json.dumps(tags, indent=2), encoding="utf-8")
    (output / "source").mkdir()
    for name in result["source_sha256"]:
        (output / "source" / name).write_bytes((source_path(name)).read_bytes())
    lines = ["# IR trên câu hỏi mô phỏng gắn đúng người dùng và mục tiêu", "",
        "Test có một nhãn đích/query; các item khác chưa được đánh giá đầy đủ. Không diễn giải P@K như precision được con người xác minh.", "",
        "| Phương án | User | Query | P@5 theo qrel | Recall@10 | NDCG@10 | MRR@10 | Coverage@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, m in result["metrics"]["test"].items():
        lines.append(f"| {method} | {m['users']} | {m['queries']} | {m['precision@5']:.4f} | {m['recall@10']:.4f} | {m['ndcg@10']:.4f} | {m['mrr@10']:.4f} | {m['coverage@10']:.4f} |")
    lines += ["", "## Giới hạn", ""] + [f"- {item}" for item in result["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--features", type=Path, default=ROOT / 'artifacts/cache/connected_features_v1')
    parser.add_argument("--work", type=Path, default=ROOT / 'artifacts/cache/connected_query_vectors_v1')
    parser.add_argument("--image-dir", type=Path, default=ROOT / 'artifacts/cache/connected_feature_work_v1/images')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_ir_v1')
    parser.add_argument("--phase", choices=("text-queries", "clip-queries", "evaluate"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) <= 0:
        parser.error("Invalid processing limits")
    data = InteractionData(args.split)
    features, manifest = load_features(args.features, data)
    source = args.split / "agent_queries.jsonl"
    queries = read_jsonl(source)
    if args.phase != "evaluate":
        encode_queries(queries, manifest, source, args.work, "text" if args.phase == "text-queries" else "clip", args.batch_size, args.threads)
    else:
        manifest["_directory"] = str(args.features)
        evaluate(data, features, manifest, queries, source, args.work, args.image_dir, args.output)


if __name__ == "__main__":
    main()
