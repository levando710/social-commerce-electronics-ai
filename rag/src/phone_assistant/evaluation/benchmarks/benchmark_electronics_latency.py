"""Electronics CPU latency by phase; run after training, with no API calls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from phone_assistant.retrieval.search.bm25 import score
from phone_assistant.retrieval.search.dense_search import load_encoder, reciprocal_rank_fusion
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import load_queries, ordered, query_chunks
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, read_jsonl, sha256

CPU_THREADS = 2
QUERY_LIMIT = 12
REPETITIONS = 5
WARMUP_REPETITIONS = 1
MILLISECONDS_PER_SECOND = 1000
TAIL_QUANTILE = 0.95
CANDIDATE_LIMIT = 80
EVIDENCE_LIMIT = 20
FIXED_BENCHMARK_SEED = 42
RRF_WEIGHTS = {"bm25": 1.0, "text": 1.0, "clip": 0.5}
OUTPUT = ROOT / 'artifacts/evaluation/results_electronics_latency_v1'


def distribution(values):
    return {"samples": len(values), "mean_ms": float(np.mean(values)), "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, TAIL_QUANTILE))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "text", "clip", "anchored"), required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{args.phase}.json"
    if path.exists():
        raise FileExistsError("Keep recorded timing samples")
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=CPU_THREADS)
    split = ROOT / 'data/external/benchmarks/connected_electronics_v2'
    data = InteractionData(split)
    features, manifest = load_features(ROOT / 'artifacts/cache/electronics_features_v1', data)
    source = split / "agent_queries.jsonl"; queries = read_jsonl(source)
    selected = sorted((i for i, row in enumerate(queries) if row["split"] == "validation"), key=lambda i: queries[i]["dialogue_id"])[:QUERY_LIMIT]
    if len(selected) != QUERY_LIMIT:
        raise ValueError("Need the predeclared number of validation queries")
    result = {"phase": args.phase, "source_sha256": sha256(Path(__file__)),
        "split_manifest_sha256": sha256(split / "manifest.json"), "feature_manifest_sha256": sha256(ROOT / 'artifacts/cache/electronics_features_v1/manifest.json'),
        "query_source_sha256": sha256(source), "query_ids": [queries[i]["dialogue_id"] for i in selected],
        "queries": len(selected), "catalog_items": data.n_items, "cpu_threads": CPU_THREADS, "api_calls": 0,
        "scope": "Local phase microbenchmark on validation queries; no web load, generation or concurrent-training comparison."}
    timings = []
    if args.phase in ("text", "clip"):
        import torch
        torch.set_num_threads(CPU_THREADS)
        config = manifest["text_encoder" if args.phase == "text" else "image_encoder"]
        started = time.perf_counter(); encoder = load_encoder(config["model"], revision=config["revision"])
        result["model_load_seconds"] = time.perf_counter() - started
        encoder.encode(query_chunks(queries[selected[0]]["query"], args.phase), normalize_embeddings=True, show_progress_bar=False)
        for index in selected:
            started = time.perf_counter()
            vectors = encoder.encode(query_chunks(queries[index]["query"], args.phase), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            vector = vectors.mean(axis=0); vector /= max(float(np.linalg.norm(vector)), np.finfo(np.float32).eps)
            timings.append((time.perf_counter() - started) * MILLISECONDS_PER_SECOND)
        result.update(timing=distribution(timings), raw_milliseconds=timings,
            encoder={key: config[key] for key in ("model", "revision")}, exclusions="Encoder loading, search, guard, transport and generation; one warmup query omitted.")
    else:
        vectors = load_queries(queries, source, ROOT / 'artifacts/cache/electronics_query_vectors_v1', "text", manifest["text_encoder"])
        if args.phase == "search":
            image_queries = load_queries(queries, source, ROOT / 'artifacts/cache/electronics_query_vectors_v1', "clip", manifest["image_encoder"])
            bm25_index = json.loads((ROOT / 'artifacts/cache/electronics_runtime_v1/bm25_index.json').read_text())
            def run(query_index):
                sparse = score(bm25_index, queries[query_index]["query"])
                text = ordered(features["text"] @ vectors[query_index], data.item_ids)
                image = ordered(features["image"] @ image_queries[query_index], data.item_ids, features["image_mask"])
                hits = reciprocal_rank_fusion({"bm25": sparse, "text": text, "clip": image}, weights=RRF_WEIGHTS, top_k=CANDIDATE_LIMIT)
                return hits[:EVIDENCE_LIMIT]
            result["exclusions"] = "Query encoding, planner/fact guard, model loading, transport and generation; full-catalog ranks and RRF included."
        else:
            from phone_assistant.recommendation.anchored_joint import normalized_rows
            from phone_assistant.training.train_anchored_joint import DEFAULT_MODELS, run_directory
            analysis_path = ROOT / 'artifacts/evaluation/results_electronics_anchored_analysis_v1/analysis.json'
            analysis = json.loads(analysis_path.read_text()); alpha = analysis["selected_alpha"]
            directory = run_directory(DEFAULT_MODELS, alpha, "residual_joint", FIXED_BENCHMARK_SEED)
            training = json.loads((directory / "training.json").read_text())
            if sha256(directory / "embeddings.npz") != training["artifacts_sha256"]["embeddings.npz"]:
                raise ValueError("Selected embeddings changed")
            with np.load(directory / "embeddings.npz", allow_pickle=False) as stored:
                weight, items = stored["query_weight"].copy(), normalized_rows(stored["items"])
            text_features = normalized_rows(features["text"])
            def run(query_index):
                query = normalized_rows(vectors[query_index:query_index + 1])
                residual_query = normalized_rows(query @ weight.T)
                scores = (query @ text_features.T + alpha * residual_query @ items.T)[0]
                return ordered(scores, data.item_ids)[:EVIDENCE_LIMIT]
            result.update(selected_alpha=alpha, embeddings_sha256=sha256(directory / "embeddings.npz"),
                analysis_sha256=sha256(analysis_path),
                exclusions="Query encoding, graph propagation/training, profile refresh, guard, transport and generation; exported item vectors cached.")
        for repetition in range(REPETITIONS):
            for index in selected:
                started = time.perf_counter(); run(index)
                elapsed = (time.perf_counter() - started) * MILLISECONDS_PER_SECOND
                if repetition >= WARMUP_REPETITIONS:
                    timings.append(elapsed)
        result.update(timing=distribution(timings), raw_milliseconds=timings, warmup_repetitions=WARMUP_REPETITIONS)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / f"{args.phase}_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"phase": args.phase, "timing": result["timing"], "model_load_seconds": result.get("model_load_seconds")}, indent=2))


if __name__ == "__main__":
    main()
