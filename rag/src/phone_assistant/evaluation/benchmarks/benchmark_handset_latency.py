"""CPU latency decomposition, cached-vector search separated from real query encoding; zero APIs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from phone_assistant.retrieval.search.bm25 import score
from phone_assistant.retrieval.search.dense_search import load_encoder, rank_vectors, reciprocal_rank_fusion
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import query_chunks
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def distribution(values):
    return {"samples": len(values), "mean_ms": float(np.mean(values)), "p50_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "text", "clip"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_handset_latency_v2')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.phase}.json"
    if path.exists():
        raise FileExistsError("Keep recorded timings")
    runtime = HandsetRuntime()
    manifest = json.loads((runtime.dense / "manifest.json").read_text())
    queries = [q["query"] for q in read_jsonl(ROOT / 'artifacts/evaluation/handset_benchmark_v1/queries.jsonl')
               if q["cohort"] == "test" and q["kind"] == "retrieval" and q["id"].endswith("_0")]
    result = {"phase": args.phase, "source_sha256": sha256(Path(__file__)), "runtime_manifest_sha256": sha256(runtime.path / "manifest.json"),
        "queries": len(queries), "api_calls": 0, "device": "cpu", "threads": 2,
        "scope": "Single-process local microbenchmark, not a concurrent service or LLM generation latency benchmark."}
    if args.phase == "search":
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=2)
        cache = {}
        arrays = {}
        for channel in ("text", "clip"):
            with np.load(ROOT / f"artifacts/cache/handset_rag_queries_v2/{channel}.npz", allow_pickle=False) as values:
                cache[channel] = dict(zip(values["texts"], values["vectors"].copy()))
            with np.load(runtime.dense / manifest["channels"][channel]["file"], allow_pickle=False) as values:
                arrays[channel] = (values["vectors"].copy(), values["doc_ids"].copy())
        index = json.loads(runtime.index.read_text())
        timings = {k: [] for k in ("bm25", "text_cosine", "clip_cosine", "fusion", "total_cached_search")}
        for repetition in range(4):
            for query in queries:
                start = time.perf_counter(); sparse = score(index, query)[:80]; a = time.perf_counter()
                text = rank_vectors(*arrays["text"], cache["text"][query], 80); b = time.perf_counter()
                clip = rank_vectors(*arrays["clip"], cache["clip"][query], 80); c = time.perf_counter()
                reciprocal_rank_fusion({"bm25": sparse, "text": text, "clip": clip}, weights={"bm25": 1., "text": 1., "clip": .5}, top_k=20)
                end = time.perf_counter()
                if repetition:
                    for key, elapsed in zip(timings, (a-start, b-a, c-b, end-c, end-start)):
                        timings[key].append(elapsed * 1000)
        result.update(timings={k: distribution(v) for k, v in timings.items()},
            exclusions="Excludes query encoding/model load, planner, fact guard, web transport and generation. One warmup repetition discarded.")
    else:
        import torch
        torch.set_num_threads(2)
        config = manifest["channels"][args.phase]
        start = time.perf_counter(); encoder = load_encoder(config["model"], revision=config["revision"])
        result["model_load_seconds"] = time.perf_counter() - start
        times = []
        for query in [queries[0]] + queries:
            start = time.perf_counter()
            values = encoder.encode(query_chunks(query, args.phase), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            vector = np.mean(values, axis=0); vector /= max(np.linalg.norm(vector), 1e-12)
            times.append((time.perf_counter() - start) * 1000)
        result.update(timing=distribution(times[1:]), encoder={k: config[k] for k in ("model", "revision")},
            exclusions="Query encoding only after one warmup query; no query cache hits, search, guard or generation.")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / (args.phase + "_source.py")).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
