"""Bilingual metadata-criteria IR diagnostic with separately cached real encoders."""
from phone_assistant.common.project_layout import source_path
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from phone_assistant.retrieval.search.bm25 import score
from phone_assistant.conversation.co_guiding import CoGuidingEngine
from phone_assistant.retrieval.search.dense_search import rank_vectors, reciprocal_rank_fusion, load_encoder
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import query_chunks
from phone_assistant.conversation.handset_dialogue import dialogue_options
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256

METHODS = ("bm25", "dense_text", "clip", "hybrid", "hybrid_guard", "rewrite_guard")


def metrics(rows, qrels, n_items, k=10):
    by_intent = defaultdict(list)
    for row in rows:
        labels = qrels[row["intent_id"]]
        relevant = {item for item, label in labels.items() if label == "relevant"}
        ids = row["ids"][:k]
        hits = [int(item in relevant) for item in ids]
        dcg = sum(hit / np.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
        rank = next((i + 1 for i, hit in enumerate(hits) if hit), None)
        by_intent[row["intent_id"]].append({"precision": sum(hits) / k, "recall": sum(hits) / len(relevant),
            "ndcg": dcg / ideal if ideal else 0., "mrr": 1 / rank if rank else 0.,
            "unjudged_fraction": sum(labels[item] == "unjudged" for item in ids) / len(ids) if ids else 0.})
    result = {f"{name}@{k}": float(np.mean([np.mean([v[name] for v in group]) for group in by_intent.values()]))
              for name in ("precision", "recall", "ndcg", "mrr", "unjudged_fraction")}
    result.update(intents=len(by_intent), queries=len(rows), **{f"coverage@{k}": len({i for r in rows for i in r["ids"][:k]}) / n_items})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("text", "clip", "evaluate"))
    parser.add_argument("--cohort", choices=("development", "test"), default="development")
    parser.add_argument("--work", type=Path, default=ROOT / 'artifacts/cache/handset_rag_queries_v1')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_handset_rag_dev_v1')
    args = parser.parse_args()
    runtime = HandsetRuntime()
    bench = ROOT / 'artifacts/evaluation/handset_benchmark_v1'
    manifest = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256(bench / name) != digest:
            raise ValueError("Benchmark changed")
    docs = read_jsonl(runtime.corpus)
    queries = read_jsonl(bench / "queries.jsonl")
    options = dialogue_options(runtime, docs)
    plans = {q["id"]: options["planner"](q["query"]) for q in queries}
    texts = sorted({text for q in queries for text in (q["query"], plans[q["id"]]["query_rewrite"])})
    dense_manifest = json.loads((runtime.dense / "manifest.json").read_text(encoding="utf-8"))
    contract = {"benchmark_manifest_sha256": sha256(bench / "manifest.json"), "runtime_manifest_sha256": sha256(runtime.path / "manifest.json"),
        "source_sha256": {name: sha256(source_path(name)) for name in
            ("evaluate_handset_rag.py", "handset_dialogue.py", "product_constraints.py", "conversation_state.py")}}
    args.work.mkdir(parents=True, exist_ok=True)
    if args.phase != "evaluate":
        import torch
        torch.set_num_threads(2)
        channel = args.phase
        config = dense_manifest["channels"][channel]
        expected = {**contract, "encoder": {k: config[k] for k in ("model", "revision")}, "channel": channel}
        path = args.work / f"{channel}.npz"
        meta = args.work / f"{channel}.json"
        if meta.exists():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if saved["contract"] != expected or sha256(path) != saved["sha256"]:
                raise ValueError("Query cache changed; use a new work directory")
            print(f"{channel} vectors already verified")
            return
        encoder = load_encoder(config["model"], revision=config["revision"])
        chunks, owners = [], []
        for i, text in enumerate(texts):
            values = query_chunks(text, channel)
            chunks.extend(values); owners.extend([i] * len(values))
        encoded = encoder.encode(chunks, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.zeros((len(texts), encoded.shape[1]), dtype=np.float32)
        np.add.at(vectors, owners, encoded)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        np.savez_compressed(path, texts=np.asarray(texts), vectors=vectors)
        meta.write_text(json.dumps({"contract": expected, "sha256": sha256(path), "queries": len(texts)}, indent=2), encoding="utf-8")
        print(f"Encoded {channel}: {len(texts)} unique original/rewritten queries")
        return
    if args.output.exists():
        raise FileExistsError("Preserve evaluation; use a new output")
    arrays, cache = {}, {}
    for channel in ("text", "clip"):
        config = dense_manifest["channels"][channel]
        meta = json.loads((args.work / f"{channel}.json").read_text(encoding="utf-8"))
        expected = {**contract, "encoder": {k: config[k] for k in ("model", "revision")}, "channel": channel}
        if meta["contract"] != expected or sha256(args.work / f"{channel}.npz") != meta["sha256"]:
            raise ValueError("Query cache/source mismatch")
        with np.load(args.work / f"{channel}.npz", allow_pickle=False) as stored:
            if list(stored["texts"]) != texts:
                raise ValueError("Query cache order mismatch")
            cache[channel] = dict(zip(texts, stored["vectors"].copy()))
        with np.load(runtime.dense / config["file"], allow_pickle=False) as stored:
            arrays[channel] = (stored["vectors"].copy(), stored["doc_ids"].copy())
    class CachedDense:
        def search(self, query, *, channel="text", top_k=60):
            return rank_vectors(*arrays[channel], cache[channel][query], top_k)
    dense = CachedDense()
    index = json.loads(runtime.index.read_text(encoding="utf-8"))
    planner = options["planner"]
    def unchanged_query(query, previous=None):
        plan = planner(query, previous)
        plan["query_rewrite"] = query
        return plan
    guarded = CoGuidingEngine(docs, index, runtime.recommender(), dense=dense, method="multimodal",
        **{**options, "planner": unchanged_query})
    rewritten = CoGuidingEngine(docs, index, guarded.recommender, dense=dense, method="multimodal", **options)
    selected = [q for q in queries if q["cohort"] == args.cohort]
    rows, cases = {method: [] for method in METHODS}, []
    for q in selected:
        text = q["query"]
        rankings = {"bm25": score(index, text), "text": dense.search(text, channel="text", top_k=80),
                    "clip": dense.search(text, channel="clip", top_k=80)}
        hybrid = reciprocal_rank_fusion({**rankings, "bm25": rankings["bm25"][:80]},
            weights={"bm25": 1., "text": 1., "clip": .5}, top_k=20)
        outputs = {"bm25": [i for i, _ in rankings["bm25"][:20]], "dense_text": [i for i, _ in rankings["text"][:20]],
            "clip": [i for i, _ in rankings["clip"][:20]], "hybrid": [r[0] for r in hybrid]}
        results = {}
        for method, engine in (("hybrid_guard", guarded), ("rewrite_guard", rewritten)):
            results[method] = engine.run(text, mode="decoupled", top_k=20)
            outputs[method] = [r["doc_id"] for r in results[method]["results"]]
        for method, ids in outputs.items():
            rows[method].append({**q, "ids": ids})
        cases.append({"query_id": q["id"], "query": text, "intent_id": q["intent_id"], "kind": q["kind"],
            "query_plan": results["rewrite_guard"]["query_plan"], "contexts": results["rewrite_guard"]["results"][:3],
            "abstention": results["rewrite_guard"]["abstention"]})
    qrels = json.loads((bench / "qrels.json").read_text(encoding="utf-8"))
    summary = {}
    for method, values in rows.items():
        summary[method] = {f"at{k}": metrics([r for r in values if r["kind"] == "retrieval"], qrels, len(docs), k) for k in (5, 10, 20)}
        rejects = [r for r in values if r["kind"] == "expected_abstention"]
        summary[method]["abstention_cases"] = {"total": len(rejects), "correct_empty": sum(not r["ids"] for r in rejects)}
        summary[method]["languages"] = {lang: metrics([r for r in values if r["kind"] == "retrieval" and r["language"] == lang], qrels, len(docs)) for lang in ("vi", "en")}
    report = {**contract, "cohort": args.cohort, "summary": summary, "rows": rows,
        "limitations": [manifest["scope"], manifest["label_policy"], "Macro average over criteria, not independent paraphrase users.",
            "Unjudged products count as nonrelevant in conventional P/R/NDCG; unjudged fractions are reported.",
            "No query-user alignment or Co-Guiding personalization benefit is inferred from these catalog tasks.",
            "Development queries may guide fixes; test criteria are evaluated separately after implementation is frozen."]}
    args.output.mkdir(parents=True)
    (args.output / "experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    (args.output / "source").mkdir()
    for name in contract["source_sha256"]:
        (args.output / "source" / name).write_bytes((source_path(name)).read_bytes())
    lines = ["# Handset metadata-criteria retrieval diagnostic", "", f"Cohort: {args.cohort}", "",
        "| Method | P@10 | Recall@10 | NDCG@10 | MRR@10 | Coverage@10 |", "|---|---:|---:|---:|---:|---:|"]
    for method, value in summary.items():
        m = value["at10"]
        lines.append(f"| {method} | " + " | ".join(f"{m[key]:.4f}" for key in ("precision@10", "recall@10", "ndcg@10", "mrr@10", "coverage@10")) + " |")
    lines += [""] + report["limitations"]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({m: v["at10"] for m, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
