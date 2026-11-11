"""Exercise a real pinned serving encoder against an existing frozen offline query."""
from phone_assistant.common.project_layout import source_path
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phone_assistant.retrieval.serving.connected_serving import ConnectedRuntime, ConnectedDenseRetriever
from phone_assistant.retrieval.search.dense_search import rank_vectors
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import load_queries
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, choices=("text", "clip"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = ConnectedRuntime()
    source = runtime.split / "agent_queries.jsonl"
    queries = read_jsonl(source)
    query_index = max(range(len(queries)), key=lambda i: len(queries[i]["query"].split()))
    documents = read_jsonl(runtime.corpus)
    retriever = ConnectedDenseRetriever(runtime.dense, documents)
    config = retriever.manifest["channels"][args.channel]
    feature_manifest = json.loads((ROOT / 'artifacts/cache/connected_features_v1/manifest.json').read_text(encoding="utf-8"))
    expected_config = feature_manifest["text_encoder" if args.channel == "text" else "image_encoder"]
    vectors = load_queries(queries, source, ROOT / 'artifacts/cache/connected_query_vectors_v1', args.channel, expected_config)
    query = queries[query_index]["query"]
    actual = retriever.search(query, channel=args.channel, top_k=10)
    expected = rank_vectors(*retriever.arrays[args.channel], vectors[query_index], 10)
    np.testing.assert_allclose(retriever.query_cache[(args.channel, query)], vectors[query_index], atol=1e-5, rtol=1e-4)
    if [i for i, _ in actual] != [i for i, _ in expected]:
        raise ValueError("Serving top10 disagrees with frozen offline vectors")
    result = {"channel": args.channel, "dialogue_id": queries[query_index]["dialogue_id"], "query_words": len(query.split()),
        "encoder": config, "runtime_manifest_sha256": sha256(runtime.path / "manifest.json"),
        "source_sha256": {name: sha256(source_path(name)) for name in ("smoke_connected_dense.py", "connected_serving.py")},
        "max_query_vector_absolute_error": float(np.max(np.abs(retriever.query_cache[(args.channel, query)] - vectors[query_index]))),
        "top10_match": True, "loaded_encoders": list(retriever.encoders), "top10": actual}
    if args.output:
        if args.output.exists():
            raise FileExistsError("Keep the previous smoke output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
