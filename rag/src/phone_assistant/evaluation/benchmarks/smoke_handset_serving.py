"""Run the real handset multimodal serving path locally, without generation/API calls."""
from phone_assistant.common.project_layout import source_path
import json
import time

import torch

from phone_assistant.conversation.co_guiding import CoGuidingEngine
from phone_assistant.retrieval.serving.connected_serving import ConnectedDenseRetriever
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.data.phone_scope import handset_decision, HandsetRecommender, HandsetRanking
from phone_assistant.retrieval.queries.product_constraints import ProductQueryPlanner, connected_facts, connected_verdict
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256
from phone_assistant.training.train_recsys_portable import peak_process_rss_mib


def main():
    output = ROOT / 'artifacts/evaluation/results_handsets_serving_smoke_v1.json'
    if output.exists():
        raise FileExistsError("Keep the prior serving smoke result")
    torch.set_num_threads(2)
    started = time.perf_counter()
    runtime = HandsetRuntime()
    docs = read_jsonl(runtime.corpus)
    if not all(handset_decision(d)[0] for d in docs):
        raise ValueError("Serving corpus contains non-handset candidates")
    recommender = HandsetRecommender(runtime.recommender(), docs)
    dense = ConnectedDenseRetriever(runtime.dense, docs)
    scoped = HandsetRanking(dense, [d["doc_id"] for d in docs], len(docs))
    index = json.loads(runtime.index.read_text(encoding="utf-8"))
    engine = CoGuidingEngine(docs, index, recommender, dense=scoped, method="multimodal",
        planner=ProductQueryPlanner(docs), fact_extractor=connected_facts, verdict=connected_verdict)
    result = engine.run("Samsung phone", mode="coupled", top_k=5)
    if not result["results"] or any(not row["constraint_check"]["eligible"] for row in result["results"]):
        raise ValueError("No verified phone retrieval results")
    if set(dense.encoders) != {"text", "clip"}:
        raise ValueError("Both real pinned neural channels were not exercised")
    payload = {"runtime_manifest_sha256": sha256(runtime.path / "manifest.json"),
        "source_sha256": {name: sha256(source_path(name)) for name in
            ("smoke_handset_serving.py", "handset_serving.py", "connected_serving.py", "phone_scope.py", "co_guiding.py")},
        "query": "Samsung phone", "loaded_encoders": list(dense.encoders), "catalog": len(docs),
        "retrieved_ids": [r["doc_id"] for r in result["results"]],
        "feed_ids": [r["doc_id"] for r in result["recommendations"]], "feedback": result["feedback"],
        "llm_calls": 0, "seconds_including_encoder_load": time.perf_counter() - started,
        "peak_process_rss_mib": peak_process_rss_mib(),
        "scope": "One real query smoke; not a relevance benchmark, concurrency test or answer-quality evaluation."}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
