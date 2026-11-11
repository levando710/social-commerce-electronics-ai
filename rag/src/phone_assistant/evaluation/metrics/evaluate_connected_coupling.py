"""Fixed-seed inference coupling ablation on the frozen simulated query cohorts.

Original query text and cached encoder vectors are identical in all arms.
Only explicit fact gating and the two inference feedback directions vary.
No generation, target-dependent rewriting or parameter training occurs here.
"""
from phone_assistant.common.project_layout import source_path
import argparse
import copy
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap
from phone_assistant.retrieval.search.bm25 import score
from phone_assistant.conversation.co_guiding import CoGuidingEngine, MODES
from phone_assistant.retrieval.serving.connected_serving import ConnectedRuntime
from phone_assistant.retrieval.search.dense_search import reciprocal_rank_fusion, result_rows
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import load_queries, metrics, ordered
from phone_assistant.retrieval.queries.product_constraints import ProductQueryPlanner, connected_facts, connected_verdict
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256, load_features


def scores_by_user(rows, key="rank"):
    users = defaultdict(list)
    for row in rows:
        rank = row[key]
        users[row["user_id"]].append(1 / np.log2(rank + 1) if rank <= 10 else 0.)
    return {u: float(np.mean(values)) for u, values in users.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_coupling_v2')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep frozen evaluations; choose a new output")
    threadpool_limits(limits=2)
    runtime = ConnectedRuntime()
    service = runtime.recommender(seed=42)
    data = service.data
    features, manifest = load_features(ROOT / 'artifacts/cache/connected_features_v1', data)
    documents = list(service.documents.values())
    index = json.loads(runtime.index.read_text(encoding="utf-8"))
    query_source = data.directory / "agent_queries.jsonl"
    queries = read_jsonl(query_source)
    vectors = {channel: load_queries(queries, query_source, ROOT / 'artifacts/cache/connected_query_vectors_v1', channel,
        manifest["text_encoder" if channel == "text" else "image_encoder"]) for channel in ("text", "clip")}
    sources = ("evaluate_connected_coupling.py", "co_guiding.py", "product_constraints.py", "conversation_state.py",
               "connected_serving.py", "recsys_service.py", "dense_search.py", "bm25.py")
    plan = {"split_manifest_sha256": sha256(data.directory / "manifest.json"),
        "feature_manifest_sha256": sha256(ROOT / 'artifacts/cache/connected_features_v1/manifest.json'),
        "runtime_manifest_sha256": sha256(runtime.path / "manifest.json"),
        "recommender_experiment_sha256": sha256(ROOT / 'artifacts/evaluation/results_connected_multimodal_v2/experiment.json'),
        "source_sha256": {n: sha256(source_path(n)) for n in sources},
        "fixed_recommender": {"method": service.method, "seed": 42},
        "arms": [f"{guard}/{mode}" for guard in ("unguarded", "guarded") for mode in MODES],
        "query_policy": "Original first user utterance in every arm; query rewrite disabled to reuse identical frozen query vectors.",
        "retrieval": "Full-catalog BM25/text/CLIP rankings; RRF weights 1/1/0.5, constant 60; take 80 candidates.",
        "feedback": "Fixed engine weights: preference RRF .2; evidence reward .03/rank; top20 evidence; feed100; extra40 preference candidates.",
        "selection": "No parameter tuning or model selection; seed 42 fixed before reading test results.",
        "development_disclosure": "Parser corrections followed inspection of v1 test failures (existing-phone references and comma-separated battery ranges). This is an exploratory re-evaluation on the same test cohort, not a new unseen test. No coupling weights or model parameters changed.",
        "limitations": ["LLM-simulated first queries, one goal per user and two personas; no human study.",
            "Unjudged products are not verified irrelevant. P@K is single-target judged precision only.",
            "Feed quality measures query-conditioned target recovery, not long-term independent recommendation quality.",
            "Hard constraints use heuristic title/spec parsing; rejection can be parser error or incomplete catalog evidence.",
            "No multi-turn or generated-answer quality is measured."]}
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.output / "source").mkdir()
    for name in sources:
        (args.output / "source" / name).write_bytes((source_path(name)).read_bytes())

    planner = ProductQueryPlanner(documents)
    plans = {}
    def original_plan(query, previous=None):
        if previous is not None:
            raise ValueError("This evaluation supports first queries only")
        if query not in plans:
            plans[query] = planner(query)
            plans[query]["query_rewrite"] = query
        return copy.deepcopy(plans[query])
    cached = {}
    def retrieve_cached(_index, _docs, query, **kwargs):
        return cached[query][:kwargs["top_k"]]
    facts = {d["doc_id"]: connected_facts(d) for d in documents}
    engines = {guard: CoGuidingEngine(documents, index, service, method="multimodal", planner=original_plan,
        fact_extractor=lambda doc: facts[doc["doc_id"]], verdict=connected_verdict if guard == "guarded" else
            lambda facts, constraints: {"eligible": True, "violations": [], "unknown": []},
        retrieval=retrieve_cached) for guard in ("unguarded", "guarded")}
    # Goal labels are evaluator-only, never passed to the planner or engine.
    labels = {r["user_id"]: r for r in read_jsonl(data.directory / "evaluator_labels.jsonl")}
    rows, summaries = {}, {}
    for split in ("validation", "test"):
        rows[split] = {arm: [] for arm in plan["arms"]}
        for q_index, query in enumerate(queries):
            if query["split"] != split:
                continue
            text = query["query"]
            if text not in cached:
                rankings = {"bm25": score(index, text),
                    "text": ordered(features["text"] @ vectors["text"][q_index], data.item_ids),
                    "clip": ordered(features["image"] @ vectors["clip"][q_index], data.item_ids, features["image_mask"])}
                hits = reciprocal_rank_fusion(rankings, weights={"bm25": 1., "text": 1., "clip": .5}, top_k=80)
                cached[text] = result_rows(documents, hits, text)
            user = data.maps["users"][query["user_id"]]
            feed = service.recommend(user, 100)
            target = labels[query["user_id"]]["item_id"]
            for guard, engine in engines.items():
                verdict = connected_verdict(engine.facts[target], original_plan(text)["constraints"])
                for mode in MODES:
                    result = engine.run(text, user_index=user, mode=mode, top_k=20, preference_feed=feed)
                    retrieved = [r["doc_id"] for r in result["results"]]
                    recommended = [r["doc_id"] for r in result["recommendations"]]
                    rank = lambda ids: ids.index(target) + 1 if target in ids else data.n_items + 1
                    rows[split][f"{guard}/{mode}"].append({"dialogue_id": query["dialogue_id"], "user_id": query["user_id"],
                        "rank": rank(retrieved), "feed_rank": rank(recommended), "top_items": retrieved,
                        "feed_items": recommended, "abstention": result["abstention"], "feedback": result["feedback"],
                        "target_verdict": verdict, "constraints": result["query_plan"]["constraints"]})
        summaries[split] = {}
        for arm, values in rows[split].items():
            feed_rows = [{**v, "rank": v["feed_rank"], "top_items": v["feed_items"]} for v in values]
            summaries[split][arm] = {"retrieval": metrics(values, data.n_items), "feed": metrics(feed_rows, data.n_items),
                "abstention_rate": float(np.mean([v["abstention"] for v in values])),
                "target_unknown_queries": sum(bool(v["target_verdict"]["unknown"]) for v in values),
                "target_violation_queries": sum(bool(v["target_verdict"]["violations"]) for v in values)}
        print(f"Completed {split}: {len(next(iter(rows[split].values())))} queries", flush=True)
    intervals = {guard: {task: paired_bootstrap(scores_by_user(rows["test"][f"{guard}/decoupled"], key),
        scores_by_user(rows["test"][f"{guard}/coupled"], key)) for task, key in (("retrieval", "rank"), ("feed", "feed_rank"))}
        for guard in ("unguarded", "guarded")}
    experiment = {**plan, "summary": summaries, "paired_test_ndcg10": intervals, "rows": rows}
    (args.output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    lines = ["# Co-Guiding inference ablation", "", plan["query_policy"], "",
        "| Arm | Target recall@10 in retrieval | NDCG@10 | Target HR@10 in feed | Feed NDCG@10 | Abstention |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, value in summaries["test"].items():
        values = (value["retrieval"]["recall@10"], value["retrieval"]["ndcg@10"], value["feed"]["recall@10"],
            value["feed"]["ndcg@10"], value["abstention_rate"])
        lines.append(f"| {arm} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", plan["development_disclosure"], ""] + plan["limitations"]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
