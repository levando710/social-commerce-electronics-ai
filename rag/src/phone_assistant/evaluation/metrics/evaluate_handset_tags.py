"""Required coarse-visual-tag baseline on the frozen handset criteria benchmark; no learned image claims."""
from phone_assistant.common.project_layout import source_path
import json

from phone_assistant.retrieval.search.bm25 import build_index, score
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import visual_tags
from phone_assistant.evaluation.metrics.evaluate_handset_rag import metrics
from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl, sha256


def main():
    output = ROOT / 'artifacts/evaluation/results_handset_visual_tags_v1'
    if output.exists():
        raise FileExistsError("Keep existing baseline results")
    runtime = HandsetRuntime()
    docs = read_jsonl(runtime.corpus)
    sources = {r["item_id"]: r for r in json.loads((ROOT / 'artifacts/cache/handset_features_v1/image_sources.json').read_text())}
    images = ROOT / 'artifacts/cache/handset_feature_work_v1/images'
    tagged = [{**d, "text": d["text"] + "\nCoarse image tags: " + visual_tags(sources[d["doc_id"]], images)} for d in docs]
    index = build_index(tagged)
    bench = ROOT / 'artifacts/evaluation/handset_benchmark_v1'
    queries = read_jsonl(bench / "queries.jsonl")
    qrels = json.loads((bench / "qrels.json").read_text(encoding="utf-8"))
    summaries, records = {}, {}
    for cohort in ("development", "test"):
        rows = [{**q, "ids": [i for i, _ in score(index, q["query"])[:20]]} for q in queries if q["cohort"] == cohort]
        records[cohort] = rows
        summaries[cohort] = {f"at{k}": metrics([r for r in rows if r["kind"] == "retrieval"], qrels, len(docs), k) for k in (5, 10, 20)}
    result = {"source_sha256": {n: sha256(source_path(n)) for n in ("evaluate_handset_tags.py", "evaluate_connected_retrieval.py", "bm25.py", "evaluate_handset_rag.py")},
        "runtime_manifest_sha256": sha256(runtime.path / "manifest.json"), "benchmark_manifest_sha256": sha256(bench / "manifest.json"),
        "summary": summaries, "rows": records, "scope": "Coarse Pillow brightness/color/aspect tags appended to metadata BM25. A weak baseline only, not semantic vision or phone-condition evidence.",
        "limitations": ["Criteria benchmark already inspected; added baseline is exploratory, not an independent new test.",
            "Questions mainly express brand/specs, not visual colors or condition. Automatic qrels are not human relevance.",
            "Main multimodal model still uses real CLIP; no replacement by these coarse tags."]}
    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source.py").write_bytes((ROOT / 'src/phone_assistant/evaluation/metrics/evaluate_handset_tags.py').read_bytes())
    lines = ["# BM25 + coarse visual tags: handset baseline", "", result["scope"], "",
        "| Cohort | P@10 | Recall@10 | NDCG@10 | MRR@10 |", "|---|---:|---:|---:|---:|"]
    for cohort, value in summaries.items():
        m = value["at10"]
        lines.append(f"| {cohort} | " + " | ".join(f"{m[k]:.4f}" for k in ("precision@10", "recall@10", "ndcg@10", "mrr@10")) + " |")
    (output / "report.md").write_text("\n".join(lines + [""] + result["limitations"]) + "\n", encoding="utf-8")
    print(json.dumps({k: v["at10"] for k, v in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
