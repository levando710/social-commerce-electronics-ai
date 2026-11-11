"""Describe history/popularity strata without claiming causal explanations or tuning models."""
from collections import Counter
import json
from pathlib import Path

from phone_assistant.evaluation.analysis.analyze_connected_results import user_scores
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, sha256


def main():
    output = ROOT / 'artifacts/evaluation/results_handset_failure_strata_v1'
    if output.exists():
        raise FileExistsError("Keep prior failure analysis")
    split = ROOT / 'data/external/benchmarks/handsets_only_v2'
    data = InteractionData(split)
    source = ROOT / 'artifacts/evaluation/results_handsets_multimodal_v1/experiment.json'
    experiment = json.loads(source.read_text(encoding="utf-8"))
    if experiment["split_manifest_sha256"] != sha256(split / "manifest.json"):
        raise ValueError("Failure analysis split mismatch")
    groups, histogram, pop = {}, Counter(), []
    for user, target in data.pairs["test"]:
        history = len(data.train_seen[int(user)])
        histogram[history] += 1
        groups[str(int(user))] = {"history": "one_train_item" if history == 1 else "two_or_more_train_items",
            "target_popularity": "train_degree_at_most_10" if data.item_degree[target] <= 10 else "train_degree_above_10"}
        blocked = data.train_seen[int(user)] | data.validation_seen[int(user)]
        order = sorted((i for i in range(data.n_items) if i not in blocked), key=lambda i: (-int(data.item_degree[i]), data.item_ids[i]))
        pop.append({"user_index": int(user), "rank": order.index(int(target)) + 1})
    methods = {method: [r for r in experiment["runs"] if r["method"] == method]
        for method in ("bm3", "graph_multimodal", "graph_multimodal_gcl")}
    methods["pop"] = [{"test_ranks": pop}]
    strata = {}
    for dimension in ("history", "target_popularity"):
        strata[dimension] = {}
        for group in sorted({v[dimension] for v in groups.values()}):
            users = {u for u, v in groups.items() if v[dimension] == group}
            strata[dimension][group] = {"users": len(users), "methods": {}}
            for method, runs in methods.items():
                values = {}
                for metric in ("hr@10", "ndcg@10"):
                    scores = user_scores(runs, metric)
                    values[metric] = sum(scores[u] for u in users) / len(users) if users else None
                strata[dimension][group]["methods"][method] = values
    result = {"source_sha256": sha256(Path(__file__)), "experiment_sha256": sha256(source),
        "split_manifest_sha256": sha256(split / "manifest.json"), "test_train_history_histogram": dict(sorted(histogram.items())),
        "strata": strata, "limitations": ["Exploratory descriptive strata on the observed test cohort, not an independent validation or causal explanation.",
            "Groups have different users/target distributions; scores cannot isolate the effect of adding history.",
            "No threshold/config/model changes are selected from these groups. Warm targets only; no cold-start conclusion."]}
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Phân tích mô tả theo lịch sử và độ phổ biến", "", "| Stratum | Users | Model | HR@10 | NDCG@10 |", "|---|---:|---|---:|---:|"]
    for dimension, values in strata.items():
        for group, value in values.items():
            for method, metrics in value["methods"].items():
                lines.append(f"| {dimension}/{group} | {value['users']} | {method} | {metrics['hr@10']:.5f} | {metrics['ndcg@10']:.5f} |")
    lines += [""] + result["limitations"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"history_histogram": result["test_train_history_histogram"], "strata": strata}, indent=2))


if __name__ == "__main__":
    main()
