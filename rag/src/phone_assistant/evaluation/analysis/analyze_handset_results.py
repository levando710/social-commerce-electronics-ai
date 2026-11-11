"""Combine handset-only baselines with paired-user bootstrap; no test-based selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_handsets_analysis_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep the frozen analysis; use a new output")
    data = InteractionData(ROOT / 'data/external/benchmarks/handsets_only_v2')
    paths = {key: ROOT / 'artifacts/evaluation' / f"results_{key}" / "experiment.json"
             for key in ("handsets_lightgcn_v1", "handsets_multimodal_v1")}
    experiments = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    runs = {}
    for name, experiment in experiments.items():
        if experiment["split_manifest_sha256"] != sha256(data.directory / "manifest.json"):
            raise ValueError("Experiments do not share the handset split")
        verification = json.loads((paths[name].parent / "checkpoint_verification.json").read_text(encoding="utf-8"))
        if not verification["all_passed"] or verification["experiment_sha256"] != sha256(paths[name]):
            raise ValueError("Verify each selected checkpoint first")
        for run in experiment["runs"]:
            runs.setdefault(run["method"], []).append(run)
    popularity = []
    for row in data.rows["test"]:
        user, target = data.maps["users"][row["user_id"]], data.maps["items"][row["item_id"]]
        blocked = data.train_seen[user] | data.validation_seen[user]
        order = sorted((i for i in range(data.n_items) if i not in blocked), key=lambda i: (-int(data.item_degree[i]), data.item_ids[i]))
        popularity.append({"user_index": user, "rank": order.index(target) + 1})
    runs["pop"] = [{"seed": None, "test_ranks": popularity}]
    summaries = {method: {metric: float(np.mean(list(user_scores(values, metric).values())))
        for metric in ("hr@10", "ndcg@10")} for method, values in runs.items()}
    comparisons = {}
    for first, second in (("pop", "lightgcn"), ("pop", "vbpr"), ("pop", "bm3"),
        ("graph_text_only", "graph_multimodal"), ("graph_multimodal", "graph_multimodal_gcl")):
        comparisons[f"{second}_minus_{first}"] = {metric: paired_bootstrap(user_scores(runs[first], metric),
            user_scores(runs[second], metric)) for metric in ("hr@10", "ndcg@10")}
    result = {"source_sha256": sha256(Path(__file__)), "split_manifest_sha256": sha256(data.directory / "manifest.json"),
        "experiment_sha256": {name: sha256(path) for name, path in paths.items()}, "summary": summaries,
        "paired_bootstrap": comparisons, "users": data.n_users, "test_users": len(data.rows["test"]), "catalog": data.n_items,
        "limitations": ["Automatically screened handset candidates; not a human-verified full catalog.",
            "Review-based chronological user holdouts, static catalog and activity-conditioned evaluation users.",
            "All test targets warm in v2; does not establish cold-item performance.",
            "Intervals resample paired users after averaging the existing seeds; do not capture training-data uncertainty.",
            "No dialogue, answer grounding or human preference-study claim; all methods kept even when worse than Pop."]}
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Kết quả gợi ý riêng cho điện thoại", "", f"Catalog {data.n_items}; train {len(data.rows['train'])}; validation/test {len(data.rows['validation'])}/{len(data.rows['test'])}.",
        "", "| Model | HR@10 | NDCG@10 |", "|---|---:|---:|"]
    for method in ("pop", "lightgcn", "vbpr", "bm3", "graph_text_only", "graph_multimodal", "graph_multimodal_gcl"):
        values = summaries[method]
        lines.append(f"| {method} | {values['hr@10']:.5f} | {values['ndcg@10']:.5f} |")
    lines += ["", "## Paired-user bootstrap", "", "| Comparison | Metric | Delta | 95% interval |", "|---|---|---:|---|"]
    for comparison, metrics in comparisons.items():
        for metric, value in metrics.items():
            lo, hi = value["percentile_ci95"]
            lines.append(f"| {comparison} | {metric} | {value['mean_delta']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines += [""] + [f"- {v}" for v in result["limitations"]]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
