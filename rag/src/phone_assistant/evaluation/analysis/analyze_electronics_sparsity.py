"""Paired-user comparisons for the predeclared missing-feedback experiment, without test selection."""
import argparse
import json
from pathlib import Path

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores
from phone_assistant.recommendation.recsys_data import ROOT, sha256

METHODS = ("graph_multimodal", "graph_multimodal_gcl")
METRICS = ("hr@10", "ndcg@10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_electronics_sparsity_analysis_v1')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Keep existing comparisons")
    paths = {name: ROOT / 'artifacts/evaluation' / folder / "experiment.json" for name, folder in
        (("full", "results_electronics_multimodal_v1"), ("sparse", "results_electronics_sparse50_v1"))}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    split_paths = {"full": ROOT / 'data/external/benchmarks/connected_electronics_v2',
                   "sparse": ROOT / 'data/external/benchmarks/electronics_sparse50_v1'}
    for filename in ("id_maps.json", "validation.jsonl", "test.jsonl", "products.jsonl"):
        if sha256(split_paths["full"] / filename) != sha256(split_paths["sparse"] / filename):
            raise ValueError("Paired evaluation requires identical IDs/heldout data/catalog")
    scores, seeds = {}, {}
    for view, report in reports.items():
        for method in METHODS:
            runs = [r for r in report["runs"] if r["method"] == method]
            seeds[view, method] = sorted(r["seed"] for r in runs)
            scores[view, method] = {metric: user_scores(runs, metric) for metric in METRICS}
    if any(value != seeds["full", METHODS[0]] for value in seeds.values()):
        raise ValueError("Paired methods/views have different seeds")
    comparisons = {}
    for view in ("full", "sparse"):
        comparisons[f"{view}:GCL_minus_MM"] = {metric: paired_bootstrap(scores[view, METHODS[0]][metric],
            scores[view, METHODS[1]][metric]) for metric in METRICS}
    for method in METHODS:
        comparisons[f"{method}:sparse_minus_full"] = {metric: paired_bootstrap(scores["full", method][metric],
            scores["sparse", method][metric]) for metric in METRICS}
    summary = json.loads((split_paths["sparse"] / "summary.json").read_text(encoding="utf-8"))
    result = {"source_sha256": sha256(Path(__file__)), "experiment_sha256": {k: sha256(v) for k, v in paths.items()},
        "sparse_summary": summary, "comparisons": comparisons,
        "limitations": ["Fixed mask and dataset, conditional on existing training seeds; no mask/source-data uncertainty.",
            "Only one missing-feedback regime; not a study of all sparsity/noise conditions.",
            "Ceiling per-user retained counts explains why actual retention exceeds 50%; the minimum-one constraint also guarantees user presence.",
            "No model or parameter selection based on these test comparisons."]}
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Controlled missing-feedback comparisons", "", f"Actual train retention: {summary['actual_retain_fraction']:.4%}; per-user ceiling, minimum one event.", "",
        "| Comparison | Metric | Mean delta | Paired-user 95% interval |", "|---|---|---:|---|"]
    for comparison, metrics in comparisons.items():
        for metric, value in metrics.items():
            lower, upper = value["percentile_ci95"]
            lines.append(f"| {comparison} | {metric} | {value['mean_delta']:+.6f} | [{lower:+.6f}, {upper:+.6f}] |")
    (args.output / "report.md").write_text("\n".join(lines + [""] + result["limitations"]) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
