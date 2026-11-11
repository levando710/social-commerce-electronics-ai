"""Paired exploration of structured-RAM quarantine with a fixed interaction split."""
import json
from pathlib import Path

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores
from phone_assistant.recommendation.recsys_data import ROOT, sha256


def main():
    output = ROOT / 'artifacts/evaluation/results_ram_quarantine_analysis_v1'
    if output.exists():
        raise FileExistsError("Keep the existing comparison")
    raw_path = ROOT / 'artifacts/evaluation/results_handsets_multimodal_v1/experiment.json'
    clean_path = ROOT / 'artifacts/evaluation/results_handsets_ram_quarantine_v1/experiment.json'
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (raw_path, clean_path)]
    if reports[0]["split_manifest_sha256"] != reports[1]["split_manifest_sha256"]:
        raise ValueError("Ablation changed the interaction split")
    for path in (raw_path, clean_path):
        check = json.loads((path.parent / "checkpoint_verification.json").read_text(encoding="utf-8"))
        if not check["all_passed"] or check["experiment_sha256"] != sha256(path):
            raise ValueError("Verify selected checkpoints before comparing")
    result = {"source_sha256": sha256(Path(__file__)), "raw_experiment_sha256": sha256(raw_path),
        "quarantine_experiment_sha256": sha256(clean_path), "comparisons": {},
        "limitations": ["Exploratory after observing raw benchmark and suspicious metadata, not an unseen confirmatory test.",
            "Quarantine is a conservative heuristic, not human-confirmed corrected RAM. Free-text descriptions remain unchanged.",
            "Same nominal seeds/config; changed spec-node count can alter random initialization/augmentation streams.",
            "Paired-user intervals conditional on existing seeds; no training-data uncertainty or multiple-testing correction.",
            "No automatic production model replacement based on test results; no API calls or LLM-quality inference."]}
    for method in ("graph_multimodal", "graph_multimodal_gcl"):
        groups = [[r for r in report["runs"] if r["method"] == method] for report in reports]
        if sorted(r["seed"] for r in groups[0]) != sorted(r["seed"] for r in groups[1]):
            raise ValueError("Training seeds differ")
        values = {}
        for metric in ("hr@10", "ndcg@10"):
            first, second = (user_scores(runs, metric) for runs in groups)
            values[metric] = {"raw": sum(first.values()) / len(first), "quarantine": sum(second.values()) / len(second),
                "paired": paired_bootstrap(first, second)}
        result["comparisons"][method] = values
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Ablation: bỏ trường RAM cấu trúc nghi vấn", "", "| Model | Metric | Raw | Quarantine | Delta | Paired-user 95% interval |",
        "|---|---|---:|---:|---:|---|"]
    for method, metrics in result["comparisons"].items():
        for metric, value in metrics.items():
            lo, hi = value["paired"]["percentile_ci95"]
            lines.append(f"| {method} | {metric} | {value['raw']:.5f} | {value['quarantine']:.5f} | {value['paired']['mean_delta']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines += [""] + result["limitations"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
