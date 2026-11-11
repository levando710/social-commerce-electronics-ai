"""Report every predeclared mask/fraction, with paired-user conditional uncertainty."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores
from phone_assistant.data.preparation.prepare_electronics_robustness import RETAIN_FRACTIONS, MASK_SEEDS
from phone_assistant.recommendation.recsys_data import ROOT, sha256
from phone_assistant.training.run_electronics_robustness import METHODS, MODEL_SEEDS, OUTPUT as INPUT
from phone_assistant.training.train_recsys_portable import atomic_json

OUTPUT = ROOT / 'artifacts/evaluation/results_electronics_robustness_analysis_v1'
METRICS = ("hr@10", "ndcg@10")


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep prior analysis; never choose the best masks after reading results")
    path = INPUT / "experiment.json"
    experiment = json.loads(path.read_text())
    expected = {(fraction, mask) for fraction in RETAIN_FRACTIONS for mask in MASK_SEEDS}
    if not experiment["complete"] or len(experiment["cells"]) != len(expected) or {(c["fraction"], c["mask_seed"]) for c in experiment["cells"]} != expected:
        raise ValueError("Incomplete robustness grid")
    grouped, cell_reports, provenance = defaultdict(list), [], {}
    full_path = ROOT / 'artifacts/evaluation/results_electronics_multimodal_v1/experiment.json'
    full = json.loads(full_path.read_text())
    full_split = ROOT / 'data/external/benchmarks/connected_electronics_v2'
    full_scores = {method: {metric: user_scores([r for r in full["runs"] if r["method"] == method], metric) for metric in METRICS} for method in METHODS}
    for cell in experiment["cells"]:
        report_path = ROOT / cell["experiment"]
        if sha256(report_path) != cell["experiment_sha256"]:
            raise ValueError("Robustness experiment changed")
        verification_path = report_path.parent / "checkpoint_verification.json"
        if sha256(verification_path) != cell["verification_sha256"]:
            raise ValueError("Checkpoint verification changed")
        verification = json.loads(verification_path.read_text())
        if not verification["all_passed"] or verification["runs"] != len(METHODS) * len(MODEL_SEEDS):
            raise ValueError("Incomplete checkpoint verification")
        split = ROOT / cell["split"]
        for filename in ("id_maps.json", "validation.jsonl", "test.jsonl", "products.jsonl", "agent_queries.jsonl", "evaluator_labels.jsonl"):
            if sha256(split / filename) != sha256(full_split / filename):
                raise ValueError("Paired heldout data/catalog differ")
        report = json.loads(report_path.read_text())
        summary = json.loads((split / "summary.json").read_text())
        row = {"fraction": cell["fraction"], "mask_seed": cell["mask_seed"],
            "actual_retain_fraction": summary["actual_retain_fraction"], "train_events": summary["train_after"], "methods": {}}
        scores = {}
        for method in METHODS:
            runs = [r for r in report["runs"] if r["method"] == method]
            if sorted(r["seed"] for r in runs) != list(MODEL_SEEDS):
                raise ValueError("Mismatched or duplicated model seeds")
            grouped[cell["fraction"], method].extend(runs)
            row["methods"][method] = {metric: float(np.mean([r["test"]["all"][metric] for r in runs])) for metric in METRICS}
            scores[method] = {metric: user_scores(runs, metric) for metric in METRICS}
        row["GCL_minus_MM"] = {metric: paired_bootstrap(scores[METHODS[0]][metric], scores[METHODS[1]][metric]) for metric in METRICS}
        cell_reports.append(row); provenance[cell["experiment"]] = sha256(report_path)
    aggregated = {}
    for fraction in RETAIN_FRACTIONS:
        score = {method: {metric: user_scores(grouped[fraction, method], metric) for metric in METRICS} for method in METHODS}
        row = {"methods": {method: {metric: float(np.mean(list(score[method][metric].values()))) for metric in METRICS} for method in METHODS},
            "GCL_minus_MM": {metric: paired_bootstrap(score[METHODS[0]][metric], score[METHODS[1]][metric]) for metric in METRICS},
            "sparse_minus_full": {method: {metric: paired_bootstrap(full_scores[method][metric], score[method][metric]) for metric in METRICS} for method in METHODS},
            "relative_change_in_GCL_advantage": {}}
        for metric in METRICS:
            users = full_scores[METHODS[0]][metric].keys()
            first = {u: full_scores[METHODS[1]][metric][u] - full_scores[METHODS[0]][metric][u] for u in users}
            second = {u: score[METHODS[1]][metric][u] - score[METHODS[0]][metric][u] for u in users}
            row["relative_change_in_GCL_advantage"][metric] = paired_bootstrap(first, second)
        aggregated[str(fraction)] = row
    result = {"experiment_sha256": sha256(path), "source_sha256": sha256(Path(__file__)),
        "full_reference_sha256": sha256(full_path), "cell_experiment_sha256": provenance,
        "cells": cell_reports, "aggregated": aggregated, "total_runs_including_reused": sum(len(v) for v in grouped.values()),
        "limitations": ["Two fixed masks and three fixed training seeds; user bootstrap does not resample masks or training data.",
            "All fractions share the same catalog, users and heldout events; not independent test populations.",
            "Per-user ceiling changes actual retention; report actual values, not just nominal percentages.",
            "Masks at different retain fractions are not required to be nested subsets of each other.",
            "Follow-up exploratory robustness after the first sparse result was inspected; no multiple-testing correction.",
            "Missing-feedback sensitivity does not establish denoising of real social relations or arbitrary robustness."]}
    OUTPUT.mkdir(parents=True); atomic_json(OUTPUT / "analysis.json", result)
    (OUTPUT / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Electronics robustness: all planned fractions and masks", "", "| Nominal keep | Mask | Actual keep | MM HR@10 | GCL HR@10 | MM NDCG@10 | GCL NDCG@10 |", "|---|---|---:|---:|---:|---:|---:|"]
    for cell in cell_reports:
        mm, gcl = (cell["methods"][m] for m in METHODS)
        lines.append(f"| {cell['fraction']:.0%} | {cell['mask_seed']} | {cell['actual_retain_fraction']:.2%} | {mm['hr@10']:.6f} | {gcl['hr@10']:.6f} | {mm['ndcg@10']:.6f} | {gcl['ndcg@10']:.6f} |")
    lines += ["", "## Mean over both masks and all three model seeds", "", "| Nominal keep | Metric | GCL minus MM | Paired-user 95% interval |", "|---|---|---:|---|"]
    for fraction, row in aggregated.items():
        for metric, value in row["GCL_minus_MM"].items():
            low, high = value["percentile_ci95"]
            lines.append(f"| {float(fraction):.0%} | {metric} | {value['mean_delta']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += [""] + result["limitations"]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"runs": result["total_runs_including_reused"], "aggregated": aggregated}, indent=2))


if __name__ == "__main__":
    main()
