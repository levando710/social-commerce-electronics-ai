"""Verify all anchored checkpoints, then evaluate only the alpha sealed by validation."""
from phone_assistant.common.project_layout import recorded_source
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from phone_assistant.recommendation.anchored_joint import AnchoredJointRanker, VARIANTS, semantic_scores, evaluate_queries
from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores
from phone_assistant.recommendation.conversational_joint import QuerySupervision
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.training.train_anchored_joint import DEFAULT_OUTPUT, DEFAULT_MODELS, CPU_THREADS, EXPORT_ATOL, EXPORT_RTOL, run_directory, seal_selection
from phone_assistant.training.train_recsys import evaluate_vectors, exported
from phone_assistant.training.train_recsys_portable import atomic_json, feature_fingerprint, restore_model

OUTPUT = ROOT / 'artifacts/evaluation/results_electronics_anchored_analysis_v1'
TOP_K = 10


def query_user_scores(runs, metric):
    grouped = defaultdict(list)
    for run in runs:
        for row in run["query_test_ranks"]:
            rank = row["rank"]
            value = float(rank <= TOP_K) if metric == "hr@10" else 1 / np.log2(rank + 1) if rank <= TOP_K else 0.0
            grouped[str(row["user_index"])].append(value)
    return {user: float(np.mean(values)) for user, values in grouped.items()}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep the first post-selection evaluation")
    if not (DEFAULT_OUTPUT / "selection.json").exists():
        raise ValueError("Training must finish and seal alpha selection before any test evaluation")
    torch.set_num_threads(CPU_THREADS)
    grid_path = DEFAULT_OUTPUT / "training_grid.json"
    grid = json.loads(grid_path.read_text())
    selected = seal_selection(grid["runs"], DEFAULT_OUTPUT)
    if grid["test_evaluated"] or selected["test_evaluated_before_selection"]:
        raise ValueError("Selection contract does not isolate test evaluation")
    for name, digest in grid["source_sha256"].items():
        if sha256(recorded_source(name, digest)) != digest or sha256(DEFAULT_OUTPUT / "source" / name) != digest:
            raise ValueError(f"Frozen training source changed: {name}")
    data = InteractionData(ROOT / 'data/external/benchmarks/connected_electronics_v2')
    features, feature_manifest = load_features(ROOT / 'artifacts/cache/electronics_features_v1', data)
    if sha256(data.directory / "manifest.json") != grid["split_manifest_sha256"] or sha256(ROOT / 'artifacts/cache/electronics_features_v1/manifest.json') != grid["feature_manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    supervision = QuerySupervision.load(data, feature_manifest, ROOT / 'artifacts/cache/electronics_query_vectors_v1')
    fingerprint = feature_fingerprint(features)
    base_scores = semantic_scores(supervision.vectors, features["text"])
    verified, runs = [], []
    for run in grid["runs"]:
        alpha, method, seed = run["alpha"], run["method"], run["seed"]
        directory = run_directory(DEFAULT_MODELS, alpha, method, seed)
        for name, digest in run["artifacts_sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError("Checkpoint/export changed")
        model = AnchoredJointRanker(data, features, alpha=alpha, dim=grid["config"]["dim"], layers=grid["config"]["layers"], **VARIANTS[method])
        restore_model(model, torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu"))
        vectors = exported(model)
        weight = model.query_projection.weight.detach().numpy().copy()
        arrays = dict(zip(("users", "items", "bias"), vectors)); arrays["query_weight"] = weight
        errors = {}
        with np.load(directory / "embeddings.npz", allow_pickle=False) as stored:
            for name, value in arrays.items():
                np.testing.assert_allclose(value, stored[name], atol=EXPORT_ATOL, rtol=EXPORT_RTOL)
                errors[name] = float(np.max(np.abs(value - stored[name])))
        validation, _ = evaluate_vectors(*vectors, data, "validation")
        query_validation, _ = evaluate_queries(weight, vectors[1], supervision, base_scores, "validation", alpha)
        if validation != run["validation"] or query_validation != run["query_validation"]:
            raise ValueError("Reconstructed validation differs from sealed grid")
        verified.append({"alpha": alpha, "method": method, "seed": seed, "max_absolute_errors": errors, "validation_reproduced": True})
        if alpha == selected["alpha"]:
            test, ranks = evaluate_vectors(*vectors, data, "test")
            query_test, query_ranks = evaluate_queries(weight, vectors[1], supervision, base_scores, "test", alpha)
            degrees = {}
            for name, cold in (("cold", True), ("warm", False)):
                grouped = defaultdict(list)
                for row in query_ranks:
                    if bool(data.item_degree[row["item_index"]] == 0) == cold:
                        grouped[str(row["user_index"])].append(float(row["rank"] <= TOP_K))
                degrees[name] = {"users": len(grouped), "hr@10": float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None}
            runs.append({"method": method, "alpha": alpha, "seed": seed, "best_epoch": run["best_epoch"],
                "validation": validation, "query_validation": query_validation, "query_train": run["query_train"],
                "test": test, "test_ranks": ranks, "query_test": query_test, "query_test_ranks": query_ranks,
                "query_target_degree": degrees})
    if feature_fingerprint(features) != fingerprint:
        raise ValueError("Evaluation mutated frozen features")
    baseline, baseline_ranks = evaluate_queries(None, None, supervision, base_scores, "test", 0.0)
    baseline_runs = [{"query_test_ranks": baseline_ranks}]
    summary, comparisons = {}, {}
    by_method = {method: [r for r in runs if r["method"] == method] for method in VARIANTS}
    for method, selected_runs in by_method.items():
        summary[method] = {task: {metric: {"mean": float(np.mean([r[task]["all"][metric] if task == "test" else r[task][metric] for r in selected_runs])),
            "std": float(np.std([r[task]["all"][metric] if task == "test" else r[task][metric] for r in selected_runs], ddof=1))}
            for metric in ("hr@10", "ndcg@10")} for task in ("test", "query_test")}
        comparisons[f"{method}_minus_frozen_text"] = {metric: paired_bootstrap(query_user_scores(baseline_runs, metric), query_user_scores(selected_runs, metric)) for metric in ("hr@10", "ndcg@10")}
    for first, second in (("residual_detached", "residual_joint"), ("residual_joint", "residual_bidirectional"), ("residual_detached", "residual_bidirectional")):
        comparisons[f"{second}_minus_{first}"] = {task: {metric: paired_bootstrap(fn(by_method[first], metric), fn(by_method[second], metric))
            for metric in ("hr@10", "ndcg@10")} for task, fn in (("query", query_user_scores), ("recsys", user_scores))}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(Path(__file__)),
        "training_grid_sha256": sha256(grid_path), "selection_sha256": sha256(DEFAULT_OUTPUT / "selection.json"),
        "selected_alpha": selected["alpha"], "verified_checkpoints": verified, "summary": summary,
        "frozen_text_baseline": baseline, "frozen_text_test_ranks": baseline_ranks, "runs": runs, "paired_bootstrap": comparisons,
        "limitations": ["Follow-up development on a previously inspected test cohort; not an untouched confirmatory benchmark.",
            "Alpha/checkpoints selected by validation before any test scoring in this new suite.",
            "Three seeds and 38 simulated query-test users; paired-user intervals conditional on these seeds, no multiplicity correction.",
            "Frozen text score is preserved additively but a nonzero residual can still worsen ranking.",
            "Query retrieval and RecSys ranking do not measure generated-answer quality or user satisfaction."]}
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT / "analysis.json", result)
    (OUTPUT / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Semantic-anchored joint: validation-selected evaluation", "", f"Selected alpha: {selected['alpha']}; reconstructed {len(verified)} checkpoints, test scored only nine selected runs.",
        "", "| Arm | Rec HR@10 | Rec NDCG@10 | Query HR@10 | Query NDCG@10 |", "|---|---:|---:|---:|---:|"]
    for method, row in summary.items():
        values = [row[task][metric]["mean"] for task in ("test", "query_test") for metric in ("hr@10", "ndcg@10")]
        lines.append(f"| {method} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", f"Frozen text query HR@10={baseline['hr@10']:.6f}, NDCG@10={baseline['ndcg@10']:.6f}", ""] + result["limitations"]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_alpha": selected["alpha"], "verified_checkpoints": len(verified), "summary": summary, "frozen_text": baseline}, indent=2))


if __name__ == "__main__":
    main()
