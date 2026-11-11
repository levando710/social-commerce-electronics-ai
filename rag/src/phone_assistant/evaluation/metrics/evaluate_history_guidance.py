"""Validation-selected train-history content prior, separately from graph/query training."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap
from phone_assistant.recommendation.anchored_joint import normalized_rows, semantic_scores, ranking_scores
from phone_assistant.recommendation.conversational_joint import QuerySupervision
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.training.train_anchored_joint import DEFAULT_OUTPUT, DEFAULT_MODELS, SEEDS, run_directory
from phone_assistant.training.train_recsys_portable import atomic_json

OUTPUT = ROOT / 'artifacts/evaluation/results_electronics_history_guidance_v1'
BETAS = (0.0, 0.1, 0.3)
RANKER_TYPES = ("frozen_text", "residual_detached", "residual_joint", "residual_bidirectional")
TOP_K = 10
CPU_THREADS = 2


def history_profiles(pairs, text_vectors, n_users):
    text = normalized_rows(text_vectors)
    profiles = np.zeros((n_users, text.shape[1]), dtype=np.float32)
    for user, item in np.asarray(pairs):
        profiles[int(user)] += text[int(item)]
    # Normalizing a sum is identical to normalizing its arithmetic mean.
    return normalized_rows(profiles)


def rank_metrics(scores, rows):
    grouped, details = defaultdict(list), []
    for row, values in zip(rows, scores):
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite personalized scores")
        order = np.lexsort((np.arange(len(values)), -values))
        rank = int(np.flatnonzero(order == row["item_index"])[0]) + 1
        grouped[str(row["user_index"])].append(rank)
        details.append({**row, "rank": rank})
    metrics = {"users": len(grouped), "queries": len(details)}
    for key in ("hr@10", "ndcg@10"):
        values = {user: float(np.mean([float(rank <= TOP_K) if key == "hr@10" else 1 / np.log2(rank + 1) if rank <= TOP_K else 0.0 for rank in ranks])) for user, ranks in grouped.items()}
        metrics[key] = float(np.mean(list(values.values()))) if values else None
    return metrics, details


def paired_scores(runs, metric):
    grouped = defaultdict(list)
    for run in runs:
        for row in run["ranks"]:
            rank = row["rank"]
            grouped[str(row["user_index"])].append(float(rank <= TOP_K) if metric == "hr@10" else 1 / np.log2(rank + 1) if rank <= TOP_K else 0.0)
    return {user: float(np.mean(values)) for user, values in grouped.items()}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep first history-guidance selection/evaluation")
    selection_path = DEFAULT_OUTPUT / "selection.json"
    if not selection_path.exists():
        raise ValueError("Wait for the complete validation-selected anchored grid")
    selection = json.loads(selection_path.read_text())
    if selection["training_report_sha256"] != sha256(DEFAULT_OUTPUT / "training_grid.json"):
        raise ValueError("Sealed anchored selection changed")
    threadpool_limits(limits=CPU_THREADS)
    data = InteractionData(ROOT / 'data/external/benchmarks/connected_electronics_v2')
    features, manifest = load_features(ROOT / 'artifacts/cache/electronics_features_v1', data)
    supervision = QuerySupervision.load(data, manifest, ROOT / 'artifacts/cache/electronics_query_vectors_v1')
    profiles = history_profiles(data.pairs["train"], features["text"], data.n_users)
    text = normalized_rows(features["text"])
    base = semantic_scores(supervision.vectors, features["text"])
    rankers, provenance = {"frozen_text": [(None, base)]}, {}
    for method in RANKER_TYPES:
        if method == "frozen_text":
            continue
        rankers[method] = []
        for seed in SEEDS:
            directory = run_directory(DEFAULT_MODELS, selection["alpha"], method, seed)
            training = json.loads((directory / "training.json").read_text())
            path = directory / "embeddings.npz"
            if sha256(path) != training["artifacts_sha256"]["embeddings.npz"]:
                raise ValueError("Anchored vectors changed")
            provenance[str(path.relative_to(ROOT))] = sha256(path)
            with np.load(path, allow_pickle=False) as stored:
                scores = ranking_scores(stored["query_weight"], stored["items"], supervision.vectors, base, selection["alpha"])
            rankers[method].append((seed, scores))
    validation_rows = supervision.rows["validation"]
    val_indices = [row["query_index"] for row in validation_rows]
    val_users = [row["user_index"] for row in validation_rows]
    prior = profiles[val_users] @ text.T
    validation = {}
    for beta in BETAS:
        validation[str(beta)] = {}
        for method, variants in rankers.items():
            values = [rank_metrics(scores[val_indices] + beta * prior, validation_rows)[0]["ndcg@10"] for _, scores in variants]
            validation[str(beta)][method] = float(np.mean(values))
    means = {beta: float(np.mean(list(validation[str(beta)].values()))) for beta in BETAS}
    selected_beta = max(BETAS, key=lambda beta: (means[beta], -beta))
    OUTPUT.mkdir(parents=True)
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "selected_beta": selected_beta,
        "validation_scores": validation, "validation_means": means, "anchored_selection_sha256": sha256(selection_path),
        "plan_sha256": sha256(ROOT / 'data/discovery/HISTORY_GUIDANCE_PLAN_2026_11_08.md'),
        "source_sha256": sha256(Path(__file__)), "embeddings_sha256": provenance,
        "split_manifest_sha256": sha256(data.directory / "manifest.json"), "test_evaluated_before_selection": False}
    atomic_json(OUTPUT / "selection.json", plan)
    test_rows = supervision.rows["test"]; indices = [row["query_index"] for row in test_rows]
    users = [row["user_index"] for row in test_rows]; prior = profiles[users] @ text.T
    arms, comparisons = {}, {}
    for method, variants in rankers.items():
        arms[method] = {}
        for beta in sorted({0.0, selected_beta}):
            runs = []
            for seed, scores in variants:
                metrics, ranks = rank_metrics(scores[indices] + beta * prior, test_rows)
                runs.append({"seed": seed, "metrics": metrics, "ranks": ranks})
            arms[method][str(beta)] = {"runs": runs, "mean": {metric: float(np.mean([r["metrics"][metric] for r in runs])) for metric in ("hr@10", "ndcg@10")}}
        baseline = arms[method]["0.0"]["runs"]
        guided = arms[method][str(selected_beta)]["runs"]
        comparisons[method] = {metric: paired_bootstrap(paired_scores(baseline, metric), paired_scores(guided, metric)) for metric in ("hr@10", "ndcg@10")}
    result = {**plan, "selection_sha256": sha256(OUTPUT / "selection.json"), "arms": arms, "paired_bootstrap": comparisons,
        "limitations": ["Train-positive-review content profiles are proxies, not live visitors' interaction histories.",
            "Common beta chosen by validation; four ranker types weighted equally, seeds averaged within type.",
            "Test cohort previously inspected; exploratory follow-up, no multiple-testing correction.",
            "A positive history prior may hurt explicit current intentions; no adaptive gate tuned using test.",
            "User bootstrap conditional on existing model seeds; no user study or generated-answer metric."]}
    atomic_json(OUTPUT / "analysis.json", result); (OUTPUT / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Train-history content guidance", "", f"Validation-selected common beta={selected_beta}; alpha={selection['alpha']} remains frozen.", "",
        "| Ranker | No history HR@10 | Guided HR@10 | No history NDCG@10 | Guided NDCG@10 |", "|---|---:|---:|---:|---:|"]
    for method, row in arms.items():
        before, after = row["0.0"]["mean"], row[str(selected_beta)]["mean"]
        lines.append(f"| {method} | {before['hr@10']:.6f} | {after['hr@10']:.6f} | {before['ndcg@10']:.6f} | {after['ndcg@10']:.6f} |")
    lines += [""] + result["limitations"]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_beta": selected_beta, "results": {method: row[str(selected_beta)]["mean"] for method, row in arms.items()}}, indent=2))


if __name__ == "__main__":
    main()
