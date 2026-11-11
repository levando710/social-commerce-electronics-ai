"""Paired user bootstrap and role-specific error summaries, without selecting models on test."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from phone_assistant.recommendation.recsys_data import ROOT, sha256


def paired_bootstrap(first, second, *, repetitions=2000, seed=20261028):
    if set(first) != set(second) or not first:
        raise ValueError("Paired bootstrap requires the same nonempty user set")
    users = sorted(first)
    differences = np.asarray([second[u] - first[u] for u in users], dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = [float(differences[generator.integers(len(users), size=len(users))].mean()) for _ in range(repetitions)]
    return {"users": len(users), "mean_delta": float(differences.mean()),
            "percentile_ci95": [float(value) for value in np.quantile(samples, [.025, .975])],
            "repetitions": repetitions, "seed": seed,
            "unit": "paired user; average over existing training seeds/personas before resampling"}


def user_scores(runs, metric):
    values = defaultdict(list)
    for run in runs:
        seen = set()
        for row in run["test_ranks"]:
            user = row["user_index"]
            if user in seen:
                raise ValueError("Expected one test target per user per training seed")
            seen.add(user)
            rank = row["rank"]
            value = float(rank <= 10) if metric == "hr@10" else 1 / np.log2(rank + 1) if rank <= 10 else 0.
            values[user].append(value)
    if any(len(v) != len(runs) for v in values.values()):
        raise ValueError("Training seeds do not cover identical test users")
    return {str(user): float(np.mean(value)) for user, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recsys", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_multimodal_v2/experiment.json')
    parser.add_argument("--ir", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_ir_v1/experiment.json')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_analysis_v2')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Analysis exists; use a new output")
    experiment = json.loads(args.recsys.read_text(encoding="utf-8"))
    retrieval = json.loads(args.ir.read_text(encoding="utf-8"))
    if experiment["split_manifest_sha256"] != retrieval["split_manifest_sha256"]:
        raise ValueError("Experiments use different frozen splits")
    by_method = defaultdict(list)
    for run in experiment["runs"]:
        by_method[run["method"]].append(run)
    recsys = {}
    for first, second in (("lightgcn", "vbpr"), ("graph_text_only", "graph_multimodal"), ("graph_multimodal", "graph_multimodal_gcl")):
        if first not in by_method or second not in by_method:
            continue
        if sorted(r["seed"] for r in by_method[first]) != sorted(r["seed"] for r in by_method[second]):
            raise ValueError("Ablation methods have different training seeds")
        recsys[f"{second}_minus_{first}"] = {metric: paired_bootstrap(user_scores(by_method[first], metric),
            user_scores(by_method[second], metric)) for metric in ("hr@10", "ndcg@10")}
    role_summary = {}
    for method, runs in by_method.items():
        role_summary[method] = {}
        roles = runs[0].get("test_by_user_role", {})
        for role in roles:
            role_summary[method][role] = {"users": roles[role]["users"]}
            for metric in ("hr@10", "ndcg@10"):
                measurements = [r["test_by_user_role"][role][metric] for r in runs]
                role_summary[method][role][metric] = {"mean": float(np.mean(measurements)),
                    "std": float(np.std(measurements, ddof=1)) if len(measurements) > 1 else 0.}
    ir_scores = {}
    for method, rows in retrieval["rows"]["test"].items():
        users = defaultdict(list)
        for row in rows:
            users[row["user_id"]].append(row["rank"])
        ir_scores[method] = {user: float(np.mean([1 / np.log2(rank + 1) if rank <= 10 else 0. for rank in ranks]))
                             for user, ranks in users.items()}
    ir = {"hybrid_minus_bm25_ndcg@10": paired_bootstrap(ir_scores["bm25"], ir_scores["hybrid"])}
    bm25_rows = {r["dialogue_id"]: r for r in retrieval["rows"]["test"]["bm25"]}
    changed = {"improved": 0, "worse": 0, "same": 0}
    for row in retrieval["rows"]["test"]["hybrid"]:
        original = bm25_rows[row["dialogue_id"]]["rank"]
        changed["improved" if row["rank"] < original else "worse" if row["rank"] > original else "same"] += 1
    result = {"source_sha256": sha256(Path(__file__)), "recsys_experiment_sha256": sha256(args.recsys),
              "ir_experiment_sha256": sha256(args.ir), "recsys_paired_bootstrap": recsys,
              "recsys_roles": role_summary, "ir_paired_bootstrap": ir, "ir_hybrid_rank_changes": changed,
              "limitations": ["Exploratory paired-user bootstrap conditional on the trained seeds, not a new model-selection criterion.",
                  "Does not model uncertainty from resampling training data or source-dataset selection.",
                  "Conversational targets are simulated; offline first-query ranking is not a user study.",
                  "IR qrels contain one target, not complete relevance judgments."]}
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Phân tích ablation và nhóm người dùng", "",
        "Chênh lệch = mô hình thứ hai trừ mô hình thứ nhất. CI bootstrap theo user, sau khi trung bình qua seed; không dùng để chọn model theo test.", "",
        "| So sánh | Δ HR@10 [CI 95%] | Δ NDCG@10 [CI 95%] |", "|---|---:|---:|"]
    for comparison, values in recsys.items():
        cells = []
        for metric in ("hr@10", "ndcg@10"):
            v = values[metric]
            cells.append(f"{v['mean_delta']:+.4f} [{v['percentile_ci95'][0]:+.4f}, {v['percentile_ci95'][1]:+.4f}]")
        lines.append(f"| {comparison} | " + " | ".join(cells) + " |")
    lines += ["", "## Nhóm có hội thoại mô phỏng", "", "| Mô hình | User | HR@10 mean | NDCG@10 mean |", "|---|---:|---:|---:|"]
    for method, roles in role_summary.items():
        v = roles.get("conversation_benchmark")
        if v:
            lines.append(f"| {method} | {v['users']} | {v['hr@10']['mean']:.4f} | {v['ndcg@10']['mean']:.4f} |")
    lines += ["", "## Giới hạn", ""] + [f"- {item}" for item in result["limitations"]]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"recsys": recsys, "ir": ir, "rank_changes": changed}, indent=2))


if __name__ == "__main__":
    main()
