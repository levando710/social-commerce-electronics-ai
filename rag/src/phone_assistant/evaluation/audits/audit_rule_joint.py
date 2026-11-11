"""Rebuild all joint checkpoints and compute paired user uncertainty without model selection."""
from phone_assistant.common.project_layout import recorded_source
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from phone_assistant.evaluation.analysis.analyze_connected_results import paired_bootstrap, user_scores
from phone_assistant.recommendation.conversational_joint import ConversationalJointRanker, QuerySupervision, VARIANTS, query_ranking_vectors, evaluate_query_vectors
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.training.train_recsys import exported
from phone_assistant.training.train_recsys_portable import restore_model, feature_fingerprint


def query_scores(runs, metric):
    values = defaultdict(list)
    for run in runs:
        for row in run["query_test_ranks"]:
            rank = row["rank"]
            values[str(row["user_index"])].append(float(rank <= 10) if metric == "hr@10" else
                1 / np.log2(rank + 1) if rank <= 10 else 0.)
    if any(len(v) != 2 * len(runs) for v in values.values()):
        raise ValueError("Expected two query personas per user per seed")
    return {user: float(np.mean(v)) for user, v in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / 'artifacts/evaluation/results_rule_joint_v1/experiment.json')
    parser.add_argument("--models", type=Path, default=ROOT / 'artifacts/models/joint/rule_phone_v1')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_rule_joint_analysis_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep previous analysis; use a new output")
    torch.set_num_threads(2)
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    for name, digest in experiment["source_sha256"].items():
        if sha256(recorded_source(name, digest)) != digest or sha256(args.experiment.parent / "source" / name) != digest:
            raise ValueError(f"Experiment source changed: {name}")
    data = InteractionData(ROOT / 'data/external/benchmarks/handsets_only_v2')
    features, manifest = load_features(ROOT / 'artifacts/cache/handset_features_v1', data)
    if sha256(data.directory / "manifest.json") != experiment["split_manifest_sha256"] or sha256(ROOT / 'artifacts/cache/handset_features_v1/manifest.json') != experiment["feature_manifest_sha256"]:
        raise ValueError("Frozen input manifests changed")
    from phone_assistant.recommendation.recsys_data import read_jsonl
    from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import load_queries
    source = ROOT / 'artifacts/evaluation/rule_phone_conversations_v1'
    queries = read_jsonl(source / "queries.jsonl")
    if sha256(source / "manifest.json") != experiment["proxy_manifest_sha256"]:
        raise ValueError("Proxy manifest changed")
    vectors = load_queries(queries, source / "queries.jsonl", ROOT / 'artifacts/cache/rule_phone_query_vectors_v1', "text", manifest["text_encoder"])
    supervision = QuerySupervision(data, vectors, queries, read_jsonl(source / "labels.jsonl"))
    fingerprint = feature_fingerprint(features)
    verified, by_method, diagnostics = [], defaultdict(list), []
    for run in experiment["runs"]:
        method, seed = run["method"], run["seed"]
        directory = args.models / method / f"seed_{seed}"
        for name, digest in run["artifacts_sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError("Checkpoint or export changed")
        model = ConversationalJointRanker(data, features, dim=experiment["config"]["dim"],
            layers=experiment["config"]["layers"], **VARIANTS[method])
        restore_model(model, torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu"))
        weight, bias, items = query_ranking_vectors(model)
        arrays = {"users": exported(model)[0], "items": exported(model)[1], "bias": exported(model)[2],
                  "query_weight": weight, "query_bias": bias}
        with np.load(directory / "embeddings.npz", allow_pickle=False) as stored:
            errors = {}
            for name, array in arrays.items():
                np.testing.assert_allclose(array, stored[name], atol=1e-6, rtol=1e-5)
                errors[name] = float(np.max(np.abs(array - stored[name])))
        metrics, ranks = evaluate_query_vectors(weight, bias, items, supervision, "test")
        if metrics != run["query_test"] or ranks != run["query_test_ranks"]:
            raise ValueError("Reconstructed query metrics differ")
        split_metrics = {"test": metrics}
        for split in ("train", "validation"):
            split_metrics[split], _ = evaluate_query_vectors(weight, bias, items, supervision, split)
        cold_query_users = defaultdict(list)
        warm_query_users = defaultdict(list)
        for row in ranks:
            grouped = cold_query_users if data.item_degree[row["item_index"]] == 0 else warm_query_users
            grouped[str(row["user_index"])].append(float(row["rank"] <= 10))
        diagnostics.append({"method": method, "seed": seed, "query_split_metrics": split_metrics,
            "test_by_target_train_degree": {name: {"users": len(group), "hr@10": float(np.mean([np.mean(v) for v in group.values()])) if group else None}
                for name, group in (("cold", cold_query_users), ("warm", warm_query_users))}})
        verified.append({"method": method, "seed": seed, "max_absolute_errors": errors,
            "query_test_reproduced": True, "best_epoch": run["best_epoch"]})
        by_method[method].append(run)
    if fingerprint != feature_fingerprint(features):
        raise ValueError("Verification mutated features")
    comparisons = {}
    for first, second in (("gradient_decoupled", "joint_retrieval"), ("joint_retrieval", "joint_bidirectional"),
                          ("gradient_decoupled", "joint_bidirectional")):
        if first not in by_method or second not in by_method:
            continue
        if sorted(r["seed"] for r in by_method[first]) != sorted(r["seed"] for r in by_method[second]):
            raise ValueError("Mismatched paired training seeds")
        comparisons[f"{second}_minus_{first}"] = {task: {metric: paired_bootstrap(fn(by_method[first], metric),
            fn(by_method[second], metric)) for metric in ("hr@10", "ndcg@10")}
            for task, fn in (("recsys", user_scores), ("query", query_scores))}
    conversation = {}
    users = supervision.user_sets["test"]
    for method, runs in by_method.items():
        grouped = defaultdict(list)
        for run in runs:
            for row in run["test_ranks"]:
                if row["user_index"] in users:
                    grouped[str(row["user_index"])].append(float(row["rank"] <= 10))
        conversation[method] = {"users": len(grouped), "hr@10": float(np.mean([np.mean(v) for v in grouped.values()]))}
    result = {"experiment_sha256": sha256(args.experiment), "source_sha256": sha256(Path(__file__)),
        "verified_checkpoints": verified, "paired_bootstrap": comparisons, "conversation_recsys_test": conversation,
        "query_diagnostics": diagnostics,
        "limitations": ["Paired-user percentile intervals conditional on the existing seeds; not training-data uncertainty.",
            "76 rule-generated test-user goals, two correlated language variants per goal; 294/304 validation+test texts also occur in train. Not human relevance, observed dialogue, or evidence of real preference understanding.",
            "Composite validation checkpoint selection differs from earlier RecSys-only experiments.",
            "Exploratory comparisons, no multiple-testing correction; no parameter selection using this analysis."]}
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Joint query training: checkpoint audit and paired analysis", "",
        f"Reconstructed {len(verified)} checkpoints from frozen inputs; reproduced query test ranks.", "",
        "| Comparison | Task | Metric | Mean delta | Paired-user 95% interval |", "|---|---|---|---:|---|"]
    for comparison, tasks in comparisons.items():
        for task, values in tasks.items():
            for metric, value in values.items():
                lower, upper = value["percentile_ci95"]
                lines.append(f"| {comparison} | {task} | {metric} | {value['mean_delta']:.5f} | [{lower:.5f}, {upper:.5f}] |")
    lines += [""] + result["limitations"]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoints": len(verified), "paired_bootstrap": comparisons, "conversation_recsys_test": conversation}, indent=2))


if __name__ == "__main__":
    main()
