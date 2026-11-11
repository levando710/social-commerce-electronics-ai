"""CPU integration pilot for rule-simulated dialogue/query/graph learning; no API or human claims."""
from phone_assistant.common.project_layout import source_path
import argparse
import json

import numpy as np
import torch

from phone_assistant.recommendation.conversational_joint import QuerySupervision, VARIANTS
from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import encode_queries, load_queries
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, read_jsonl, sha256
from phone_assistant.training.train_conversational_joint import train_one
from phone_assistant.training.train_recsys_portable import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("encode", "train"))
    parser.add_argument("--resume", action="store_true")
    args_cli = parser.parse_args()
    source = ROOT / 'artifacts/evaluation/rule_phone_conversations_v1'
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError("Rule conversation fixtures changed")
    data = InteractionData(ROOT / 'data/external/benchmarks/handsets_only_v2')
    features_path = ROOT / 'artifacts/cache/handset_features_v1'
    features, feature_manifest = load_features(features_path, data)
    queries = read_jsonl(source / "queries.jsonl")
    work = ROOT / 'artifacts/cache/rule_phone_query_vectors_v1'
    if args_cli.phase == "encode":
        encode_queries(queries, feature_manifest, source / "queries.jsonl", work, "text", 8, 2)
        return
    vectors = load_queries(queries, source / "queries.jsonl", work, "text", feature_manifest["text_encoder"])
    supervision = QuerySupervision(data, vectors, queries, read_jsonl(source / "labels.jsonl"))
    # Predeclared small integration pilot, not a fully trained conversational benchmark.
    args = argparse.Namespace(dim=16, layers=2, epochs=30, batch_size=1024, query_batch_size=32,
        resume=args_cli.resume, max_epochs_this_invocation=0)
    output = ROOT / 'artifacts/evaluation/results_rule_joint_v1'
    models = ROOT / 'artifacts/models/joint/rule_phone_v1'
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    names = ("train_rule_joint.py", "prepare_rule_conversations.py", "train_conversational_joint.py", "conversational_joint.py",
             "recsys_models.py", "recsys_data.py", "train_recsys.py", "train_recsys_portable.py", "evaluate_connected_retrieval.py")
    plan = {"scope": manifest["scope"], "split_manifest_sha256": sha256(data.directory / "manifest.json"),
        "feature_manifest_sha256": sha256(features_path / "manifest.json"), "proxy_manifest_sha256": sha256(source / "manifest.json"),
        "query_manifest_sha256": sha256(work / "text_manifest.json"), "source_sha256": {n: sha256(source_path(n)) for n in names},
        "config": {"dim": 16, "layers": 2, "epochs": 30, "batch_size": 1024, "query_batch_size": 32, "seeds": [42, 43, 44]},
        "packages": {"torch": str(torch.__version__), "numpy": np.__version__},
        "selection": "Fixed helper: mean validation RecSys/query NDCG@10; no test selection. 30 epochs integration pilot only.",
        "limitations": manifest["limitations"]}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_plan.json").exists():
        if not args_cli.resume or json.loads((output / "run_plan.json").read_text()) != plan:
            raise ValueError("Use unchanged --resume or a new run version")
    else:
        atomic_json(output / "run_plan.json", plan)
        (output / "source").mkdir()
        for name in names:
            (output / "source" / name).write_bytes((source_path(name)).read_bytes())
    runs = []
    for variant in VARIANTS:
        for seed in (42, 43, 44):
            runs.append(train_one(variant, seed, data, features, supervision, args,
                models / variant / f"seed_{seed}", {**plan, "method": variant, "seed": seed}))
            atomic_json(output / "completed_runs.json", runs)
    summary = {method: {task: {metric: float(np.mean([r[task]["all"][metric] if task == "test" else r[task][metric]
        for r in runs if r["method"] == method])) for metric in ("hr@10", "ndcg@10")} for task in ("test", "query_test")} for method in VARIANTS}
    atomic_json(output / "experiment.json", {**plan, "feature_manifest": feature_manifest, "summary": summary,
        "query_counts": manifest["counts"], "runs": runs, "not_deployed": True})
    lines = ["# Rule-generated phone joint integration pilot", "", plan["scope"], "",
        "30 epochs, three seeds; not real conversational benefit or directly comparable to 100-epoch RecSys-only results.", "",
        "| Variant | RecSys HR@10 | Query target HR@10 |", "|---|---:|---:|"]
    for method, tasks in summary.items():
        lines.append(f"| {method} | {tasks['test']['hr@10']:.5f} | {tasks['query_test']['hr@10']:.5f} |")
    lines += [""] + plan["limitations"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
