"""Checkpointed ablation using train-only simulated first-query/user/target supervision."""
from __future__ import annotations
from phone_assistant.common.project_layout import source_path

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from phone_assistant.recommendation.conversational_joint import ConversationalJointRanker, QuerySupervision, VARIANTS, parameter_state, query_ranking_vectors, evaluate_query_vectors
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.training.train_recsys import exported, evaluate_vectors
from phone_assistant.training.train_recsys_portable import atomic_json, atomic_checkpoint, feature_fingerprint, peak_process_rss_mib, random_state, restore_random, restore_model


def train_one(name, seed, data, features, supervision, args, directory, contract):
    if (directory / "training.json").exists():
        if not args.resume:
            raise FileExistsError("Completed joint run exists; pass --resume or use a new directory")
        result = json.loads((directory / "training.json").read_text(encoding="utf-8"))
        if result["contract"] != contract or any(sha256(directory / n) != d for n, d in result["artifacts_sha256"].items()):
            raise ValueError("Completed joint run changed")
        return result
    progress = directory / "progress.pt"
    if progress.exists() and not args.resume:
        raise FileExistsError("Interrupted joint run exists; pass --resume")
    if directory.exists() and not progress.exists() and any(directory.iterdir()):
        raise FileExistsError("Artifacts without a joint checkpoint; use a new model directory")
    directory.mkdir(parents=True, exist_ok=True)
    original_features = feature_fingerprint(features)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator, query_generator = np.random.default_rng(seed), np.random.default_rng(seed + 9000)
    model = ConversationalJointRanker(data, features, dim=args.dim, layers=args.layers, **VARIANTS[name])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    first_epoch, best_state, best_key, best_epoch, history, elapsed_before = 1, None, None, None, [], 0.
    if progress.exists():
        checkpoint = torch.load(progress, map_location="cpu", weights_only=True)
        if checkpoint["contract"] != contract:
            raise ValueError("Joint resume contract changed")
        restore_model(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_random(checkpoint["random"], generator)
        query_generator.bit_generator.state = checkpoint["query_random"]
        first_epoch, best_state, best_key, best_epoch, history, elapsed_before = (
            checkpoint["epoch"] + 1, checkpoint["best_state"], checkpoint["best_key"],
            checkpoint["best_epoch"], checkpoint["history"], checkpoint["seconds"])
    started = time.perf_counter()
    end = min(args.epochs, first_epoch + args.max_epochs_this_invocation - 1) if args.max_epochs_this_invocation else args.epochs
    pairs = data.pairs["train"]
    for epoch in range(first_epoch, end + 1):
        model.train()
        order = generator.permutation(len(pairs))
        losses = {}
        for begin in range(0, len(pairs), args.batch_size):
            batch = pairs[order[begin:begin + args.batch_size]]
            users, positive = torch.tensor(batch[:, 0]), torch.tensor(batch[:, 1])
            negative = torch.tensor(data.negatives(batch[:, 0], generator))
            selected = query_generator.choice(len(supervision.rows["train"]), size=args.query_batch_size,
                                              replace=len(supervision.rows["train"]) < args.query_batch_size)
            query_features, query_users, targets = supervision.batch("train", selected)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(users, positive, negative, query_features, query_users, targets)
            if not torch.isfinite(loss["total"]):
                raise ValueError("Nonfinite conversational joint loss")
            loss["total"].backward()
            # Separate clipping avoids indirect head-gradient feedback in the detached control.
            torch.nn.utils.clip_grad_norm_(model.core.parameters(), 5.)
            torch.nn.utils.clip_grad_norm_(model.query_projection.parameters(), 5.)
            optimizer.step()
            for key, value in loss.items():
                losses[key] = losses.get(key, 0.) + float(value.detach()) * len(batch) / len(pairs)
        record = {"epoch": epoch, "loss": losses}
        if epoch % 10 == 0 or epoch == args.epochs:
            rec_validation, _ = evaluate_vectors(*exported(model), data, "validation")
            query_validation, _ = evaluate_query_vectors(*query_ranking_vectors(model), supervision, "validation")
            score = .5 * rec_validation["all"]["ndcg@10"] + .5 * query_validation["ndcg@10"]
            key = (float(score), query_validation["ndcg@10"], float(rec_validation["all"]["ndcg@10"]))
            if best_key is None or key > best_key:
                best_key, best_state, best_epoch = key, parameter_state(model), epoch
            record.update(validation=json.loads(json.dumps(rec_validation)), query_validation=query_validation,
                          selection_score=float(score))
        history.append(record)
        atomic_checkpoint(progress, {"contract": contract, "epoch": epoch, "model": parameter_state(model),
            "optimizer": optimizer.state_dict(), "random": random_state(generator, "cpu"),
            "query_random": query_generator.bit_generator.state, "best_state": best_state,
            "best_key": best_key, "best_epoch": best_epoch, "history": history,
            "seconds": elapsed_before + time.perf_counter() - started})
        atomic_json(directory / "progress.json", {"epoch": epoch, "epochs": args.epochs, "loss": losses, "best_epoch": best_epoch})
        if epoch % 10 == 0 or epoch == end:
            print(f"{name} seed={seed} epoch={epoch}/{args.epochs} loss={losses['total']:.4f}", flush=True)
    if feature_fingerprint(features) != original_features:
        raise ValueError("Joint training mutated the shared feature store")
    if end < args.epochs:
        return None
    seconds = elapsed_before + time.perf_counter() - started
    restore_model(model, best_state)
    vectors = exported(model)
    weight, bias, item_query_vectors = query_ranking_vectors(model)
    rec_validation, _ = evaluate_vectors(*vectors, data, "validation")
    query_validation, _ = evaluate_query_vectors(weight, bias, item_query_vectors, supervision, "validation")
    test, ranks = evaluate_vectors(*vectors, data, "test")
    query_test, query_ranks = evaluate_query_vectors(weight, bias, item_query_vectors, supervision, "test")
    atomic_checkpoint(directory / "checkpoint.pt", parameter_state(model))
    with (directory / "embeddings.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, users=vectors[0], items=vectors[1], bias=vectors[2],
                            query_weight=weight, query_bias=bias)
    (directory / "embeddings.npz.tmp").replace(directory / "embeddings.npz")
    reconstructed = ConversationalJointRanker(data, features, dim=args.dim, layers=args.layers, **VARIANTS[name])
    restore_model(reconstructed, torch.load(directory / "checkpoint.pt", weights_only=True))
    for a, b in zip(exported(reconstructed), vectors):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    for a, b in zip(query_ranking_vectors(reconstructed), (weight, bias, item_query_vectors)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    result = {"method": name, "seed": seed, "contract": contract, "best_epoch": best_epoch,
        "validation": rec_validation, "query_validation": query_validation, "test": test,
        "test_ranks": ranks, "query_test": query_test, "query_test_ranks": query_ranks,
        "loss_history": history, "train_seconds": seconds, "peak_process_rss_mib": peak_process_rss_mib(),
        "checkpoint_verified": True, "graph_schema": model.core.schema,
        "artifacts_sha256": {n: sha256(directory / n) for n in ("checkpoint.pt", "embeddings.npz")}}
    atomic_json(directory / "training.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--features", type=Path, default=ROOT / 'artifacts/cache/connected_features_v1')
    parser.add_argument("--queries", type=Path, default=ROOT / 'artifacts/cache/connected_query_vectors_v1')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_conversational_joint_v1')
    parser.add_argument("--models", type=Path, default=ROOT / 'artifacts/models/joint/conversational_v1')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-invocation", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.dim, args.layers, args.batch_size, args.query_batch_size) <= 0 or args.max_epochs_this_invocation < 0:
        parser.error("Invalid processing limits")
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate variants or seeds")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    data = InteractionData(args.split)
    features, feature_manifest = load_features(args.features, data)
    supervision = QuerySupervision.load(data, feature_manifest, args.queries)
    if any(not supervision.rows[name] for name in ("train", "validation", "test")):
        raise ValueError("Need query labels in all three fixed cohorts")
    sources = ("train_conversational_joint.py", "conversational_joint.py", "recsys_models.py", "recsys_data.py",
               "train_recsys.py", "train_recsys_portable.py", "evaluate_connected_retrieval.py")
    plan = {"split_manifest_sha256": sha256(args.split / "manifest.json"),
        "feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "query_manifest_sha256": sha256(args.queries / "text_manifest.json"),
        "source_sha256": {n: sha256(source_path(n)) for n in sources},
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path) and k not in ("resume", "max_epochs_this_invocation")},
        "variants": VARIANTS, "packages": {"torch": str(torch.__version__), "numpy": np.__version__},
        "checkpoint_selection": "0.5 RecSys validation NDCG@10 + 0.5 query validation user-macro NDCG@10; then query, then RecSys; earliest ties"}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "run_plan.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing run plan: use --resume with unchanged source/config or new paths")
    else:
        atomic_json(path, plan)
        (args.output / "source").mkdir()
        for name in sources:
            (args.output / "source" / name).write_bytes((source_path(name)).read_bytes())
    runs = []
    for name in args.variants:
        for seed in args.seeds:
            result = train_one(name, seed, data, features, supervision, args, args.models / name / f"seed_{seed}",
                               {**plan, "method": name, "seed": seed})
            if result is None:
                print("Saved epoch; continue with the same command and --resume", flush=True)
                return
            runs.append(result)
            atomic_json(args.output / "completed_runs.json", runs)
    summary = {}
    for name in args.variants:
        selected = [r for r in runs if r["method"] == name]
        summary[name] = {}
        for task in ("test", "query_test"):
            task_values = [r[task]["all"] if task == "test" else r[task] for r in selected]
            summary[name][task] = {key: {"mean": float(np.mean([v[key] for v in task_values])),
                "std": float(np.std([v[key] for v in task_values], ddof=1)) if len(selected) > 1 else 0.}
                for key in ("hr@10", "ndcg@10")}
    experiment = {**plan, "created_at_utc": datetime.now(timezone.utc).isoformat(), "feature_manifest": feature_manifest,
        "summary": summary, "runs": runs,
        "query_counts": {split: {"queries": len(rows), "users": len(supervision.user_sets[split])} for split, rows in supervision.rows.items()},
        "scope": "LLM-simulated first-query supervision grounded in review histories; no human dialogue claim; generator/encoders frozen",
        "limitations": ["One target per query, incomplete relevance labels.", "User-goal cohorts are disjoint but all prior user histories are transductive training context.",
            "Query metrics are retrieval-only, not answer faithfulness or interactive user benefit.",
            "Joint objective is prototype local architecture, not an official benchmark reproduction."]}
    atomic_json(args.output / "experiment.json", experiment)
    lines = ["# Joint training với câu hỏi mô phỏng gắn đúng user và target", "", experiment["scope"], "",
        "Control vẫn train query head; item vectors detach khỏi query loss. Hai nhánh joint cho query loss cập nhật graph, thêm JS ở nhánh bidirectional.",
        "", "| Variant | RecSys HR@10 | RecSys NDCG@10 | Query HR@10 | Query NDCG@10 |", "|---|---:|---:|---:|---:|"]
    for name, tasks in summary.items():
        values = [tasks[t][k]["mean"] for t in ("test", "query_test") for k in ("hr@10", "ndcg@10")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", plan["checkpoint_selection"], ""] + [f"- {v}" for v in experiment["limitations"]]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
