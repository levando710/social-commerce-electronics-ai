"""Validation-only grid for semantic-anchored graph/query learning; no test scoring during fitting."""
from phone_assistant.common.project_layout import source_path
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from phone_assistant.recommendation.anchored_joint import AnchoredJointRanker, VARIANTS, semantic_scores, evaluate_queries
from phone_assistant.recommendation.conversational_joint import QuerySupervision, parameter_state
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.training.train_recsys import evaluate_vectors, exported
from phone_assistant.training.train_recsys_portable import atomic_json, atomic_checkpoint, feature_fingerprint, peak_process_rss_mib, random_state, restore_random, restore_model

DEFAULT_OUTPUT = ROOT / 'artifacts/evaluation/results_electronics_anchored_joint_v1'
DEFAULT_MODELS = ROOT / 'artifacts/models/joint/electronics_anchored_v1'
ALPHAS = (0.1, 0.3)
SEEDS = (42, 43, 44)
EPOCHS = 100
DIMENSION = 16
LAYERS = 2
BATCH_SIZE = 1024
QUERY_BATCH_SIZE = 32
LEARNING_RATE = 0.001
EVALUATION_INTERVAL = 10
CPU_THREADS = 2
GRADIENT_NORM_LIMIT = 5.0
QUERY_SEED_OFFSET = 9000
VALIDATION_TASK_WEIGHT = 0.5
ALPHA_NAME_SCALE = 100
EXPORT_ATOL, EXPORT_RTOL = 1e-6, 1e-5
SOURCES = ("train_anchored_joint.py", "anchored_joint.py", "conversational_joint.py", "recsys_models.py",
    "recsys_data.py", "train_recsys.py", "train_recsys_portable.py", "evaluate_connected_retrieval.py")


def run_directory(models, alpha, method, seed):
    return models / f"alpha_{round(alpha * ALPHA_NAME_SCALE):03d}" / method / f"seed_{seed}"


def selection_key(rec_validation, query_validation):
    rec, query = float(rec_validation["all"]["ndcg@10"]), float(query_validation["ndcg@10"])
    return (VALIDATION_TASK_WEIGHT * rec + VALIDATION_TASK_WEIGHT * query, query, rec)


def train_one(method, alpha, seed, data, features, supervision, base_scores, args, directory, contract):
    completed_path, progress = directory / "training.json", directory / "progress.pt"
    if completed_path.exists():
        if not args.resume:
            raise FileExistsError("Completed run exists; use unchanged --resume")
        result = json.loads(completed_path.read_text())
        if result["contract"] != contract or any(sha256(directory / n) != digest for n, digest in result["artifacts_sha256"].items()):
            raise ValueError("Completed run changed")
        return result
    if progress.exists() and not args.resume:
        raise FileExistsError("Use --resume for an interrupted run")
    if directory.exists() and not progress.exists() and any(directory.iterdir()):
        raise FileExistsError("Output contains artifacts without a resumable checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator, query_generator = np.random.default_rng(seed), np.random.default_rng(seed + QUERY_SEED_OFFSET)
    model = AnchoredJointRanker(data, features, alpha=alpha, dim=args.dim, layers=args.layers, **VARIANTS[method])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    first_epoch, best_state, best_key, best_epoch, history, elapsed = 1, None, None, None, [], 0.0
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        if saved["contract"] != contract:
            raise ValueError("Resume config/source/environment changed")
        restore_model(model, saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        restore_random(saved["random"], generator); query_generator.bit_generator.state = saved["query_random"]
        first_epoch, best_state, best_key, best_epoch, history, elapsed = (saved["epoch"] + 1,
            saved["best_state"], saved["best_key"], saved["best_epoch"], saved["history"], saved["seconds"])
    fingerprint = feature_fingerprint(features)
    started = time.perf_counter()
    end = min(args.epochs, first_epoch + args.max_epochs_this_invocation - 1) if args.max_epochs_this_invocation else args.epochs
    pairs = data.pairs["train"]
    for epoch in range(first_epoch, end + 1):
        model.train(); order = generator.permutation(len(pairs)); losses = {}
        for begin in range(0, len(pairs), args.batch_size):
            batch = pairs[order[begin:begin + args.batch_size]]
            users, positives = torch.tensor(batch[:, 0]), torch.tensor(batch[:, 1])
            negatives = torch.tensor(data.negatives(batch[:, 0], generator))
            chosen = query_generator.choice(len(supervision.rows["train"]), size=args.query_batch_size,
                replace=len(supervision.rows["train"]) < args.query_batch_size)
            query, query_users, targets = supervision.batch("train", chosen)
            query_indices = [supervision.rows["train"][int(i)]["query_index"] for i in chosen]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(users, positives, negatives, query, query_users, targets, torch.from_numpy(base_scores[query_indices]))
            if not torch.isfinite(loss["total"]):
                raise ValueError("Nonfinite anchored joint loss")
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.core.parameters(), GRADIENT_NORM_LIMIT)
            torch.nn.utils.clip_grad_norm_(model.query_projection.parameters(), GRADIENT_NORM_LIMIT)
            optimizer.step()
            for key, value in loss.items():
                losses[key] = losses.get(key, 0.0) + float(value.detach()) * len(batch) / len(pairs)
        record = {"epoch": epoch, "loss": losses}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            vectors = exported(model)
            validation, _ = evaluate_vectors(*vectors, data, "validation")
            query_validation, _ = evaluate_queries(model.query_projection.weight.detach().numpy(), vectors[1], supervision, base_scores, "validation", alpha)
            key = selection_key(validation, query_validation)
            if best_key is None or key > best_key:
                best_key, best_state, best_epoch = key, parameter_state(model), epoch
            record.update(validation=json.loads(json.dumps(validation)), query_validation=query_validation, selection_score=key[0])
        history.append(record)
        atomic_checkpoint(progress, {"contract": contract, "epoch": epoch, "model": parameter_state(model),
            "optimizer": optimizer.state_dict(), "random": random_state(generator, "cpu"),
            "query_random": query_generator.bit_generator.state, "best_state": best_state, "best_key": best_key,
            "best_epoch": best_epoch, "history": history, "seconds": elapsed + time.perf_counter() - started})
        atomic_json(directory / "progress.json", {"epoch": epoch, "epochs": args.epochs, "best_epoch": best_epoch})
        if epoch % args.eval_every == 0 or epoch == end:
            print(f"alpha={alpha} {method} seed={seed} epoch={epoch}/{args.epochs} loss={losses['total']:.4f}", flush=True)
    if feature_fingerprint(features) != fingerprint:
        raise ValueError("Training mutated frozen features")
    if end < args.epochs:
        return None
    seconds = elapsed + time.perf_counter() - started
    restore_model(model, best_state)
    vectors = exported(model)
    query_weight = model.query_projection.weight.detach().numpy().copy()
    validation, _ = evaluate_vectors(*vectors, data, "validation")
    query_validation, _ = evaluate_queries(query_weight, vectors[1], supervision, base_scores, "validation", alpha)
    query_train, _ = evaluate_queries(query_weight, vectors[1], supervision, base_scores, "train", alpha)
    atomic_checkpoint(directory / "checkpoint.pt", parameter_state(model))
    with (directory / "embeddings.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, users=vectors[0], items=vectors[1], bias=vectors[2], query_weight=query_weight)
    (directory / "embeddings.npz.tmp").replace(directory / "embeddings.npz")
    rebuilt = AnchoredJointRanker(data, features, alpha=alpha, dim=args.dim, layers=args.layers, **VARIANTS[method])
    restore_model(rebuilt, torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu"))
    errors = {}
    for name, expected, actual in [*(zip(("users", "items", "bias"), vectors, exported(rebuilt))),
        ("query_weight", query_weight, rebuilt.query_projection.weight.detach().numpy())]:
        np.testing.assert_allclose(actual, expected, atol=EXPORT_ATOL, rtol=EXPORT_RTOL)
        errors[name] = float(np.max(np.abs(actual - expected)))
    result = {"method": method, "alpha": alpha, "seed": seed, "contract": contract, "best_epoch": best_epoch,
        "validation": validation, "query_validation": query_validation, "query_train": query_train,
        "selection_score": selection_key(validation, query_validation)[0], "loss_history": history,
        "train_seconds": seconds, "peak_process_rss_mib": peak_process_rss_mib(),
        "checkpoint_max_errors": errors, "test_evaluated": False,
        "artifacts_sha256": {name: sha256(directory / name) for name in ("checkpoint.pt", "embeddings.npz")}}
    atomic_json(completed_path, result)
    return result


def seal_selection(runs, output):
    expected = {(alpha, method, seed) for alpha in ALPHAS for method in VARIANTS for seed in SEEDS}
    if len(runs) != len(expected) or {(r["alpha"], r["method"], r["seed"]) for r in runs} != expected:
        raise ValueError("Do not select from an incomplete or duplicated grid")
    means = {str(alpha): float(np.mean([r["selection_score"] for r in runs if r["alpha"] == alpha])) for alpha in ALPHAS}
    selected = max(ALPHAS, key=lambda alpha: (means[str(alpha)], -alpha))
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "alpha": selected,
        "criterion": "Mean validation composite across all three methods and seeds; smaller alpha on ties",
        "validation_means": means, "training_report_sha256": sha256(output / "training_grid.json"),
        "test_evaluated_before_selection": False}
    path = output / "selection.json"
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[key] != result[key] for key in result if key != "created_at_utc"):
            raise ValueError("Sealed alpha selection changed")
        return old
    atomic_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_electronics_v2')
    parser.add_argument("--features", type=Path, default=ROOT / 'artifacts/cache/electronics_features_v1')
    parser.add_argument("--queries", type=Path, default=ROOT / 'artifacts/cache/electronics_query_vectors_v1')
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-invocation", type=int, default=0)
    args = parser.parse_args()
    if args.max_epochs_this_invocation < 0:
        parser.error("Epoch limit cannot be negative")
    args.epochs, args.dim, args.layers = EPOCHS, DIMENSION, LAYERS
    args.batch_size, args.query_batch_size, args.eval_every = BATCH_SIZE, QUERY_BATCH_SIZE, EVALUATION_INTERVAL
    torch.set_num_threads(CPU_THREADS); torch.use_deterministic_algorithms(True)
    data = InteractionData(args.split); features, feature_manifest = load_features(args.features, data)
    supervision = QuerySupervision.load(data, feature_manifest, args.queries)
    if any(not supervision.rows[split] for split in ("train", "validation", "test")):
        raise ValueError("Missing fixed query cohorts")
    base_scores = semantic_scores(supervision.vectors, features["text"])
    plan = {"source_sha256": {name: sha256(source_path(name)) for name in SOURCES},
        "split_manifest_sha256": sha256(args.split / "manifest.json"), "feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "query_manifest_sha256": sha256(args.queries / "text_manifest.json"),
        "research_plan_sha256": sha256(ROOT / 'data/discovery/ELECTRONICS_EXTENSION_PLAN_2026_11_08.md'),
        "config": {"alphas": ALPHAS, "seeds": SEEDS, "variants": VARIANTS, "epochs": EPOCHS, "dim": DIMENSION,
            "layers": LAYERS, "batch_size": BATCH_SIZE, "query_batch_size": QUERY_BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "evaluation_interval": EVALUATION_INTERVAL, "cpu_threads": CPU_THREADS},
        "packages": {"torch": str(torch.__version__), "numpy": np.__version__}}
    plan = json.loads(json.dumps(plan)); args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "run_plan.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != plan:
            raise ValueError("Use --resume with identical source/config or new paths")
    else:
        atomic_json(path, plan); (args.output / "source").mkdir()
        for name in SOURCES:
            (args.output / "source" / name).write_bytes(source_path(name).read_bytes())
    runs = []
    for alpha in ALPHAS:
        for method in VARIANTS:
            for seed in SEEDS:
                contract = {**plan, "alpha": alpha, "method": method, "seed": seed}
                result = train_one(method, alpha, seed, data, features, supervision, base_scores, args,
                    run_directory(args.models, alpha, method, seed), contract)
                if result is None:
                    print("Saved epoch; use the same command with --resume", flush=True)
                    return
                runs.append(result); atomic_json(args.output / "completed_runs.json", runs)
    atomic_json(args.output / "training_grid.json", {**plan, "runs": runs, "test_evaluated": False})
    selection = seal_selection(runs, args.output)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
