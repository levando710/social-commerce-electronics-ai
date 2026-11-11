"""Bounded-memory CPU/CUDA experiments with epoch checkpoints and exact same-device resume.

Training artifacts are local tensor/dict checkpoints, loaded with weights_only=True.
Model selection uses validation only. Completed runs are reused without re-evaluating test.
"""
from __future__ import annotations
from phone_assistant.common.project_layout import source_path

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, load_features, sha256
from phone_assistant.recommendation.recsys_models import METHODS, make_model
from phone_assistant.training.train_recsys import evaluate_vectors, exported, summarize


def peak_process_rss_mib():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                 "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                 "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        value = MemoryCounters()
        value.cb = ctypes.sizeof(value)
        if ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(value), value.cb):
            return value.PeakWorkingSetSize / (1024 * 1024)
        return None
    import resource
    import sys
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    # Reconstruct immutable graph/features from their verified inputs instead of
    # copying them into every epoch checkpoint (particularly costly on laptops).
    static_buffers = {"adjacency", "edges", "text", "image", "image_mask"}
    if not set(dict(model.named_buffers())) <= static_buffers:
        raise ValueError("Model has mutable/unknown buffers; checkpoint policy must be extended explicitly")
    parameters = set(dict(model.named_parameters()))
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if key in parameters}


def restore_model(model, state):
    if set(state) != set(dict(model.named_parameters())):
        raise ValueError("Checkpoint parameter schema differs from this model")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != set(model.state_dict()) - set(state):
        raise ValueError("Unexpected checkpoint buffer mismatch")


def feature_fingerprint(features):
    if features is None:
        return None
    digest = hashlib.sha256()
    for key in ("text", "image", "image_mask"):
        values = np.ascontiguousarray(features[key])
        digest.update(f"{key}:{values.shape}:{values.dtype}".encode())
        digest.update(memoryview(values).cast("B"))
    digest.update(json.dumps({key: features[key] for key in ("specs", "spec_hierarchy") if key in features},
                             sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def random_state(generator, device):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "generator": generator.bit_generator.state,
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []}


def restore_random(state, generator):
    random.setstate(state["python"])
    generator.bit_generator.state = state["generator"]
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def fit(method, seed, data, features, args, model_dir, contract):
    completed_path = model_dir / "training.json"
    progress_path = model_dir / "progress.pt"
    if completed_path.exists():
        if not args.resume:
            raise FileExistsError("Run complete; use --resume to reuse it or a new model directory")
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed["contract"] != contract:
            raise ValueError("Completed run contract changed")
        for name, digest in completed["artifacts_sha256"].items():
            if sha256(model_dir / name) != digest:
                raise ValueError(f"Completed artifact changed: {name}")
        return completed
    if progress_path.exists() and not args.resume:
        raise FileExistsError("Interrupted run exists; pass --resume")
    if model_dir.exists() and not progress_path.exists():
        if any(model_dir.iterdir()):
            raise FileExistsError("Incomplete artifacts without an epoch checkpoint; use a new model path")
    model_dir.mkdir(parents=True, exist_ok=True)
    original_features = feature_fingerprint(features)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    generator = np.random.default_rng(seed)
    model = make_model(method, data, features, dim=args.dim, layers=args.layers,
                       reg=args.reg, ssl_weight=args.ssl_weight).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best_state, best_key, best_epoch, first_epoch, prior_seconds = [], None, None, None, 1, 0.0
    if progress_path.exists():
        stored = torch.load(progress_path, map_location=args.device, weights_only=True)
        if stored["contract"] != contract:
            raise ValueError("Cannot resume with different inputs, source, environment, device or hyperparameters")
        restore_model(model, stored["model"])
        optimizer.load_state_dict(stored["optimizer"])
        restore_random(stored["random"], generator)
        history, best_state = stored["history"], stored["best_state"]
        best_key, best_epoch = stored["best_key"], stored["best_epoch"]
        first_epoch, prior_seconds = stored["epoch"] + 1, stored["train_seconds"]
    started = time.perf_counter()
    pairs = data.pairs["train"]
    invocation_end = min(args.epochs, first_epoch + args.max_epochs_this_invocation - 1) if args.max_epochs_this_invocation else args.epochs
    for epoch in range(first_epoch, invocation_end + 1):
        model.train()
        # For full batch retain deterministic pair ordering, including legacy behavior.
        order = generator.permutation(len(pairs)) if args.batch_size and args.batch_size < len(pairs) else np.arange(len(pairs))
        step = args.batch_size or len(pairs)
        losses = {}
        for begin in range(0, len(pairs), step):
            batch = pairs[order[begin:begin + step]]
            users = torch.as_tensor(batch[:, 0].copy(), dtype=torch.long, device=args.device)
            positives = torch.as_tensor(batch[:, 1].copy(), dtype=torch.long, device=args.device)
            negatives = None if method == "bm3" else torch.as_tensor(data.negatives(batch[:, 0], generator), dtype=torch.long, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(users, positives, negatives)
            if not torch.isfinite(loss["total"]):
                raise ValueError(f"Nonfinite {method} loss at epoch {epoch}")
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in loss.items():
                losses[key] = losses.get(key, 0.0) + float(value.detach()) * len(batch) / len(pairs)
        record = {"epoch": epoch, "loss": losses}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation, _ = evaluate_vectors(*exported(model), data, "validation")
            # Keep checkpoints loadable by weights_only: metrics may contain NumPy scalars.
            validation = json.loads(json.dumps(validation))
            key = (validation["all"]["ndcg@10"], validation["all"]["hr@10"])
            if best_key is None or key > best_key:
                best_key, best_epoch, best_state = key, epoch, cpu_state(model)
            record["validation"] = validation
        history.append(record)
        elapsed = prior_seconds + time.perf_counter() - started
        atomic_checkpoint(progress_path, {"contract": contract, "epoch": epoch, "model": cpu_state(model),
            "optimizer": optimizer.state_dict(), "random": random_state(generator, args.device),
            "history": history, "best_state": best_state, "best_key": best_key, "best_epoch": best_epoch,
            "train_seconds": elapsed})
        atomic_json(model_dir / "progress.json", {"epoch": epoch, "epochs": args.epochs,
            "best_epoch": best_epoch, "best_validation_key": best_key, "loss": losses,
            "train_seconds": elapsed, "status": "ready_to_finalize" if epoch == args.epochs else "checkpointed"})
        if epoch % args.eval_every == 0 or epoch == invocation_end:
            print(f"{method} seed={seed} epoch={epoch}/{args.epochs} loss={losses['total']:.5f} saved", flush=True)
    if feature_fingerprint(features) != original_features:
        raise ValueError("Training mutated the shared feature store; run invalid, do not evaluate or reuse it")
    if invocation_end < args.epochs:
        return None
    if best_state is None:
        raise ValueError("No validation-selected checkpoint")
    train_seconds = prior_seconds + time.perf_counter() - started
    restore_model(model, best_state)
    users, items, bias = exported(model)
    evaluation_started = time.perf_counter()
    validation, _ = evaluate_vectors(users, items, bias, data, "validation")
    test, ranks = evaluate_vectors(users, items, bias, data, "test")
    atomic_checkpoint(model_dir / "checkpoint.pt", cpu_state(model))
    vector_path = model_dir / "embeddings.npz"
    with vector_path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, users=users, items=items, bias=bias)
    vector_path.with_suffix(".npz.tmp").replace(vector_path)
    roles_path = data.directory / "user_roles.json"
    role_metrics = {}
    if roles_path.exists():
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
        for role in sorted(set(roles.values())):
            selected = [r for r in ranks if roles[data.user_ids[r["user_index"]]] == role]
            role_metrics[role] = {"users": len(selected)}
            for k in (5, 10, 20):
                role_metrics[role][f"hr@{k}"] = sum(r["rank"] <= k for r in selected) / len(selected) if selected else None
                role_metrics[role][f"ndcg@{k}"] = sum(1 / np.log2(r["rank"] + 1) for r in selected if r["rank"] <= k) / len(selected) if selected else None
    result = {"method": method, "seed": seed, "contract": contract, "best_epoch": best_epoch,
        "checkpoint_format": "named_parameters_only; immutable graph/features reconstructed from verified inputs",
        "checkpoint_selection": "validation NDCG@10, then HR@10; earliest ties; test never selects checkpoint",
        "validation": validation, "test": test, "test_by_user_role": role_metrics, "test_ranks": ranks,
        "train_seconds": train_seconds, "evaluation_seconds": time.perf_counter() - evaluation_started,
        "peak_process_rss_mib": peak_process_rss_mib(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "loss_history": history,
        "artifacts_sha256": {name: sha256(model_dir / name) for name in ("checkpoint.pt", "embeddings.npz")}}
    if hasattr(model, "schema"):
        result["graph_schema"] = model.schema
    atomic_json(completed_path, result)
    atomic_json(model_dir / "progress.json", {"status": "complete", "epoch": args.epochs,
                "best_epoch": best_epoch, "train_seconds": train_seconds})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--features", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / 'artifacts/models/recommendation/connected_phones_cpu_v1')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_phones_cpu_v1')
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["lightgcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024, help="0 = full-batch; smaller batches still propagate the full sparse graph")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--ssl-weight", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-invocation", type=int, default=0, help="Stop cleanly after this many epochs; 0 = all remaining epochs")
    args = parser.parse_args()
    if min(args.epochs, args.eval_every, args.dim, args.layers, args.threads) <= 0 or min(args.batch_size, args.max_epochs_this_invocation, args.reg, args.ssl_weight) < 0 or args.learning_rate <= 0:
        parser.error("Invalid training hyperparameters")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate methods or seeds")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = InteractionData(args.split)
    if not len(data.pairs["validation"]) or not len(data.pairs["test"]):
        raise ValueError("Need validation and test; cannot select on test")
    features, feature_manifest = None, None
    if any(method != "lightgcn" for method in args.methods):
        if args.features is None:
            parser.error("Multimodal/typed models require verified --features")
        features, feature_manifest = load_features(args.features, data)
    config = {key: value for key, value in vars(args).items() if not isinstance(value, Path) and key not in ("resume", "max_epochs_this_invocation")}
    sources = {name: sha256(source_path(name)) for name in
               ("train_recsys_portable.py", "train_recsys.py", "recsys_models.py", "recsys_data.py")}
    plan = {"config": config, "split_manifest_sha256": sha256(args.split / "manifest.json"),
            "feature_manifest_sha256": sha256(args.features / "manifest.json") if feature_manifest else None,
            "source_sha256": sources, "packages": {"torch": str(torch.__version__), "numpy": np.__version__}}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "run_plan.json"
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError("Run plan exists: use --resume or a new output directory")
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Run plan changed: restore original source/config or start a new experiment")
    else:
        atomic_json(plan_path, plan)
        source_dir = args.output / "source"
        source_dir.mkdir(exist_ok=True)
        for name in sources:
            (source_dir / name).write_bytes((source_path(name)).read_bytes())
    runs = []
    for method in args.methods:
        for seed in args.seeds:
            contract = {**plan, "method": method, "seed": seed}
            result = fit(method, seed, data, features, args, args.models / method / f"seed_{seed}", contract)
            if result is None:
                print("Stopped after a saved epoch. Run the same command with --resume to continue.", flush=True)
                return
            runs.append(result)
            atomic_json(args.output / "completed_runs.json", runs)
    summary = summarize(runs)
    pop = json.loads((data.directory / "popularity_metrics.json").read_text(encoding="utf-8"))
    experiment = {**plan, "created_at_utc": datetime.now(timezone.utc).isoformat(), "features": feature_manifest,
                  "platform": platform.platform(), "summary": summary, "popularity": pop, "runs": runs}
    atomic_json(args.output / "experiment.json", experiment)
    lines = ["# Thực nghiệm RecSys với dữ liệu bổ sung", "",
        f"User: {data.n_users}; catalog: {data.n_items}; train: {len(data.pairs['train'])}; validation: {len(data.pairs['validation'])}; test: {len(data.pairs['test'])}.",
        "", "Chọn checkpoint bằng validation; full-catalog ranking, loại item đã thấy, báo cáo cả cold-item.",
        "", "| Mô hình | HR@10 | NDCG@10 |", "|---|---:|---:|",
        f"| Pop | {pop['test']['all']['hr@10']:.4f} | {pop['test']['all']['ndcg@10']:.4f} |"]
    for method, values in summary.items():
        metrics = values["test"]["all"]
        lines.append(f"| {method} | {metrics['hr@10']['mean']:.4f} ± {metrics['hr@10']['std']:.4f} | {metrics['ndcg@10']['mean']:.4f} ± {metrics['ndcg@10']['std']:.4f} |")
    lines += ["", "Các baseline là bản triển khai nội bộ, không phải tái lập benchmark chính thức.",
        "Kết quả là dự đoán review tích cực theo từng user; không phải thử nghiệm người dùng thật hay hội thoại tương tác.",
        "Các nhóm user hội thoại và user review bổ sung được báo cáo riêng trong experiment.json.",
        "Thu nhỏ batch không loại bỏ bộ nhớ đồ thị: mỗi batch vẫn lan truyền toàn đồ thị thưa.",
        "Độ lệch chuẩn qua seed không phải khoảng tin cậy theo quần thể người dùng.", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
