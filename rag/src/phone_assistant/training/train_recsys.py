"""Train baselines/typed GCL on frozen train edges; select checkpoints on validation only."""
from __future__ import annotations
from phone_assistant.common.project_layout import source_path

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, load_features, sha256
from phone_assistant.recommendation.recsys_models import METHODS, make_model


def evaluate_vectors(user_vectors, item_vectors, bias, data, split, k_values=(5, 10, 20)):
    events = data.pairs[split]
    if not len(events):
        return {"all": {"users": 0}}, []
    groups = defaultdict(list)
    details = []
    item_indices = np.arange(data.n_items)
    for user, target in events:
        scores = np.asarray(user_vectors[user] @ item_vectors.T + bias, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Model produced nonfinite recommendation scores.")
        blocked = set(data.train_seen[int(user)])
        if split == "test":
            blocked.update(data.validation_seen[int(user)])
        if int(target) in blocked:
            raise ValueError("Held-out target is already present in context.")
        if blocked:
            scores[list(blocked)] = -np.inf
        ranking = np.lexsort((item_indices, -scores))
        rank = int(np.flatnonzero(ranking == target)[0]) + 1
        group = "warm_item" if data.item_degree[target] else "cold_item"
        groups["all"].append(rank)
        groups[group].append(rank)
        details.append({"user_index": int(user), "item_index": int(target), "rank": rank, "group": group})
    result = {}
    for name in ("all", "warm_item", "cold_item"):
        ranks = groups[name]
        metrics = {"users": len(ranks)}
        for k in k_values:
            metrics[f"hr@{k}"] = sum(rank <= k for rank in ranks) / len(ranks) if ranks else None
            metrics[f"ndcg@{k}"] = sum(1 / np.log2(rank + 1) for rank in ranks if rank <= k) / len(ranks) if ranks else None
        result[name] = metrics
    return result, details


def exported(model):
    model.eval()
    with torch.no_grad():
        return tuple(value.detach().cpu().numpy().copy() for value in model.export_vectors())


def train_one(method, seed, data, features, args, model_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(method, data, features, dim=args.dim, layers=args.layers,
                       reg=args.reg, ssl_weight=args.ssl_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pairs = data.pairs["train"]
    users = torch.tensor(pairs[:, 0].copy(), dtype=torch.long)
    positives = torch.tensor(pairs[:, 1].copy(), dtype=torch.long)
    history, best_state, best_key, best_epoch = [], None, None, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        negatives = torch.tensor(data.negatives(pairs[:, 0], rng), dtype=torch.long) if method != "bm3" else None
        loss = model.loss(users, positives, negatives)
        if not torch.isfinite(loss["total"]):
            raise ValueError(f"Nonfinite {method} loss at epoch {epoch}")
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            vectors = exported(model)
            validation, _ = evaluate_vectors(*vectors, data, "validation")
            key = (validation["all"]["ndcg@10"], validation["all"]["hr@10"])
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            history.append({"epoch": epoch, "loss": {name: float(value.detach()) for name, value in loss.items()},
                            "validation": validation})
            if epoch == args.epochs or epoch % 50 == 0:
                print(f"{method} seed={seed} epoch={epoch}, loss={float(loss['total'].detach()):.4f}, val_NDCG@10={key[0]:.4f}", flush=True)
    model.load_state_dict(best_state)
    user_vectors, item_vectors, bias = exported(model)
    train_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    validation, _ = evaluate_vectors(user_vectors, item_vectors, bias, data, "validation")
    test, details = evaluate_vectors(user_vectors, item_vectors, bias, data, "test")
    evaluation_seconds = time.perf_counter() - inference_started
    model_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), model_dir / "checkpoint.pt")
    np.savez_compressed(model_dir / "embeddings.npz", users=user_vectors, items=item_vectors, bias=bias)
    result = {
        "method": method, "seed": seed, "best_epoch": best_epoch,
        "checkpoint_selection": "validation NDCG@10, then HR@10; earliest checkpoint for ties; never test",
        "validation": validation, "test": test, "test_ranks": details,
        "train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "loss_history": history,
        "artifacts_sha256": {name: sha256(model_dir / name) for name in ("checkpoint.pt", "embeddings.npz")},
    }
    if hasattr(model, "schema"):
        result["graph_schema"] = model.schema
    (model_dir / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize(runs):
    summary = {}
    for method in sorted({run["method"] for run in runs}):
        selected = [run for run in runs if run["method"] == method]
        values = {}
        for group in ("all", "warm_item", "cold_item"):
            values[group] = {"users": selected[0]["test"][group]["users"]}
            for key in selected[0]["test"][group]:
                if key == "users":
                    continue
                measurements = [run["test"][group][key] for run in selected]
                values[group][key] = None if measurements[0] is None else {
                    "mean": float(np.mean(measurements)),
                    "std": float(np.std(measurements, ddof=1)) if len(measurements) > 1 else 0.0,
                }
        summary[method] = {"seeds": [run["seed"] for run in selected], "test": values,
                           "mean_train_seconds": float(np.mean([run["train_seconds"] for run in selected]))}
    return summary


def write_report(path, summary, pop, config):
    lines = ["# Thực nghiệm RecSys và đồ thị đa phương thức", "",
        "Đây là kiểm chứng triển khai trên tập điện thoại nhỏ, không phải tái lập kết quả benchmark của các tác giả.",
        "", "## Giao thức", "",
        "- Chỉ train trên cạnh train đã đóng băng. Chọn checkpoint bằng validation, đánh giá test sau khi chọn.",
        "- Full-catalog ranking; loại train/validation đã thấy khi đánh giá test; không lấy mẫu negative cho đánh giá.",
        "- Negative sampling huấn luyện chỉ biết train positives, không nhìn nhãn validation/test.",
        "- Text features chỉ từ metadata allowlist; không dùng review, rating hoặc mức phổ biến.",
        f"- Cấu hình: dim={config['dim']}, layers={config['layers']}, epochs={config['epochs']}, lr={config['learning_rate']}, seeds={config['seeds']}.",
        "- Chỉ 5 người validation và 58 người test. Không dùng kết quả này để khẳng định tính vượt trội hay tính mới.",
        "", "## Kết quả test: mean ± sample std qua các seed", "",
        "| Phương án | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 |", "|---|---:|---:|---:|---:|---:|"]
    keys = ("hr@5", "ndcg@5", "hr@10", "ndcg@10", "hr@20")
    lines.append("| Pop | " + " | ".join(f"{pop['test']['all'][key]:.4f}" for key in keys) + " |")
    for method, values in summary.items():
        lines.append(f"| {method} | " + " | ".join(
            f"{values['test']['all'][key]['mean']:.4f} ± {values['test']['all'][key]['std']:.4f}" for key in keys) + " |")
    lines.extend(["", "## Ablation mô tả", "",
                  "| So sánh | Δ HR@10 | Δ NDCG@10 |", "|---|---:|---:|"])
    for first, second in (("graph_text_only", "graph_multimodal"), ("graph_multimodal", "graph_multimodal_gcl")):
        if first in summary and second in summary:
            delta = [summary[second]["test"]["all"][key]["mean"] - summary[first]["test"]["all"][key]["mean"]
                     for key in ("hr@10", "ndcg@10")]
            lines.append(f"| {second} − {first} | {delta[0]:+.4f} | {delta[1]:+.4f} |")
    lines.extend(["", "## Giới hạn", "",
        "- Các baseline là local reimplementations với SBERT/CLIP và xử lý ảnh thiếu, không phải mã chạy benchmark nguyên bản.",
        "- Đồ thị gồm user, item, metadata text, image và giá trị thông số; chưa mô hình hóa quan hệ xã hội hoặc cây phân cấp specs đầy đủ.",
        "- GCL dùng hai graph views edge dropout và InfoNCE theo loại user/item. Chưa có Co-Guiding hoặc joint loss với RAG.",
        "- Metadata/ảnh là catalog tĩnh transductive, chưa xác minh thời điểm xuất hiện cho từng tương tác.",
        "- Mỗi user chỉ có một cạnh train trong split này; thay seed hoặc vài người test có thể làm kết quả thay đổi mạnh.",
        "- Độ lệch chuẩn qua seed không phải khoảng tin cậy trên toàn bộ quần thể người dùng.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/fixtures/recommendation')
    parser.add_argument("--features", type=Path, default=ROOT / 'artifacts/features/legacy')
    parser.add_argument("--models", type=Path, default=ROOT / 'artifacts/models/recommendation')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_recsys')
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--ssl-weight", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.eval_every, args.dim, args.layers, args.threads) <= 0 or args.learning_rate <= 0 or min(args.reg, args.ssl_weight) < 0:
        parser.error("Invalid training hyperparameters")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Methods and seeds must not contain duplicates")
    if (args.output / "experiment.json").exists():
        raise FileExistsError("Experiment exists; use new --models/--output paths.")
    for method in args.methods:
        for seed in args.seeds:
            if (args.models / method / f"seed_{seed}").exists():
                raise FileExistsError("Checkpoint directory exists; use a new --models directory.")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = InteractionData(args.split)
    if not len(data.pairs["validation"]) or not len(data.pairs["test"]):
        raise ValueError("Need nonempty validation and test; do not select checkpoints on test.")
    features, feature_manifest = load_features(args.features, data)
    config = {key: value for key, value in vars(args).items() if not isinstance(value, Path)}
    runs = []
    args.output.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for seed in args.seeds:
            result = train_one(method, seed, data, features, args, args.models / method / f"seed_{seed}")
            runs.append(result)
            (args.output / "completed_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    summary = summarize(runs)
    pop = json.loads((data.directory / "popularity_metrics.json").read_text(encoding="utf-8"))
    experiment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "split_manifest_sha256": sha256(data.directory / "manifest.json"),
        "feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "features": feature_manifest, "packages": {"torch": torch.__version__, "numpy": np.__version__},
        "platform": platform.platform(), "python": platform.python_version(),
        "source_sha256": {name: sha256(source_path(name)) for name in (
            "recsys_data.py", "recsys_models.py", "train_recsys.py")},
        "summary": summary, "popularity": pop, "runs": runs,
    }
    (args.output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    write_report(args.output / "recsys_evaluation_report.md", summary, pop, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
