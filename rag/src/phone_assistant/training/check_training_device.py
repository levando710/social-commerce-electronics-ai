"""Exercise each real recommender for one optimizer step; not a benchmark or training run."""
from __future__ import annotations
from phone_assistant.common.project_layout import source_path

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.recommendation.recsys_models import METHODS, make_model

DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_DIM = 16
GRAPH_LAYERS = 2
CPU_THREADS = 2
RANDOM_SEED = 42
LEARNING_RATE = 0.001
REGULARIZATION_WEIGHT = 1e-4
CONTRASTIVE_WEIGHT = 0.05
MEBIBYTE_BYTES = 1024 ** 2
GIBIBYTE_BYTES = 1024 ** 3
DETERMINISTIC_CUBLAS_CONFIG = ":4096:8"
EXIT_SUCCESS = 0
EXIT_CHECK_FAILED = 1


def check_device(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DETERMINISTIC_CUBLAS_CONFIG)
    result = {"scope": __doc__, "requested_device": args.device, "all_passed": False,
        "api_calls": 0, "checkpoint_files_written": 0,
        "packages": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
        "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
        "source_sha256": {name: sha256(source_path(name)) for name in
            ("check_training_device.py", "recsys_models.py", "recsys_data.py")}, "models": []}
    if args.device == "cuda" and not result["cuda_available"]:
        result["error"] = "CUDA unavailable. Verify NVIDIA driver and a CUDA-enabled PyTorch installation. No CPU fallback was used."
        return result
    if args.device == "cuda":
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu"] = {"name": properties.name, "vram_gib": properties.total_memory / GIBIBYTE_BYTES,
            "compute_capability": [properties.major, properties.minor]}
    data = InteractionData(args.split)
    features, _ = load_features(args.features, data)
    result.update({"split_manifest_sha256": sha256(args.split / "manifest.json"),
        "feature_manifest_sha256": sha256(args.features / "manifest.json"),
        "users": data.n_users, "items": data.n_items,
        "config": {"batch_size": args.batch_size, "dim": args.dim, "layers": GRAPH_LAYERS, "threads": CPU_THREADS}})
    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    for method in METHODS:
        torch.manual_seed(RANDOM_SEED)
        generator = np.random.default_rng(RANDOM_SEED)
        model = optimizer = loss = None
        started = time.perf_counter()
        try:
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model = make_model(method, data, features, dim=args.dim, layers=GRAPH_LAYERS,
                reg=REGULARIZATION_WEIGHT, ssl_weight=CONTRASTIVE_WEIGHT).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            batch = data.pairs["train"][:args.batch_size]
            users = torch.as_tensor(batch[:, 0].copy(), dtype=torch.long, device=args.device)
            positives = torch.as_tensor(batch[:, 1].copy(), dtype=torch.long, device=args.device)
            negatives = None if method == "bm3" else torch.as_tensor(
                data.negatives(batch[:, 0], generator), dtype=torch.long, device=args.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(users, positives, negatives)["total"]
            if not torch.isfinite(loss):
                raise ValueError("Non-finite forward loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Non-finite gradient")
            optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize()
            entry = {"method": method, "passed": True, "seconds": time.perf_counter() - started,
                "loss": float(loss.detach().cpu())}
            if args.device == "cuda":
                entry["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / MEBIBYTE_BYTES
            result["models"].append(entry)
        except (RuntimeError, ValueError) as exc:
            result["models"].append({"method": method, "passed": False, "error": str(exc)})
            # An accelerator error may invalidate subsequent work in the same process.
            break
        finally:
            del loss, optimizer, model
    result["all_passed"] = len(result["models"]) == len(METHODS) and all(row["passed"] for row in result["models"])
    result["limitations"] = "One small-batch step on existing phone features only. Does not establish larger-dataset VRAM needs, training quality, exact CUDA resume, joint-head compatibility, or end-to-end speed."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/handsets_only_v2')
    parser.add_argument("--features", type=Path, default=ROOT / 'artifacts/cache/handset_features_v1')
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.dim) <= 0:
        parser.error("Batch size and embedding dimension must be positive")
    if args.output.exists():
        raise FileExistsError("Keep the recorded device check; use a new output")
    result = check_device(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return EXIT_SUCCESS if result["all_passed"] else EXIT_CHECK_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
