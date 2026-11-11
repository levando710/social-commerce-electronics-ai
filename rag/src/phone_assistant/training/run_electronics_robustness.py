"""Resume the predeclared multi-mask suite; never overwrite frozen masks/features/results."""
import json
from pathlib import Path
import subprocess
import sys

from phone_assistant.data.preparation.prepare_electronics_robustness import RETAIN_FRACTIONS, MASK_SEEDS, PERCENT_SCALE, prepare
from phone_assistant.features.rebind_catalog_features import rebind
from phone_assistant.recommendation.recsys_data import ROOT, InteractionData, load_features, sha256
from phone_assistant.training.train_recsys_portable import atomic_json

MODEL_SEEDS = (42, 43, 44)
METHODS = ("graph_multimodal", "graph_multimodal_gcl")
EPOCHS = 100
DIMENSION = 16
GRAPH_LAYERS = 2
BATCH_SIZE = 1024
CPU_THREADS = 2
EVALUATION_INTERVAL = 10
REUSED_FRACTION = 0.5
REUSED_MASK_SEED = 20261108
PLAN_PATH = ROOT / 'data/discovery/ELECTRONICS_EXTENSION_PLAN_2026_11_08.md'
OUTPUT = ROOT / 'artifacts/evaluation/results_electronics_robustness_v1'


def invoke(script, arguments):
    subprocess.run([sys.executable, "-B", str(ROOT / "src/run.py"), Path(script).stem, *map(str, arguments)], check=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/external/benchmarks/connected_electronics_v2'
    source_features = ROOT / 'artifacts/cache/electronics_features_v1'
    plan = {"plan_sha256": sha256(PLAN_PATH), "runner_sha256": sha256(Path(__file__)),
        "fractions": RETAIN_FRACTIONS, "mask_seeds": MASK_SEEDS, "model_seeds": MODEL_SEEDS,
        "methods": METHODS, "epochs": EPOCHS, "dimension": DIMENSION, "batch_size": BATCH_SIZE,
        "source_manifest_sha256": sha256(source / "manifest.json")}
    plan = json.loads(json.dumps(plan))
    plan_path = OUTPUT / "run_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Robustness plan changed; use new output paths")
    atomic_json(plan_path, plan)
    cells = []
    for fraction in RETAIN_FRACTIONS:
        for mask_seed in MASK_SEEDS:
            reused = fraction == REUSED_FRACTION and mask_seed == REUSED_MASK_SEED
            name = "electronics_sparse50_v1" if reused else f"electronics_keep{int(fraction * PERCENT_SCALE)}_mask{mask_seed}_v1"
            split = ROOT / 'data/external/benchmarks' / name
            features = ROOT / 'artifacts/cache' / ("electronics_sparse50_features_v1" if reused else f"{name}_features")
            results = ROOT / 'artifacts/evaluation' / f"results_{name}"
            models = ROOT / 'artifacts/models/recommendation' / name
            if not split.exists():
                prepare(source, split, fraction, mask_seed)
            data = InteractionData(split)
            config = data.manifest["config"]
            if config != {"retain_fraction": fraction, "mask_seed": mask_seed} or data.manifest["parent_manifest_sha256"] != plan["source_manifest_sha256"]:
                raise ValueError("Existing mask differs from the planned cell")
            if not features.exists():
                rebind(source, source_features, split, features)
            load_features(features, data)
            if not reused:
                arguments = ["--split", split, "--features", features, "--models", models, "--output", results,
                    "--methods", *METHODS, "--seeds", *MODEL_SEEDS, "--epochs", EPOCHS, "--dim", DIMENSION,
                    "--layers", GRAPH_LAYERS, "--batch-size", BATCH_SIZE, "--threads", CPU_THREADS,
                    "--eval-every", EVALUATION_INTERVAL, "--device", "cpu"]
                if (results / "run_plan.json").exists():
                    arguments.append("--resume")
                invoke("train_recsys_portable.py", arguments)
            verification = results / "checkpoint_verification.json"
            if not verification.exists():
                invoke("verify_recsys_checkpoints.py", ["--split", split, "--features", features,
                    "--models", models, "--experiment", results / "experiment.json"])
            report = json.loads((results / "experiment.json").read_text())
            verified = json.loads(verification.read_text())
            expected = {(m, s) for m in METHODS for s in MODEL_SEEDS}
            if {(r["method"], r["seed"]) for r in report["runs"]} != expected or len(report["runs"]) != len(expected):
                raise ValueError("Incomplete or duplicate robustness runs")
            if not verified["all_passed"] or verified["runs"] != len(expected) or verified["experiment_sha256"] != sha256(results / "experiment.json"):
                raise ValueError("Checkpoint verification missing or stale")
            cells.append({"fraction": fraction, "mask_seed": mask_seed, "reused": reused,
                "split": str(split.relative_to(ROOT)), "features": str(features.relative_to(ROOT)),
                "experiment": str((results / "experiment.json").relative_to(ROOT)),
                "experiment_sha256": sha256(results / "experiment.json"), "verification_sha256": sha256(verification)})
            atomic_json(OUTPUT / "completed_cells.json", cells)
            print(f"Completed robustness cell {len(cells)}/{len(RETAIN_FRACTIONS) * len(MASK_SEEDS)}: {name}", flush=True)
    atomic_json(OUTPUT / "experiment.json", {**plan, "cells": cells, "complete": True})


if __name__ == "__main__":
    main()
