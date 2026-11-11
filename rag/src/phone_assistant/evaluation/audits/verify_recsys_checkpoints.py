"""Reconstruct every selected checkpoint from frozen inputs and compare inference exports."""
from __future__ import annotations
from phone_assistant.common.project_layout import recorded_source

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, load_features, sha256
from phone_assistant.recommendation.recsys_models import make_model
from phone_assistant.training.train_recsys import exported
from phone_assistant.training.train_recsys_portable import atomic_json, restore_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / 'data/external/benchmarks/connected_phones_v1')
    parser.add_argument("--features", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / 'artifacts/models/recommendation/connected_multimodal_v2')
    parser.add_argument("--experiment", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_multimodal_v2/experiment.json')
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.experiment.parent / "INVALIDATED.md").exists():
        raise ValueError("Experiment has been invalidated; do not verify or distribute it as final")
    report = json.loads(args.experiment.read_text(encoding="utf-8"))
    data = InteractionData(args.split)
    if report["split_manifest_sha256"] != sha256(args.split / "manifest.json"):
        raise ValueError("Experiment split mismatch")
    for name, digest in report["source_sha256"].items():
        if sha256(recorded_source(name, digest)) != digest:
            raise ValueError("Restore the experiment source snapshot before verifying checkpoints")
    features = None
    if report["feature_manifest_sha256"] is not None:
        if args.features is None:
            parser.error("This experiment requires --features")
        features, _ = load_features(args.features, data)
        if report["feature_manifest_sha256"] != sha256(args.features / "manifest.json"):
            raise ValueError("Experiment feature store mismatch")
    torch.set_num_threads(report["config"]["threads"])
    config, verified = report["config"], []
    for run in report["runs"]:
        directory = args.models / run["method"] / f"seed_{run['seed']}"
        for name, digest in run["artifacts_sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Run artifact changed: {name}")
        torch.manual_seed(run["seed"])
        model = make_model(run["method"], data, features, dim=config["dim"], layers=config["layers"],
                           reg=config["reg"], ssl_weight=config["ssl_weight"])
        state = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
        restore_model(model, state)
        errors = {}
        with np.load(directory / "embeddings.npz", allow_pickle=False) as stored:
            for name, value in zip(("users", "items", "bias"), exported(model)):
                np.testing.assert_allclose(value, stored[name], atol=1e-6, rtol=1e-5,
                    err_msg=f"Checkpoint reconstruction mismatch: {run['method']} seed={run['seed']} {name}")
                errors[name] = float(np.max(np.abs(value - stored[name])))
        verified.append({"method": run["method"], "seed": run["seed"], "max_absolute_errors": errors})
    result = {"all_passed": True, "runs": len(verified), "verification": verified,
              "experiment_sha256": sha256(args.experiment), "source_sha256": sha256(Path(__file__)),
              "split_manifest_sha256": report["split_manifest_sha256"],
              "feature_manifest_sha256": report["feature_manifest_sha256"],
              "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "verification_device": "cpu",
              "scope": "Rebuild static graph/features and selected parameters; compare exported inference vectors, no training or test reranking"}
    output = args.output or args.experiment.parent / "checkpoint_verification.json"
    atomic_json(output, result)
    print(json.dumps({"all_passed": True, "runs": len(verified), "output": str(output)}))


if __name__ == "__main__":
    main()
