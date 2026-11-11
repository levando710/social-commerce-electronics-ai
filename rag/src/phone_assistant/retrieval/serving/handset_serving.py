"""Serving configuration for the new handset-only trained models and metadata."""
import json
from pathlib import Path

from phone_assistant.recommendation.recsys_data import ROOT, sha256


class HandsetRuntime:
    def __init__(self, path=ROOT / 'artifacts/cache/handset_runtime_v1'):
        self.path = Path(path)
        self.corpus, self.index, self.dense = self.path / "documents.jsonl", self.path / "bm25_index.json", self.path / "dense_index"
        self.split = ROOT / 'data/external/benchmarks/handsets_only_v2'
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.feature_digest = manifest["feature_manifest_sha256"]
        if manifest["split_manifest_sha256"] != sha256(self.split / "manifest.json"):
            raise ValueError("Handset runtime split changed")
        for name, digest in manifest["artifacts_sha256"].items():
            if sha256(self.path / name) != digest:
                raise ValueError(f"Handset runtime changed: {name}")

    def recommender(self, method="graph_multimodal_gcl", seed=42, joint=None):
        from phone_assistant.recommendation.recsys_service import RecommendationService
        if joint:
            raise ValueError("No query-supervised handset-only model is approved for serving; rule-generated pilot heads are not deployed")
        run = "handsets_lightgcn_v1" if method == "lightgcn" else "handsets_multimodal_v1"
        report = ROOT / 'artifacts/evaluation' / f"results_{run}" / "experiment.json"
        if method != "lightgcn" and json.loads(report.read_text(encoding="utf-8"))["feature_manifest_sha256"] != self.feature_digest:
            raise ValueError("Handset recommender and retrieval use different feature stores")
        return RecommendationService(split=self.split, corpus=self.corpus, seed=seed, method=method,
            models=ROOT / 'artifacts/models/recommendation' / run, experiment=report)
