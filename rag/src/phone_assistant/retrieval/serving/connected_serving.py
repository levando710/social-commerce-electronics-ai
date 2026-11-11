"""Serving adapters for the frozen connected corpus and conversational query tower."""
import json
from pathlib import Path
import threading

import numpy as np

from phone_assistant.recommendation.recsys_data import ROOT, sha256


class ConnectedRuntime:
    def __init__(self, path=ROOT / 'artifacts/cache/connected_runtime_v1'):
        self.path = Path(path)
        self.corpus, self.index, self.dense = self.path / "documents.jsonl", self.path / "bm25_index.json", self.path / "dense_index"
        self.split = ROOT / 'data/external/benchmarks/connected_phones_v1'
        self.joint_models = ROOT / 'artifacts/models/joint/conversational_v1'
        self.joint_experiment = ROOT / 'artifacts/evaluation/results_conversational_joint_v1/experiment.json'
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["split_manifest_sha256"] != sha256(self.split / "manifest.json"):
            raise ValueError("Connected runtime split changed")
        for name, digest in manifest["artifacts_sha256"].items():
            if sha256(self.path / name) != digest:
                raise ValueError(f"Connected runtime artifact changed: {name}")

    def recommender(self, method="graph_multimodal_gcl", seed=42, joint=None):
        from phone_assistant.recommendation.recsys_service import RecommendationService
        return RecommendationService(split=self.split, corpus=self.corpus, seed=seed,
            models=self.joint_models if joint else ROOT / 'artifacts/models/recommendation/connected_multimodal_v2',
            experiment=self.joint_experiment if joint else ROOT / 'artifacts/evaluation/results_connected_multimodal_v2/experiment.json',
            method=joint or method)


def pool_query(encoder, query):
    from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import query_chunks
    vectors = encoder.encode(query_chunks(query, "text"), normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
    vector = np.mean(vectors, axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


from phone_assistant.retrieval.search.dense_search import DenseRetriever, rank_vectors, load_encoder


class ConnectedDenseRetriever(DenseRetriever):
    """Pool all query chunks consistently with the offline connected experiment."""
    def search(self, query, *, channel="text", top_k=60):
        from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import query_chunks
        if top_k <= 0 or channel not in self.arrays:
            raise ValueError("Invalid dense channel or top_k")
        with self.lock:
            config = self.manifest["channels"][channel]
            if channel not in self.encoders:
                self.encoders[channel] = load_encoder(config["model"], revision=config["revision"],
                    device=self.device, allow_download=self.allow_download)
            key = (channel, query)
            if key not in self.query_cache:
                if len(self.query_cache) >= 512:
                    self.query_cache.clear()
                vectors = self.encoders[channel].encode(query_chunks(query, channel), normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
                vector = np.mean(vectors, axis=0)
                self.query_cache[key] = vector / max(float(np.linalg.norm(vector)), 1e-12)
            vector = self.query_cache[key]
        return rank_vectors(*self.arrays[channel], vector, top_k)


class ConversationalQueryRanking:
    """Same chunk pooling and learned projection as conversational training."""
    def __init__(self, service, models, experiment, *, dense=None):
        self.service, self.dense = service, dense
        path = Path(models) / service.method / f"seed_{service.seed}" / "embeddings.npz"
        with np.load(path, allow_pickle=False) as stored:
            self.weight, self.bias = stored["query_weight"].copy(), stored["query_bias"].copy()
        self.config = json.loads(Path(experiment).read_text(encoding="utf-8"))["feature_manifest"]["text_encoder"]
        self.items = service.items / np.maximum(np.linalg.norm(service.items, axis=1, keepdims=True), 1e-12)
        self.encoder, self.lock = None, threading.RLock()

    def rank(self, query, top_k=80):
        from phone_assistant.retrieval.search.dense_search import load_encoder
        lock = self.dense.lock if self.dense is not None else self.lock
        with lock:
            if self.dense is not None:
                if "text" not in self.dense.encoders:
                    self.dense.encoders["text"] = load_encoder(self.config["model"], revision=self.config["revision"])
                encoder = self.dense.encoders["text"]
            else:
                if self.encoder is None:
                    self.encoder = load_encoder(self.config["model"], revision=self.config["revision"])
                encoder = self.encoder
            vector = pool_query(encoder, query) @ self.weight.T + self.bias
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.items @ vector
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.service.data.item_ids[i]))[:top_k]
        return [(self.service.data.item_ids[i], float(scores[i])) for i in order]
