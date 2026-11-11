"""Inference over verified exported vectors; no checkpoint unpickling or refitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, read_jsonl, sha256


class RecommendationService:
    def __init__(self, *, split=ROOT / 'data/fixtures/recommendation', models=ROOT / 'artifacts/models/recommendation',
                 experiment=ROOT / 'artifacts/evaluation/results_recsys/experiment.json',
                 corpus=ROOT / 'artifacts/runtime/catalog/documents.jsonl', method="graph_multimodal_gcl", seed=42):
        self.data = InteractionData(split)
        self.method, self.seed = method, seed
        if (Path(experiment).parent / "INVALIDATED.md").exists():
            raise ValueError("This recommender experiment was invalidated; choose a verified run.")
        report = json.loads(Path(experiment).read_text(encoding="utf-8"))
        if report["split_manifest_sha256"] != sha256(self.data.directory / "manifest.json"):
            raise ValueError("Recommender experiment and current split do not match.")
        run = next((run for run in report["runs"] if run["method"] == method and run["seed"] == seed), None)
        if run is None:
            raise ValueError(f"No completed recommender run for {method}, seed {seed}")
        vector_file = Path(models) / method / f"seed_{seed}" / "embeddings.npz"
        if sha256(vector_file) != run["artifacts_sha256"]["embeddings.npz"]:
            raise ValueError("Exported model vectors changed since the experiment.")
        with np.load(vector_file, allow_pickle=False) as stored:
            self.users, self.items, self.bias = (stored[key].copy() for key in ("users", "items", "bias"))
        if self.users.shape[0] != self.data.n_users or self.items.shape[0] != self.data.n_items:
            raise ValueError("Recommender vector dimensions do not match ID maps.")
        if self.users.shape[1] != self.items.shape[1] or self.bias.shape != (self.data.n_items,):
            raise ValueError("Invalid recommender vector dimensions.")
        if not all(np.isfinite(array).all() for array in (self.users, self.items, self.bias)):
            raise ValueError("Nonfinite recommender vectors.")
        self.documents = {doc["doc_id"]: doc for doc in read_jsonl(corpus)}
        if not set(self.data.item_ids) <= self.documents.keys():
            raise ValueError("Current corpus is missing trained catalog items.")

    def recommend(self, user_index=None, top_k=5):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        if user_index is not None and (not isinstance(user_index, int) or isinstance(user_index, bool)
                                       or not 0 <= user_index < self.data.n_users):
            raise ValueError("user_index must identify a profile in the frozen demo split")
        scores = self.data.item_degree.astype(float) if user_index is None else (
            self.users[user_index] @ self.items.T + self.bias)
        scores = scores.copy()
        if user_index is not None:
            scores[list(self.data.train_seen[user_index])] = -np.inf
        indices = np.lexsort((np.arange(self.data.n_items), -scores))
        rows = []
        for index in indices:
            if not np.isfinite(scores[index]):
                continue
            doc_id = self.data.item_ids[index]
            doc = self.documents[doc_id]
            rows.append({"doc_id": doc_id, "title": doc.get("title"), "brand": doc.get("brand"),
                         "score": float(scores[index]), "product_images": doc.get("product_image_paths", [])[:1]})
            if len(rows) == top_k:
                break
        return {"method": self.method if user_index is not None else "popularity",
                "seed": self.seed if user_index is not None else None,
                "personalized": user_index is not None, "user_index": user_index,
                "context": "train history only; offline demo profile", "results": rows}
