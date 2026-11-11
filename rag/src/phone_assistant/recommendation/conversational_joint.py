"""Shared graph/query learning from released simulated user utterances, not target metadata."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phone_assistant.evaluation.metrics.evaluate_connected_retrieval import load_queries
from phone_assistant.recommendation.recsys_data import read_jsonl
from phone_assistant.recommendation.recsys_models import TypedGraphGCL

VARIANTS = {
    "gradient_decoupled": {"retrieval_weight": .1, "alignment_weight": 0., "detach_items": True},
    "joint_retrieval": {"retrieval_weight": .1, "alignment_weight": 0., "detach_items": False},
    "joint_bidirectional": {"retrieval_weight": .1, "alignment_weight": .02, "detach_items": False},
}


class QuerySupervision:
    def __init__(self, data, vectors, queries, labels):
        if len(vectors) != len(queries) or vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Invalid query vectors")
        if len({q["dialogue_id"] for q in queries}) != len(queries):
            raise ValueError("Duplicate dialogue IDs")
        by_user = {row["user_id"]: row for row in labels}
        if len(by_user) != len(labels):
            raise ValueError("Duplicate user goal labels")
        self.vectors = np.asarray(vectors, dtype=np.float32).copy()
        self.rows, self.user_sets = {}, {}
        pair_sets = {name: set(map(tuple, data.pairs[name])) for name in data.pairs}
        for split in ("train", "validation", "test"):
            selected = []
            for index, query in enumerate(queries):
                if query["split"] != split:
                    continue
                label = by_user[query["user_id"]]
                if label["cohort"] != split:
                    raise ValueError("Query persona and canonical user goal split differ")
                user, item = data.maps["users"][query["user_id"]], data.maps["items"][label["item_id"]]
                if (user, item) not in pair_sets[split]:
                    raise ValueError("Query goal is not in its designated frozen split")
                selected.append({"query_index": index, "user_index": user, "item_index": item,
                                 "dialogue_id": query["dialogue_id"]})
            self.rows[split] = selected
            self.user_sets[split] = {r["user_index"] for r in selected}
        if any(self.user_sets[a] & self.user_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Canonical users must not cross query-goal cohorts")

    @classmethod
    def load(cls, data, feature_manifest, work):
        source = data.directory / "agent_queries.jsonl"
        queries = read_jsonl(source)
        vectors = load_queries(queries, source, work, "text", feature_manifest["text_encoder"])
        return cls(data, vectors, queries, read_jsonl(data.directory / "evaluator_labels.jsonl"))

    def batch(self, split, indices):
        rows = [self.rows[split][int(i)] for i in indices]
        return (torch.tensor(self.vectors[[r["query_index"] for r in rows]], dtype=torch.float32),
                torch.tensor([r["user_index"] for r in rows], dtype=torch.long),
                torch.tensor([r["item_index"] for r in rows], dtype=torch.long))


class ConversationalJointRanker(nn.Module):
    def __init__(self, data, features, *, dim=16, layers=2, reg=1e-4, ssl_weight=.05,
                 retrieval_weight=.1, alignment_weight=.02, detach_items=False):
        super().__init__()
        self.core = TypedGraphGCL(data, features, dim=dim, layers=layers, reg=reg, ssl_weight=ssl_weight)
        state = torch.get_rng_state()
        self.query_projection = nn.Linear(features["text"].shape[1], dim)
        torch.set_rng_state(state)
        self.retrieval_weight, self.alignment_weight = retrieval_weight, alignment_weight
        self.detach_items = detach_items

    def query_losses(self, query_features, users, targets):
        user_vectors, items = self.core.forward()
        query = F.normalize(self.query_projection(query_features), dim=-1)
        item_vectors = F.normalize(items.detach() if self.detach_items else items, dim=-1)
        logits = query @ item_vectors.T / .2
        retrieval = F.cross_entropy(logits, targets)
        alignment = retrieval.new_zeros(())
        if self.alignment_weight:
            preference = F.normalize(user_vectors[users], dim=-1) @ F.normalize(items, dim=-1).T / .2
            p, q = logits.softmax(-1), preference.softmax(-1)
            mixture = (p + q) / 2
            alignment = .5 * ((p * (p.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(-1).mean()
                              + (q * (q.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum(-1).mean())
        return {"query_retrieval": retrieval, "bidirectional_js": alignment,
                "total": self.retrieval_weight * retrieval + self.alignment_weight * alignment}

    def loss(self, users, positives, negatives, query_features, query_users, query_targets):
        result = self.core.loss(users, positives, negatives)
        joint = self.query_losses(query_features, query_users, query_targets)
        result["total"] = result["total"] + joint["total"]
        result.update({key: value for key, value in joint.items() if key != "total"})
        return result

    def export_vectors(self):
        return self.core.export_vectors()


def parameter_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def query_ranking_vectors(model):
    model.eval()
    with torch.no_grad():
        _, items, _ = model.export_vectors()
    return (model.query_projection.weight.detach().cpu().numpy().copy(),
            model.query_projection.bias.detach().cpu().numpy().copy(),
            F.normalize(items, dim=-1).detach().cpu().numpy().copy())


def evaluate_query_vectors(weight, bias, items, supervision, split):
    from collections import defaultdict
    values, ranks = defaultdict(list), []
    for row in supervision.rows[split]:
        query = supervision.vectors[row["query_index"]] @ weight.T + bias
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = items @ query
        ranking = np.lexsort((np.arange(len(scores)), -scores))
        rank = int(np.flatnonzero(ranking == row["item_index"])[0]) + 1
        values[row["user_index"]].append(rank)
        ranks.append({**row, "rank": rank})
    result = {"users": len(values), "queries": len(ranks)}
    for k in (5, 10, 20):
        result[f"hr@{k}"] = float(np.mean([np.mean([r <= k for r in rows]) for rows in values.values()])) if ranks else None
        result[f"ndcg@{k}"] = float(np.mean([np.mean([1 / np.log2(r + 1) if r <= k else 0 for r in rows]) for rows in values.values()])) if ranks else None
    return result, ranks
