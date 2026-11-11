"""Train a bounded graph residual while retaining frozen query/catalog semantic similarity."""
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from phone_assistant.recommendation.recsys_models import TypedGraphGCL

VARIANTS = {
    "residual_detached": {"detach_items": True, "alignment_weight": 0.0},
    "residual_joint": {"detach_items": False, "alignment_weight": 0.0},
    "residual_bidirectional": {"detach_items": False, "alignment_weight": 0.02},
}
QUERY_CE_WEIGHT = 0.1
DISTILLATION_WEIGHT = 1.0
SOFTMAX_TEMPERATURE = 0.2
NORMALIZATION_EPSILON = 1e-12
RANKING_CUTOFFS = (5, 10, 20)


def normalized_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite feature matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), NORMALIZATION_EPSILON)


def semantic_scores(query_vectors, catalog_vectors):
    return normalized_rows(query_vectors) @ normalized_rows(catalog_vectors).T


class AnchoredJointRanker(nn.Module):
    def __init__(self, data, features, *, alpha, dim=16, layers=2, detach_items=False, alignment_weight=0.0):
        super().__init__()
        if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(alignment_weight) or alignment_weight < 0:
            raise ValueError("Invalid residual/alignment weights")
        self.core = TypedGraphGCL(data, features, dim=dim, layers=layers)
        state = torch.get_rng_state()
        self.query_projection = nn.Linear(features["text"].shape[1], dim, bias=False)
        torch.set_rng_state(state)
        with torch.no_grad():
            self.query_projection.weight.copy_(self.core.text_projection.weight)
        self.alpha, self.detach_items, self.alignment_weight = alpha, detach_items, alignment_weight

    def query_losses(self, query_vectors, users, targets, frozen_scores):
        user_vectors, items = self.core.forward()
        candidate_items = items.detach() if self.detach_items else items
        query = F.normalize(self.query_projection(query_vectors), dim=-1)
        residual = query @ F.normalize(candidate_items, dim=-1).T
        logits = (frozen_scores.detach() + self.alpha * residual) / SOFTMAX_TEMPERATURE
        teacher = (frozen_scores.detach() / SOFTMAX_TEMPERATURE).softmax(-1)
        cross_entropy = F.cross_entropy(logits, targets)
        distillation = F.kl_div(logits.log_softmax(-1), teacher, reduction="batchmean")
        alignment = logits.new_zeros(())
        if self.alignment_weight:
            preference = F.normalize(user_vectors[users], dim=-1) @ F.normalize(items, dim=-1).T
            first, second = logits.softmax(-1), (preference / SOFTMAX_TEMPERATURE).softmax(-1)
            mixture = (first + second) / 2
            alignment = ((first * (first.clamp_min(NORMALIZATION_EPSILON).log() - mixture.clamp_min(NORMALIZATION_EPSILON).log())).sum(-1).mean()
                + (second * (second.clamp_min(NORMALIZATION_EPSILON).log() - mixture.clamp_min(NORMALIZATION_EPSILON).log())).sum(-1).mean()) / 2
        return {"query_ce": cross_entropy, "semantic_kl": distillation, "bidirectional_js": alignment,
            "total": QUERY_CE_WEIGHT * cross_entropy + DISTILLATION_WEIGHT * distillation + self.alignment_weight * alignment}

    def loss(self, users, positives, negatives, query_vectors, query_users, targets, frozen_scores):
        result = self.core.loss(users, positives, negatives)
        query = self.query_losses(query_vectors, query_users, targets, frozen_scores)
        result["total"] = result["total"] + query["total"]
        result.update({key: value for key, value in query.items() if key != "total"})
        return result

    def export_vectors(self):
        return self.core.export_vectors()


def ranking_scores(weight, items, query_vectors, frozen_scores, alpha):
    projected = normalized_rows(np.asarray(query_vectors) @ weight.T)
    return np.asarray(frozen_scores) + alpha * (projected @ normalized_rows(items).T)


def evaluate_queries(weight, items, supervision, frozen_scores, split, alpha):
    selected = supervision.rows[split]
    indices = [row["query_index"] for row in selected]
    if weight is None:
        scores = frozen_scores[indices]
    else:
        scores = ranking_scores(weight, items, supervision.vectors[indices], frozen_scores[indices], alpha)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite query ranking scores")
    values, ranks = defaultdict(list), []
    for row, score in zip(selected, scores):
        ranking = np.lexsort((np.arange(len(score)), -score))
        rank = int(np.flatnonzero(ranking == row["item_index"])[0]) + 1
        values[row["user_index"]].append(rank)
        ranks.append({**row, "rank": rank})
    result = {"users": len(values), "queries": len(ranks)}
    for cutoff in RANKING_CUTOFFS:
        result[f"hr@{cutoff}"] = float(np.mean([np.mean([rank <= cutoff for rank in rows]) for rows in values.values()])) if ranks else None
        result[f"ndcg@{cutoff}"] = float(np.mean([np.mean([1 / np.log2(rank + 1) if rank <= cutoff else 0 for rank in rows]) for rows in values.values()])) if ranks else None
    return result, ranks
