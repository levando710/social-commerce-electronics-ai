"""Shared graph ranking objectives with metadata reconstruction supervision.

No conversational labels are invented: this is a catalog-alignment proxy,
not evidence of trained conversational recommendation or LLM fine-tuning.
"""
import torch
from torch import nn
from torch.nn import functional as F

from phone_assistant.recommendation.recsys_models import TypedGraphGCL


class JointGraphRanker(nn.Module):
    def __init__(self, data, features, dim=32, retrieval_weight=.1, alignment_weight=.02):
        super().__init__()
        self.core = TypedGraphGCL(data, features, dim=dim)
        state = torch.get_rng_state()
        self.query_projection = nn.Linear(features['text'].shape[1], dim)
        torch.set_rng_state(state)  # paired graph augmentation seeds across ablations
        self.register_buffer('metadata', torch.tensor(features['text'], dtype=torch.float32))
        self.retrieval_weight, self.alignment_weight = retrieval_weight, alignment_weight

    def loss(self, users, positives, negatives):
        result = self.core.loss(users, positives, negatives)
        user_vectors, items = self.core.forward()
        # Positive catalog metadata is available ONLY for train interactions.
        query = F.normalize(self.query_projection(self.metadata[positives]),dim=-1)
        item_vectors = F.normalize(items,dim=-1)
        retrieval_logits = query @ item_vectors.T / .2
        preference_logits = F.normalize(user_vectors[users],dim=-1) @ item_vectors.T / .2
        retrieval = F.cross_entropy(retrieval_logits,positives)
        p, q = retrieval_logits.softmax(-1), preference_logits.softmax(-1)
        mixture = (p + q) / 2
        alignment = .5 * ((p * (p.clamp_min(1e-12).log()-mixture.clamp_min(1e-12).log())).sum(-1).mean()
                         + (q * (q.clamp_min(1e-12).log()-mixture.clamp_min(1e-12).log())).sum(-1).mean())
        result['metadata_retrieval'] = retrieval
        result['bidirectional_js'] = alignment
        result['total'] = result['total'] + self.retrieval_weight*retrieval + self.alignment_weight*alignment
        return result

    def export_vectors(self):
        return self.core.export_vectors()


class JointQueryRanking:
    """Serve the trained query tower with the same normalized item representations."""
    def __init__(self, service, models, experiment):
        import json
        import numpy as np
        from pathlib import Path

        self.service = service
        path = Path(models) / service.method / f'seed_{service.seed}' / 'embeddings.npz'
        with np.load(path,allow_pickle=False) as stored:
            self.weight,self.bias=stored['query_weight'].copy(),stored['query_bias'].copy()
        self.config=json.loads(Path(experiment).read_text(encoding='utf-8'))['feature_manifest']['text_encoder']
        self.items=service.items/np.maximum(np.linalg.norm(service.items,axis=1,keepdims=True),1e-12)
        self.encoder=None
        self.lock=__import__('threading').RLock()

    def rank(self,query,top_k=80):
        import numpy as np
        from phone_assistant.retrieval.search.dense_search import load_encoder

        with self.lock:
            if self.encoder is None:
                self.encoder=load_encoder(self.config['model'],revision=self.config['revision'])
            feature=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
            vector=feature@self.weight.T+self.bias
        vector=vector/max(float(np.linalg.norm(vector)),1e-12)
        scores=self.items@vector
        order=np.lexsort((np.arange(len(scores)),-scores))[:top_k]
        return [(self.service.data.item_ids[index],float(scores[index])) for index in order]
