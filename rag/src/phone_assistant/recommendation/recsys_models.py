"""Research implementations of LightGCN, VBPR, BM3 and a typed graph + GCL.

These are local adaptations to the frozen phone subset and SBERT/CLIP features,
not reproductions of the authors' published benchmark scores. See RECSYS_MODELS.md.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalized_adjacency(edges, n_nodes, drop_rate=0.0):
    """Edges are undirected pairs; dropout always removes both directions."""
    if not 0 <= drop_rate < 1:
        raise ValueError("Edge drop rate must lie in [0, 1).")
    if drop_rate:
        edges = edges[torch.rand(len(edges), device=edges.device) >= drop_rate]
    directed = torch.cat((edges, edges.flip(1)), dim=0).T
    degree = torch.bincount(directed[0], minlength=n_nodes).float()
    inverse = degree.clamp(min=1).rsqrt()
    values = inverse[directed[0]] * inverse[directed[1]]
    return torch.sparse_coo_tensor(directed, values, (n_nodes, n_nodes), check_invariants=True).coalesce()


def propagate(initial, adjacency, layers):
    states = [initial]
    for _ in range(layers):
        states.append(torch.sparse.mm(adjacency, states[-1]))
    return torch.stack(states).mean(dim=0)


def info_nce(first, second, temperature=0.2):
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive.")
    if len(first) < 2:
        return (first.sum() + second.sum()) * 0
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T / temperature
    targets = torch.arange(len(first), device=first.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


class Recommender(nn.Module):
    def __init__(self, data, dim=32, reg=1e-4):
        super().__init__()
        self.n_users, self.n_items = data.n_users, data.n_items
        self.dim, self.reg = dim, reg
        self.user_embedding = nn.Embedding(self.n_users, dim)
        self.item_embedding = nn.Embedding(self.n_items, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def export_vectors(self):
        users, items = self.forward()
        return users, items, torch.zeros(self.n_items, device=items.device)

    def score_all(self, users):
        user_vectors, item_vectors, bias = self.export_vectors()
        return user_vectors[users] @ item_vectors.T + bias

    def regularization(self, users, positives, negatives):
        return (self.user_embedding(users).square().sum()
                + self.item_embedding(positives).square().sum()
                + self.item_embedding(negatives).square().sum()) / (2 * len(users))

    def loss(self, users, positives, negatives):
        user_vectors, item_vectors, bias = self.export_vectors()
        positive = (user_vectors[users] * item_vectors[positives]).sum(-1) + bias[positives]
        negative = (user_vectors[users] * item_vectors[negatives]).sum(-1) + bias[negatives]
        ranking = F.softplus(negative - positive).mean()
        regularization = self.reg * self.regularization(users, positives, negatives)
        return {"total": ranking + regularization, "ranking": ranking, "regularization": regularization}


class LightGCN(Recommender):
    def __init__(self, data, features=None, *, dim=32, layers=2, reg=1e-4):
        super().__init__(data, dim, reg)
        edges = torch.as_tensor(data.pairs["train"].copy(), dtype=torch.long)
        edges[:, 1] += self.n_users
        self.layers = layers
        self.register_buffer("adjacency", normalized_adjacency(edges, self.n_users + self.n_items))

    def forward(self):
        result = propagate(torch.cat((self.user_embedding.weight, self.item_embedding.weight)),
                           self.adjacency, self.layers)
        return result[:self.n_users], result[self.n_users:]


class VBPR(Recommender):
    """Visual-only VBPR: ID factors, user visual factors, visual/item biases."""
    def __init__(self, data, features, *, dim=32, reg=1e-4, **unused):
        super().__init__(data, dim, reg)
        self.user_visual = nn.Embedding(self.n_users, dim)
        nn.init.normal_(self.user_visual.weight, std=0.1)
        self.register_buffer("image", torch.as_tensor(features["image"], dtype=torch.float32))
        self.visual_projection = nn.Linear(self.image.shape[1], dim, bias=False)
        self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.visual_bias = nn.Parameter(torch.zeros(self.image.shape[1]))

    def export_vectors(self):
        users = torch.cat((self.user_embedding.weight, self.user_visual.weight), dim=1)
        items = torch.cat((self.item_embedding.weight, self.visual_projection(self.image)), dim=1)
        return users, items, self.item_bias + self.image @ self.visual_bias

    def regularization(self, users, positives, negatives):
        return (super().regularization(users, positives, negatives)
                + self.user_visual(users).square().sum() / (2 * len(users))
                + self.visual_projection.weight.square().sum() / 2
                + self.visual_bias.square().sum() / 2
                + (self.item_bias[positives].square().sum() + self.item_bias[negatives].square().sum()) / (2 * len(users)))


class BM3(LightGCN):
    """Bootstrap targets with stop-gradient; no BPR/negative sampling objective."""
    def __init__(self, data, features, *, dim=32, layers=2, reg=1e-4, dropout=0.3, **unused):
        super().__init__(data, dim=dim, layers=layers, reg=reg)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.predictor = nn.Linear(dim, dim)
        # from_pretrained keeps the tensor storage. Clone before making it trainable:
        # as_tensor alone aliases the shared NumPy catalog and contaminates later runs.
        self.text_embedding = nn.Embedding.from_pretrained(torch.as_tensor(features["text"], dtype=torch.float32).clone(), freeze=False)
        self.image_embedding = nn.Embedding.from_pretrained(torch.as_tensor(features["image"], dtype=torch.float32).clone(), freeze=False)
        self.text_projection = nn.Linear(features["text"].shape[1], dim)
        self.image_projection = nn.Linear(features["image"].shape[1], dim)
        self.register_buffer("image_mask", torch.as_tensor(features["image_mask"], dtype=torch.bool))
        self.dropout = dropout

    def forward(self):
        users, items = super().forward()
        return users, items + self.item_embedding.weight

    def export_vectors(self):
        users, items = self.forward()
        return self.predictor(users), self.predictor(items), torch.zeros(self.n_items, device=items.device)

    @staticmethod
    def distance(first, second):
        return (1 - F.cosine_similarity(first, second.detach(), dim=-1)).mean()

    def loss(self, users, positives, negatives=None):
        user_base, item_base = self.forward()
        # Each target is detached before dropout; no target-network gradients.
        user_target = F.dropout(user_base.detach(), p=self.dropout, training=True)
        item_target = F.dropout(item_base.detach(), p=self.dropout, training=True)
        online_user, online_item = self.predictor(user_base), self.predictor(item_base)
        bootstrap = self.distance(online_user[users], item_target[positives])
        bootstrap = bootstrap + self.distance(online_item[positives], user_target[users])
        modality = bootstrap.new_zeros(())
        for projection, embedding, mask in (
            (self.text_projection, self.text_embedding, torch.ones_like(positives, dtype=torch.bool)),
            (self.image_projection, self.image_embedding, self.image_mask[positives]),
        ):
            if not mask.any():
                continue
            selected = positives[mask]
            projected = projection(embedding(selected))
            target = F.dropout(projected.detach(), p=self.dropout, training=True)
            predicted = self.predictor(projected)
            modality = modality + self.distance(predicted, item_target[selected]) + self.distance(predicted, target)
        regularization = self.reg * (user_base.norm() + item_base.norm()) / self.n_items
        return {"total": bootstrap + modality + regularization, "bootstrap": bootstrap,
                "modality": modality, "regularization": regularization}


def typed_graph(data, features, use_image=True):
    """Typed catalog nodes, optionally linked group -> attribute -> value hierarchy."""
    n_users, n_items = data.n_users, data.n_items
    image_items = np.flatnonzero(features["image_mask"]) if use_image else np.array([], dtype=int)
    spec_values = sorted({token for values in features["specs"].values() for token in values})
    spec_index = {token: index for index, token in enumerate(spec_values)}
    offsets = {"user": 0, "item": n_users, "text": n_users + n_items,
               "image": n_users + n_items * 2,
               "spec": n_users + n_items * 2 + len(image_items)}
    relation_edges = {
        "user_item": [(int(user), n_users + int(item)) for user, item in data.pairs["train"]],
        "item_text": [(n_users + item, offsets["text"] + item) for item in range(n_items)],
        "item_image": [(n_users + int(item), offsets["image"] + index) for index, item in enumerate(image_items)],
        "item_spec": [(n_users + item, offsets["spec"] + spec_index[token])
                      for item, item_id in enumerate(data.item_ids) for token in features["specs"][item_id]],
    }
    hierarchy = features.get("spec_hierarchy")
    attributes, groups = [], []
    if hierarchy is not None:
        value_to_attribute = hierarchy["value_to_attribute"]
        attribute_to_group = hierarchy["attribute_to_group"]
        if set(value_to_attribute) != set(spec_values):
            raise ValueError("Hierarchy values differ from explicit item spec tokens")
        attributes = sorted(set(value_to_attribute.values()))
        if set(attribute_to_group) != set(attributes):
            raise ValueError("Hierarchy attributes have missing or orphan groups")
        groups = sorted(set(attribute_to_group.values()))
        offsets["spec_attribute"] = offsets["spec"] + len(spec_values)
        offsets["spec_group"] = offsets["spec_attribute"] + len(attributes)
        attribute_index = {key: i for i, key in enumerate(attributes)}
        group_index = {key: i for i, key in enumerate(groups)}
        relation_edges["spec_attribute"] = [(offsets["spec"] + spec_index[token],
            offsets["spec_attribute"] + attribute_index[value_to_attribute[token]]) for token in spec_values]
        relation_edges["attribute_group"] = [(offsets["spec_attribute"] + attribute_index[key],
            offsets["spec_group"] + group_index[attribute_to_group[key]]) for key in attributes]
        # Attribute-presence shortcuts keep group context within two hops of items.
        relation_edges["item_attribute"] = [(n_users + item, offsets["spec_attribute"] + attribute_index[key])
            for item, item_id in enumerate(data.item_ids)
            for key in sorted({value_to_attribute[token] for token in features["specs"][item_id]})]
    edges = torch.tensor([edge for values in relation_edges.values() for edge in values], dtype=torch.long)
    schema = {"nodes": {"user": n_users, "item": n_items, "text": n_items,
                        "image": len(image_items), "spec": len(spec_values)},
              "undirected_edges": {name: len(values) for name, values in relation_edges.items()},
              "interaction_policy": "train_only", "spec_tokens": spec_values}
    if hierarchy is not None:
        schema["nodes"].update(spec_attribute=len(attributes), spec_group=len(groups))
        schema.update(spec_attributes=attributes, spec_groups=groups,
                      hierarchy_policy="explicit group-attribute-value with item-attribute presence shortcuts")
    return edges, offsets, image_items, schema


class TypedGraphGCL(Recommender):
    def __init__(self, data, features, *, dim=32, layers=2, reg=1e-4,
                 use_image=True, ssl_weight=0.05, edge_dropout=0.2, temperature=0.2):
        super().__init__(data, dim, reg)
        edges, self.offsets, image_items, self.schema = typed_graph(data, features, use_image)
        self.n_nodes = sum(self.schema["nodes"].values())
        self.layers, self.ssl_weight = layers, ssl_weight
        self.edge_dropout, self.temperature = edge_dropout, temperature
        self.register_buffer("edges", edges)
        self.register_buffer("adjacency", normalized_adjacency(edges, self.n_nodes))
        self.register_buffer("text", torch.as_tensor(features["text"], dtype=torch.float32))
        self.register_buffer("image", torch.as_tensor(features["image"][image_items], dtype=torch.float32))
        self.text_projection = nn.Linear(features["text"].shape[1], dim, bias=False)
        self.image_projection = nn.Linear(features["image"].shape[1], dim, bias=False)
        self.spec_embedding = nn.Embedding(len(self.schema["spec_tokens"]), dim)
        nn.init.normal_(self.spec_embedding.weight, std=0.1)
        self.attribute_embedding = self.group_embedding = None
        if "spec_attribute" in self.schema["nodes"]:
            self.attribute_embedding = nn.Embedding(len(self.schema["spec_attributes"]), dim)
            self.group_embedding = nn.Embedding(len(self.schema["spec_groups"]), dim)
            nn.init.normal_(self.attribute_embedding.weight, std=0.1)
            nn.init.normal_(self.group_embedding.weight, std=0.1)

    def forward(self, augment=False):
        states = [self.user_embedding.weight, self.item_embedding.weight,
                             self.text_projection(self.text), self.image_projection(self.image),
                             self.spec_embedding.weight]
        if self.attribute_embedding is not None:
            states.extend((self.attribute_embedding.weight, self.group_embedding.weight))
        initial = torch.cat(states)
        adjacency = normalized_adjacency(self.edges, self.n_nodes, self.edge_dropout) if augment else self.adjacency
        result = propagate(initial, adjacency, self.layers)
        return result[:self.n_users], result[self.n_users:self.n_users + self.n_items]

    def loss(self, users, positives, negatives):
        result = super().loss(users, positives, negatives)
        if self.ssl_weight:
            first_users, first_items = self.forward(augment=True)
            second_users, second_items = self.forward(augment=True)
            selected_users = torch.unique(users)
            selected_items = torch.unique(positives)
            selected_users = selected_users[torch.randperm(len(selected_users))[:256]]
            selected_items = selected_items[torch.randperm(len(selected_items))[:256]]
            contrastive = info_nce(first_users[selected_users], second_users[selected_users], self.temperature)
            contrastive = contrastive + info_nce(first_items[selected_items], second_items[selected_items], self.temperature)
            result["contrastive"] = contrastive
            result["total"] = result["total"] + self.ssl_weight * contrastive
        return result


METHODS = ("lightgcn", "vbpr", "bm3", "graph_text_only", "graph_multimodal", "graph_multimodal_gcl")


def make_model(method, data, features, *, dim=32, layers=2, reg=1e-4, ssl_weight=0.05):
    if method == "lightgcn":
        return LightGCN(data, dim=dim, layers=layers, reg=reg)
    if method == "vbpr":
        return VBPR(data, features, dim=dim, reg=reg)
    if method == "bm3":
        return BM3(data, features, dim=dim, layers=layers, reg=reg)
    if method in METHODS:
        return TypedGraphGCL(data, features, dim=dim, layers=layers, reg=reg,
            use_image=method != "graph_text_only", ssl_weight=ssl_weight if method.endswith("_gcl") else 0.0)
    raise ValueError(f"Unknown recommender: {method}")
