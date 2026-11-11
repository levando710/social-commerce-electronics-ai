"""Verified frozen interaction data and metadata-only item features."""
from __future__ import annotations
from phone_assistant.common.project_layout import MODULE_ROOT

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from phone_assistant.data.preparation.prepare_corpus import DETAIL_KEYS

ROOT = MODULE_ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def metadata_text(product):
    """Explicit allowlist: no reviews, ratings, popularity or helpful-vote totals."""
    details = product.get("details") or {}
    parts = [str(product.get("title") or ""), str(product.get("store") or "")]
    for key in ("features", "description", "categories"):
        values = product.get(key) or []
        parts.extend([str(values)] if isinstance(values, str) else map(str, values))
    parts.extend(f"{key}: {details[key]}" for key in DETAIL_KEYS if details.get(key))
    return "\n".join(part for part in parts if part.strip())


def spec_tokens(product):
    details = product.get("details") or {}
    # Shared attribute nodes carry only explicit catalog values, never review labels.
    return sorted({f"{key.lower()}={str(details[key]).strip().lower()}"
                   for key in DETAIL_KEYS if details.get(key)})


class InteractionData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in self.manifest["artifacts_sha256"].items():
            if sha256(self.directory / name) != digest:
                raise ValueError(f"Frozen RecSys artifact changed: {name}")
        self.maps = json.loads((self.directory / "id_maps.json").read_text(encoding="utf-8"))
        self.user_ids = sorted(self.maps["users"], key=self.maps["users"].get)
        self.item_ids = sorted(self.maps["items"], key=self.maps["items"].get)
        for mapping in self.maps.values():
            if sorted(mapping.values()) != list(range(len(mapping))):
                raise ValueError("ID mapping must be contiguous.")
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)
        self.rows = {name: read_jsonl(self.directory / f"{name}.jsonl")
                     for name in ("train", "validation", "test")}
        self.pairs = {name: np.asarray([
            (self.maps["users"][row["user_id"]], self.maps["items"][row["item_id"]])
            for row in rows], dtype=np.int64).reshape(-1, 2) for name, rows in self.rows.items()}
        pair_sets = [set(map(tuple, pairs)) for pairs in self.pairs.values()]
        if any(len(pair_sets[idx]) != len(pairs) for idx, pairs in enumerate(self.pairs.values())):
            raise ValueError("Duplicate user-item pairs in split.")
        if any(pair_sets[a] & pair_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Train/validation/test pairs overlap.")
        self.train_seen = defaultdict(set)
        self.validation_seen = defaultdict(set)
        for user, item in self.pairs["train"]:
            self.train_seen[int(user)].add(int(item))
        for user, item in self.pairs["validation"]:
            self.validation_seen[int(user)].add(int(item))
        if not len(self.pairs["train"]) or any(len(items) == self.n_items for items in self.train_seen.values()):
            raise ValueError("Training data must allow negative sampling.")
        self.item_degree = np.bincount(self.pairs["train"][:, 1], minlength=self.n_items)

    def negatives(self, users, rng):
        """Only train positives excluded; held-out labels never guide sampling."""
        result = rng.integers(self.n_items, size=len(users))
        for index, user in enumerate(users):
            while int(result[index]) in self.train_seen[int(user)]:
                result[index] = rng.integers(self.n_items)
        return result


def load_features(path, data):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["split_manifest_sha256"] != sha256(data.directory / "manifest.json"):
        raise ValueError("Item features do not match the frozen split.")
    if manifest.get("text_policy") != "catalog_metadata_allowlist_no_reviews_no_ratings":
        raise ValueError("Unverified feature provenance; do not use review-enriched RAG vectors.")
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Feature artifact changed: {name}")
    with np.load(path / "features.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    if list(arrays["item_ids"]) != data.item_ids:
        raise ValueError("Item vector ordering differs from ID map.")
    for key in ("text", "image"):
        if arrays[key].shape[0] != data.n_items or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key} feature matrix")
    arrays["specs"] = json.loads((path / "specs.json").read_text(encoding="utf-8"))
    hierarchy_path = path / "spec_hierarchy.json"
    if hierarchy_path.exists():
        if "spec_hierarchy.json" not in manifest["artifacts_sha256"]:
            raise ValueError("Spec hierarchy lacks feature provenance checksum")
        arrays["spec_hierarchy"] = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    return arrays, manifest
