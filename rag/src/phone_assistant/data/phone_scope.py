"""Conservative handset-only serving scope; frozen mixed-catalog experiments stay intact."""
import re

import numpy as np

from phone_assistant.conversation.conversation_state import normalize
from phone_assistant.retrieval.queries.product_constraints import product_types


EXCLUDED = re.compile(r"\b(?:accessor\w*|case\w*|cover\w*|holster\w*|sleeve\w*|protector\w*|charger\w*|cable\w*|adapter\w*|"
    r"headphone\w*|earphone\w*|earbud\w*|headset\w*|signal booster\w*|antenna\w*|sim card\w*|"
    r"replacement\w*|repair\w*|digitizer\w*|lcd assembly|screen assembly|phone charm\w*|"
    r"phone shell\w*|wallet\w*|juice pack\w*|tool\w*|stylus\w*|mount\w*|holder\w*|stand\w*|"
    r"smartwatch\w*|watch band\w*|tablet\w*|ipad\w*|laptop\w*|notebook\w*|macbook\w*)\b")


def handset_decision(doc):
    title = normalize(doc.get("title") or "")
    categories = doc.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]
    # The parent category includes accessories; it is never positive handset evidence.
    specific = [normalize(x) for x in categories if normalize(x) != "cell phones & accessories"]
    if EXCLUDED.search(title) or any(EXCLUDED.search(x) for x in specific):
        return False, "explicit_non_handset_or_accessory"
    kinds = product_types(title)
    if kinds and kinds != ["phone"]:
        return False, "other_product_type"
    device_title = re.search(r"\b(?:smartphone|smart phone|cell phone|mobile phone|phone|iphone\s*\d{1,2}|galaxy\s*[sa]\d{1,3}|pixel\s*\d{1,2})\b", title)
    details = doc.get("details") or {}
    hardware = any(details.get(key) for key in ("Wireless Carrier", "Cellular Technology", "Operating System", "Memory Storage Capacity"))
    if (kinds == ["phone"] and (device_title or hardware)) or any(x in {"cell phones", "smartphones", "unlocked cell phones", "basic cell phones", "carrier cell phones"} for x in specific):
        return True, "handset_title_or_specific_category"
    return False, "insufficient_handset_evidence"


def handset_documents(documents):
    return [doc for doc in documents if handset_decision(doc)[0]]


class HandsetRecommender:
    def __init__(self, base, documents):
        self.base = base
        self.documents = {d["doc_id"]: d for d in documents}
        self.allowed = np.asarray([i for i, item in enumerate(base.data.item_ids) if item in self.documents], dtype=np.int64)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def recommend(self, user_index=None, top_k=5):
        # Delegate argument validation and public response metadata to the existing service.
        result = self.base.recommend(user_index, top_k)
        scores = self.data.item_degree.astype(float) if user_index is None else self.users[user_index] @ self.items.T + self.bias
        scores = scores.copy()
        if user_index is not None:
            scores[list(self.data.train_seen[user_index])] = -np.inf
        indices = sorted(self.allowed, key=lambda i: (-float(scores[i]), self.data.item_ids[i]))
        result["results"] = []
        for i in indices:
            if not np.isfinite(scores[i]):
                continue
            doc_id = self.data.item_ids[i]
            doc = self.documents[doc_id]
            result["results"].append({"doc_id": doc_id, "title": doc.get("title"), "brand": doc.get("brand"),
                "score": float(scores[i]), "product_images": doc.get("product_image_paths", [])[:1]})
            if len(result["results"]) == top_k:
                break
        result["catalog_scope"] = "handsets_only; existing mixed-catalog model, no handset-only retraining claim"
        return result


class HandsetRanking:
    def __init__(self, base, ids, catalog_size):
        self.base, self.ids, self.catalog_size = base, set(ids), catalog_size

    def __getattr__(self, name):
        return getattr(self.base, name)

    def search(self, query, *, channel="text", top_k=60):
        hits = self.base.search(query, channel=channel, top_k=self.catalog_size)
        return [(item, score) for item, score in hits if item in self.ids][:top_k]

    def rank(self, query, top_k=80):
        hits = self.base.rank(query, top_k=self.catalog_size)
        return [(item, score) for item, score in hits if item in self.ids][:top_k]
