"""Lightweight semantic retrieval baseline for offline evaluation.

This is not a neural embedding model. It is a local semantic TF-IDF baseline:
tokens are expanded into coarse concepts such as battery_good, camera_good,
unlocked, review_rich, image_evidence, then ranked with cosine similarity.

The goal is to provide a cheap semantic baseline for comparison without
downloading models or calling external APIs.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata


TOKEN = re.compile(r"[a-z0-9]+", re.I)

CONCEPTS = {
    "battery_good": [
        "battery", "pin", "charger", "charging", "long", "lasting", "mah",
        "dung", "ngay", "trau",
    ],
    "camera_good": [
        "camera", "photo", "picture", "image", "rear", "front", "mp",
        "chup", "anh",
    ],
    "unlocked": [
        "unlocked", "international", "gsm", "global", "quoc", "te", "mo", "khoa",
    ],
    "locked": [
        "locked", "carrier", "verizon", "tmobile", "sprint", "att", "khoa", "mang",
    ],
    "review_rich": [
        "review", "rating", "verified", "purchase", "danh", "gia", "nguoi", "dung",
    ],
    "budget": [
        "cheap", "budget", "basic", "affordable", "gia", "re", "tiet", "kiem",
    ],
    "storage": [
        "storage", "memory", "gb", "tb", "capacity", "bo", "nho", "dung", "luong",
    ],
    "good_condition": [
        "good", "great", "excellent", "renewed", "refurbished", "tot", "on", "muot",
    ],
    "image_evidence": [
        "image", "photo", "picture", "visual", "evidence", "anh", "hinh",
    ],
    "basic_phone": [
        "basic", "keypad", "keyboard", "qwerty", "flip", "bar", "ban", "phim",
    ],
}

BRAND_TERMS = {
    "apple": ["apple", "iphone", "ios"],
    "samsung": ["samsung", "galaxy"],
    "google": ["google", "pixel"],
    "sony": ["sony", "xperia"],
    "motorola": ["motorola", "moto"],
    "oneplus": ["oneplus"],
    "nokia": ["nokia"],
    "xiaomi": ["xiaomi", "redmi"],
    "oppo": ["oppo"],
    "vivo": ["vivo"],
    "realme": ["realme"],
    "huawei": ["huawei"],
    "lg": ["lg"],
    "blackberry": ["blackberry"],
}


def strip_accents(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(strip_accents(text))


def visual_text(doc: dict) -> str:
    parts = []
    for item in doc.get("visual_evidence") or []:
        for key in ("caption_vi", "caption_en", "visible_product_type", "visible_brand_or_model"):
            if item.get(key):
                parts.append(str(item[key]))
        for key in ("visual_tags", "condition_tags", "risk_tags", "evidence"):
            parts.extend(str(value) for value in item.get(key) or [] if value)
    return " ".join(parts)


def doc_text(doc: dict) -> str:
    return " ".join(str(part) for part in [
        doc.get("title", ""),
        doc.get("brand", ""),
        doc.get("text", ""),
        doc.get("details", ""),
        visual_text(doc),
    ] if part)


def semantic_terms(text: str) -> list[str]:
    tokens = tokenize(text)
    token_set = set(tokens)
    output = list(tokens)

    for concept, aliases in CONCEPTS.items():
        if token_set & set(aliases):
            output.extend([f"concept_{concept}"] * 3)

    for brand, aliases in BRAND_TERMS.items():
        if token_set & set(aliases):
            output.extend([f"brand_{brand}"] * 4)

    for token in tokens:
        if token.endswith("gb") or token.endswith("tb") or token in {"16", "32", "64", "128", "256", "512"}:
            output.extend(["concept_storage"] * 2)
    return output


def build_semantic_index(documents: list[dict]) -> dict:
    vectors = []
    doc_freq = Counter()
    for doc in documents:
        terms = semantic_terms(doc_text(doc))
        counts = Counter(terms)
        vectors.append({"doc_id": doc.get("doc_id"), "terms": counts})
        doc_freq.update(counts.keys())

    doc_count = len(vectors)
    idf = {
        term: math.log(1 + doc_count / (1 + freq))
        for term, freq in doc_freq.items()
    }
    for vector in vectors:
        weighted = {
            term: (1 + math.log(freq)) * idf.get(term, 0.0)
            for term, freq in vector["terms"].items()
            if freq > 0
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        vector["weighted"] = weighted
        vector["norm"] = norm
        del vector["terms"]

    return {"doc_count": doc_count, "idf": idf, "documents": vectors}


def query_vector(index: dict, query: str) -> tuple[dict[str, float], float]:
    counts = Counter(semantic_terms(query))
    weighted = {
        term: (1 + math.log(freq)) * index.get("idf", {}).get(term, 0.0)
        for term, freq in counts.items()
        if freq > 0 and term in index.get("idf", {})
    }
    norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
    return weighted, norm


def semantic_search(index: dict, query: str, top_k=10) -> list[tuple[str, float]]:
    qvec, qnorm = query_vector(index, query)
    if not qvec:
        return []
    scores = []
    q_terms = set(qvec)
    for doc in index.get("documents", []):
        dvec = doc.get("weighted") or {}
        overlap = q_terms & set(dvec)
        if not overlap:
            continue
        dot = sum(qvec[term] * dvec[term] for term in overlap)
        score = dot / (qnorm * (doc.get("norm") or 1.0))
        if score:
            scores.append((doc.get("doc_id"), score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def semantic_results(index: dict, documents_by_id: dict, query: str, top_k=10) -> list[dict]:
    rows = []
    for doc_id, value in semantic_search(index, query, top_k=top_k):
        doc = documents_by_id.get(doc_id)
        if not doc:
            continue
        rows.append({
            "doc_id": doc_id,
            "score": value,
            "title": doc.get("title"),
            "brand": doc.get("brand"),
            "average_rating": doc.get("average_rating"),
            "review_count": doc.get("review_count"),
            "product_images": doc.get("product_image_paths", [])[:2],
            "review_images": doc.get("review_image_paths", [])[:2],
        })
    return rows

