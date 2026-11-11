"""Hybrid retrieval helpers for the phone RAG demo.

The retriever keeps BM25 as an explainable baseline and adds small, deterministic
reranking signals that make the demo friendlier for Vietnamese and English queries.
"""
import re

from phone_assistant.retrieval.search.bm25 import score, tokenize
from phone_assistant.retrieval.queries.query_understanding import understand_query


QUERY_EXPANSIONS = {
    "pin": ["battery", "battery health", "long lasting"],
    "camera": ["camera", "mp", "rear", "front", "photo"],
    "ảnh": ["image", "camera", "photo"],
    "anh": ["image", "camera", "photo"],
    "mở khóa": ["unlocked", "gsm", "cdma"],
    "mo khoa": ["unlocked", "gsm", "cdma"],
    "quốc tế": ["unlocked", "international", "gsm"],
    "quoc te": ["unlocked", "international", "gsm"],
    "lock": ["locked", "carrier", "verizon", "at&t", "t-mobile"],
    "khóa mạng": ["locked", "carrier", "verizon", "at&t", "t-mobile"],
    "khoa mang": ["locked", "carrier", "verizon", "at&t", "t-mobile"],
    "giá rẻ": ["cheap", "budget", "basic"],
    "gia re": ["cheap", "budget", "basic"],
    "rẻ": ["cheap", "budget", "basic"],
    "re": ["cheap", "budget", "basic"],
    "đánh giá": ["review", "rating", "verified purchase"],
    "danh gia": ["review", "rating", "verified purchase"],
    "review": ["review", "rating", "verified purchase"],
    "tốt": ["good", "excellent", "great"],
    "tot": ["good", "excellent", "great"],
    "cũ": ["renewed", "used", "refurbished"],
    "cu": ["renewed", "used", "refurbished"],
    "mới": ["new", "renewed"],
    "moi": ["new", "renewed"],
    "sạc": ["charger", "charging", "battery"],
    "sac": ["charger", "charging", "battery"],
    "bộ nhớ": ["storage", "gb", "memory"],
    "bo nho": ["storage", "gb", "memory"],
}

BRANDS = {
    "apple", "iphone", "samsung", "galaxy", "nokia", "motorola", "moto",
    "oneplus", "pixel", "google", "sony", "xperia", "lg", "blackberry",
    "htc", "huawei", "xiaomi", "redmi", "oppo", "vivo", "realme", "blu",
}


def expand_query(query):
    expanded = [query]
    lower = query.lower()
    for key, values in QUERY_EXPANSIONS.items():
        if key in lower:
            expanded.extend(values)
    return " ".join(expanded)


def wants_reviews(query):
    lower = query.lower()
    return any(term in lower for term in ("review", "rating", "đánh giá", "danh gia", "người dùng", "nguoi dung"))


def wants_images(query):
    lower = query.lower()
    return any(term in lower for term in ("image", "photo", "picture", "ảnh", "anh", "hình", "hinh"))


def wants_unlocked(query):
    lower = query.lower()
    return any(term in lower for term in ("unlocked", "mở khóa", "mo khoa", "quốc tế", "quoc te", "gsm"))


def wants_good_quality(query):
    lower = query.lower()
    return any(term in lower for term in ("good", "great", "excellent", "tốt", "tot", "ổn", "on"))


def is_probably_locked(doc):
    title = str(doc.get("title") or "").lower()
    text = str(doc.get("text") or "").lower()
    if "unlocked" in title:
        return False
    locked_patterns = (
        "locked to", "for verizon", "for t-mobile only", "for at&t", "at&t (renewed)",
        "verizon (renewed)", "t-mobile (renewed)", "sprint (renewed)", "carrier locked",
        "not compatible with any other carrier",
    )
    return any(pattern in title or pattern in text[:1000] for pattern in locked_patterns)


def normalize_scores(items):
    if not items:
        return {}
    max_score = max(value for _, value in items) or 1.0
    return {doc_id: value / max_score for doc_id, value in items}


def title_brand_boost(doc, query_tokens):
    title_tokens = set(tokenize(doc.get("title", "")))
    brand_tokens = set(tokenize(doc.get("brand", "")))
    boost = 0.0
    for token in query_tokens:
        if token in title_tokens:
            boost += 0.10
        if token in brand_tokens or token in BRANDS and token in title_tokens:
            boost += 0.12
    title = str(doc.get("title") or "").lower()
    query_text = " ".join(query_tokens)
    if query_text and query_text in title:
        boost += 0.30
    return min(boost, 0.80)


def visual_evidence_boost(doc, query_tokens):
    visual_items = doc.get("visual_evidence") or []
    if not visual_items:
        return 0.0
    visual_text_parts = []
    for item in visual_items:
        for key in ("caption_vi", "caption_en", "visible_product_type", "visible_brand_or_model"):
            if item.get(key):
                visual_text_parts.append(str(item[key]))
        for key in ("visual_tags", "condition_tags", "risk_tags", "evidence"):
            visual_text_parts.extend(str(value) for value in item.get(key) or [] if value)
    visual_tokens = set(tokenize(" ".join(visual_text_parts)))
    overlap = len(set(query_tokens) & visual_tokens)
    if not overlap:
        return 0.0
    return min(0.06 * overlap, 0.24)


def evidence_sentences(doc, query_tokens, limit=3):
    text = re.sub(r"<[^>]+>", " ", str(doc.get("text") or ""))
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    ranked = []
    token_set = set(query_tokens)
    for part in parts:
        clean = part.strip()
        if not clean:
            continue
        part_tokens = set(tokenize(clean))
        overlap = len(token_set & part_tokens)
        if overlap:
            ranked.append((overlap, clean))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = [sentence for _, sentence in ranked[:limit]]
    if not chosen:
        chosen = [p.strip() for p in parts if p.strip()][:limit]
    evidence = " ".join(chosen)
    return evidence[:700].strip()


def doc_search_text(doc):
    visual_parts = []
    for item in doc.get("visual_evidence") or []:
        for key in ("caption_vi", "caption_en", "visible_product_type", "visible_brand_or_model"):
            if item.get(key):
                visual_parts.append(str(item[key]))
        for key in ("visual_tags", "condition_tags", "risk_tags", "evidence"):
            visual_parts.extend(str(value) for value in item.get(key) or [] if value)
    parts = [
        doc.get("title", ""),
        doc.get("brand", ""),
        doc.get("text", ""),
        doc.get("details", ""),
        " ".join(visual_parts),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def plan_based_boost(doc, query_plan):
    if not query_plan:
        return 0.0
    text = doc_search_text(doc)
    final = 0.0

    for term in query_plan.get("must_have") or []:
        term = str(term).lower()
        if term and term in text:
            final += 0.12
        elif term:
            final -= 0.03

    for term in query_plan.get("nice_to_have") or []:
        term = str(term).lower()
        if term and term in text:
            final += 0.06

    for term in query_plan.get("avoid") or []:
        term = str(term).lower()
        if term and term in text:
            final -= 0.20

    tags = set(query_plan.get("tags") or [])
    filters = query_plan.get("filters") or {}

    preferred_brands = [str(value).lower() for value in filters.get("brand_preference") or []]
    if preferred_brands and any(brand in text for brand in preferred_brands):
        final += 0.18

    if filters.get("require_review") and doc.get("review_count", 0) > 0:
        final += 0.22
    elif filters.get("require_review"):
        final -= 0.12

    if filters.get("require_product_image") and doc.get("product_image_paths"):
        final += 0.20
    elif filters.get("require_product_image"):
        final -= 0.12

    if filters.get("network") == "unlocked":
        if "unlocked" in text or "gsm" in text:
            final += 0.22
        if is_probably_locked(doc):
            final -= 0.35

    if "review_rich" in tags and doc.get("review_count", 0) > 0:
        final += min(0.18, 0.03 * max(1, min(int(doc.get("review_count") or 0), 6)))
    if "has_image" in tags and doc.get("product_image_paths"):
        final += 0.12
    if "good_quality" in tags and doc.get("average_rating") is not None:
        final += 0.10 if doc["average_rating"] >= 4.0 else -0.10

    return final


def hybrid_search(index, documents, query, top_k=5, candidate_k=60, query_plan=None):
    if query_plan:
        retrieval_query = " ".join([
            query,
            str(query_plan.get("query_rewrite") or ""),
            " ".join(query_plan.get("keywords") or []),
        ])
    else:
        retrieval_query = query
    expanded_query = expand_query(retrieval_query or query)
    bm25_items = score(index, expanded_query)[:candidate_k]
    bm25_norm = normalize_scores(bm25_items)
    by_id = {doc["doc_id"]: doc for doc in documents}
    query_tokens = tokenize(expanded_query)
    original_tokens = tokenize(query)
    review_intent = wants_reviews(query)
    image_intent = wants_images(query)
    unlocked_intent = wants_unlocked(query)
    good_intent = wants_good_quality(query)

    ranked = []
    for doc_id, raw_score in bm25_items:
        doc = by_id.get(doc_id)
        if not doc:
            continue
        final = bm25_norm[doc_id]
        final += title_brand_boost(doc, original_tokens or query_tokens)
        final += visual_evidence_boost(doc, query_tokens)
        final += plan_based_boost(doc, query_plan)
        if doc.get("review_count", 0) > 0:
            final += 0.08
        else:
            final -= 0.04
        if doc.get("product_image_paths"):
            final += 0.06
        if review_intent and doc.get("review_count", 0) > 0:
            final += 0.20
        if image_intent and doc.get("product_image_paths"):
            final += 0.18
        if unlocked_intent:
            title = str(doc.get("title") or "").lower()
            details = str(doc.get("details") or "").lower()
            if "unlocked" in title or "unlocked" in details:
                final += 0.25
            if is_probably_locked(doc):
                final -= 0.35
        if doc.get("average_rating") and doc["average_rating"] >= 4.0:
            final += 0.04
        if good_intent and doc.get("average_rating") is not None:
            if doc["average_rating"] >= 4.0:
                final += 0.08
            elif doc["average_rating"] < 3.5:
                final -= 0.15
        ranked.append((doc_id, final, raw_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    results = []
    for doc_id, final, raw_score in ranked[:top_k]:
        doc = by_id[doc_id]
        results.append({
            "score": round(final, 4),
            "bm25_score": round(raw_score, 4),
            "doc_id": doc_id,
            "title": doc.get("title"),
            "brand": doc.get("brand"),
            "average_rating": doc.get("average_rating"),
            "review_count": doc.get("review_count"),
            "product_images": doc.get("product_image_paths", [])[:2],
            "review_images": doc.get("review_image_paths", [])[:2],
            "visual_evidence": doc.get("visual_evidence", [])[:2],
            "evidence": evidence_sentences(doc, query_tokens),
        })
    return results


def intent_hybrid_search(index, documents, query, top_k=5, candidate_k=80, use_llm=False):
    query_plan = understand_query(query, use_llm=use_llm)
    results = hybrid_search(
        index,
        documents,
        query,
        top_k=top_k,
        candidate_k=candidate_k,
        query_plan=query_plan,
    )
    for item in results:
        item["query_plan"] = query_plan
    return results
