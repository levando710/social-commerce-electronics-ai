"""Small BM25 utilities for the offline RAG demo."""
from collections import Counter
import math
import re


TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text):
    return TOKEN.findall(str(text or "").lower())


def build_index(documents):
    tokenized = [tokenize(doc.get("text", "")) for doc in documents]
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    doc_count = len(documents)
    avgdl = sum(len(tokens) for tokens in tokenized) / max(doc_count, 1)
    return {
        "doc_count": doc_count,
        "avgdl": avgdl,
        "documents": [
            {
                "doc_id": doc.get("doc_id"),
                "length": len(tokens),
                "terms": dict(Counter(tokens)),
            }
            for doc, tokens in zip(documents, tokenized)
        ],
        "idf": {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        },
    }


def score(index, query, k1=1.5, b=0.75):
    query_terms = tokenize(query)
    if not query_terms:
        return []
    query_counter = Counter(query_terms)
    avgdl = index.get("avgdl") or 1
    scores = []
    for doc in index.get("documents", []):
        total = 0.0
        length = doc.get("length") or 0
        terms = doc.get("terms") or {}
        for term, query_weight in query_counter.items():
            freq = terms.get(term, 0)
            if not freq:
                continue
            idf = index.get("idf", {}).get(term, 0.0)
            denom = freq + k1 * (1 - b + b * length / avgdl)
            total += query_weight * idf * (freq * (k1 + 1) / denom)
        if total:
            scores.append((doc.get("doc_id"), total))
    return sorted(scores, key=lambda item: item[1], reverse=True)
