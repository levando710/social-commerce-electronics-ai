"""Neural text/CLIP retrieval and weighted reciprocal-rank fusion.

Heavy dependencies and models are loaded only when a dense method is requested.
The small corpus uses exact cosine search over normalized NumPy vectors.
"""
from __future__ import annotations
from phone_assistant.common.project_layout import MODULE_ROOT

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import threading


ROOT = MODULE_ROOT
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"


def corpus_fingerprint(documents):
    payload = json.dumps(documents, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_document(doc):
    return "\n".join(
        line for line in str(doc.get("text") or doc.get("title") or "").splitlines()
        if not line.startswith("Visual evidence:")
    )


def text_chunks(doc, words=80, overlap=16):
    """Avoid silently embedding just the first 128 tokens of a product document."""
    if words <= overlap or overlap < 0:
        raise ValueError("Chunk size must be larger than overlap >= 0.")
    text = text_document(doc)
    tokens = text.split()
    chunks = []
    for offset in range(0, len(tokens), words - overlap):
        chunks.append(" ".join(tokens[offset:offset + words]))
        if offset + words >= len(tokens):
            break
    return chunks or [str(doc.get("title") or "unknown product")]


def load_encoder(model_name, *, revision="main", device="cpu", allow_download=False):
    # Keep downloaded artifacts inside the project, rather than the user's home.
    cache = ROOT / 'artifacts/cache' / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install rag/requirements-dense.txt to use neural retrieval.") from exc
    return SentenceTransformer(
        model_name, revision=revision, device=device,
        cache_folder=str(cache / "hub"), local_files_only=not allow_download,
        trust_remote_code=False,
    )


def rank_vectors(vectors, doc_ids, query_vector, top_k):
    """Use the best chunk/image per product; repeated images cannot inflate ranks."""
    import numpy as np

    scores = np.asarray(vectors) @ np.asarray(query_vector)
    best = {}
    for doc_id, value in zip(doc_ids, scores):
        key = str(doc_id)
        best[key] = max(best.get(key, float("-inf")), float(value))
    return sorted(best.items(), key=lambda row: (-row[1], row[0]))[:top_k]


def reciprocal_rank_fusion(rankings, *, weights=None, rank_constant=60, top_k=60):
    if rank_constant <= 0:
        raise ValueError("RRF rank_constant must be positive.")
    weights = weights or {name: 1.0 for name in rankings}
    totals = defaultdict(float)
    contributions = defaultdict(dict)
    for name, rows in rankings.items():
        weight = weights.get(name, 1.0)
        if weight < 0:
            raise ValueError("RRF weights must be nonnegative.")
        if weight == 0:
            continue
        seen = set()
        for rank, (doc_id, raw_score) in enumerate(rows, 1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            totals[doc_id] += weight / (rank_constant + rank)
            contributions[doc_id][name] = {"rank": rank, "score": float(raw_score)}
    ranked = sorted(totals.items(), key=lambda row: (-row[1], row[0]))[:top_k]
    return [(doc_id, value, contributions[doc_id]) for doc_id, value in ranked]


class DenseRetriever:
    def __init__(self, path, documents, *, device="cpu", allow_download=False):
        import numpy as np

        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported dense index schema; rebuild the index.")
        if self.manifest["corpus_sha256"] != corpus_fingerprint(documents):
            raise ValueError("Dense index does not match the current corpus; rebuild it.")
        self.arrays = {}
        for channel, config in self.manifest["channels"].items():
            with np.load(self.path / config["file"], allow_pickle=False) as data:
                vectors = data["vectors"].copy()
                doc_ids = data["doc_ids"].copy()
            if vectors.ndim != 2 or len(vectors) != len(doc_ids) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid vectors for channel {channel}.")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
                raise ValueError(f"Unnormalized vectors for channel {channel}.")
            self.arrays[channel] = (vectors, doc_ids)
        self.device = device
        self.allow_download = allow_download
        self.encoders = {}
        self.query_cache = {}
        self.lock = threading.RLock()

    def search(self, query, *, channel="text", top_k=60):
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if channel not in self.arrays:
            raise ValueError(f"Dense index has no {channel} channel; rebuild with --mode both.")
        config = self.manifest["channels"][channel]
        with self.lock:
            if channel not in self.encoders:
                self.encoders[channel] = load_encoder(
                    config["model"], revision=config["revision"], device=self.device,
                    allow_download=self.allow_download,
                )
            key = (channel, query)
            if key not in self.query_cache:
                if len(self.query_cache) >= 512:
                    self.query_cache.clear()
                self.query_cache[key] = self.encoders[channel].encode(
                    [query], normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False,
                )[0]
            vector = self.query_cache[key]
        return rank_vectors(*self.arrays[channel], vector, top_k)


def result_rows(documents, ranked, query):
    from phone_assistant.retrieval.search.bm25 import tokenize
    from phone_assistant.retrieval.search.hybrid_search import evidence_sentences

    by_id = {doc["doc_id"]: doc for doc in documents}
    rows = []
    for doc_id, value, sources in ranked:
        doc = by_id[doc_id]
        rows.append({
            "doc_id": doc_id, "score": round(float(value), 6),
            "bm25_score": sources.get("bm25", {}).get("score", 0.0),
            "retrieval_sources": sources,
            "title": doc.get("title"), "brand": doc.get("brand"),
            "average_rating": doc.get("average_rating"), "review_count": doc.get("review_count"),
            "product_images": doc.get("product_image_paths", [])[:2],
            "review_images": doc.get("review_image_paths", [])[:2],
            "visual_evidence": doc.get("visual_evidence", [])[:2],
            "evidence": evidence_sentences(doc, tokenize(query)),
        })
    return rows


def dense_results(retriever, documents, query, *, channel="text", top_k=5):
    hits = retriever.search(query, channel=channel, top_k=top_k)
    return result_rows(documents, [
        (doc_id, value, {channel: {"rank": rank, "score": value}})
        for rank, (doc_id, value) in enumerate(hits, 1)
    ], query)


def multimodal_search(index, documents, retriever, query, *, top_k=5, candidate_k=80,
                      use_clip=True, use_query_plan=False, use_llm=False):
    from phone_assistant.retrieval.search.bm25 import score
    from phone_assistant.retrieval.queries.query_understanding import understand_query

    plan = understand_query(query, use_llm=use_llm) if use_query_plan else None
    retrieval_query = (plan.get("query_rewrite") or query) if plan else query
    rankings = {
        "bm25": score(index, retrieval_query)[:candidate_k],
        "text": retriever.search(retrieval_query, channel="text", top_k=candidate_k),
    }
    if use_clip:
        # CLIP's text encoder is English-centric. Original-query results remain
        # the controlled baseline; an optional rewritten query is a separate run.
        rankings["clip"] = retriever.search(retrieval_query, channel="clip", top_k=candidate_k)
    ranked = reciprocal_rank_fusion(
        rankings, weights={"bm25": 1.0, "text": 1.0, "clip": 0.5}, top_k=top_k,
    )
    rows = result_rows(documents, ranked, query)
    for row in rows:
        row["query_plan"] = plan
    return rows
