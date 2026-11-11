"""Shared method routing for CLI and chat; baseline mode needs no neural packages."""
METHODS = ("legacy", "dense-text", "dense-clip", "sparse-dense", "multimodal", "intent-multimodal")


def retrieve(index, documents, query, *, method="legacy", dense=None, top_k=5, use_llm=False):
    if method not in METHODS:
        raise ValueError(f"Unknown retrieval method: {method}")
    if not query.strip() or not 1 <= top_k <= 100:
        raise ValueError("Query must be nonempty and top_k must be between 1 and 100.")
    if method == "legacy":
        from phone_assistant.retrieval.search.hybrid_search import intent_hybrid_search
        return intent_hybrid_search(index, documents, query, top_k=top_k, use_llm=use_llm)
    if dense is None:
        raise ValueError("Dense index is required for this method.")
    from phone_assistant.retrieval.search.dense_search import dense_results, multimodal_search
    if method in ("dense-text", "dense-clip"):
        return dense_results(dense, documents, query,
                             channel="text" if method == "dense-text" else "clip", top_k=top_k)
    return multimodal_search(index, documents, dense, query, top_k=top_k,
        use_clip=method != "sparse-dense", use_query_plan=method == "intent-multimodal", use_llm=use_llm)
