"""Query the offline hybrid RAG index and print evidence for a demo answer."""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
import json
from pathlib import Path
import sys

from phone_assistant.llm.gemini_client import generate_answer
from phone_assistant.retrieval.search.hybrid_search import hybrid_search
from phone_assistant.retrieval.serving.retrieval_service import METHODS, retrieve


ROOT = MODULE_ROOT
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query, for example: unlocked samsung good battery")
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "documents.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "bm25_index.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", action="store_true", help="Ask Gemini to synthesize an answer")
    parser.add_argument("--method", choices=METHODS, default="legacy")
    parser.add_argument("--dense-index", type=Path, default=ROOT / 'artifacts/runtime/catalog/dense_index')
    args = parser.parse_args()

    documents = read_jsonl(args.corpus.resolve())
    index = json.loads(args.index.resolve().read_text(encoding="utf-8"))
    if not 1 <= args.top_k <= 100:
        parser.error("top-k must be between 1 and 100")
    dense = None
    if args.method != "legacy":
        from phone_assistant.retrieval.search.dense_search import DenseRetriever
        dense = DenseRetriever(args.dense_index, documents)
        from phone_assistant.retrieval.search.bm25 import build_index
        from phone_assistant.retrieval.search.dense_search import text_document
        index = build_index([{**doc, "text": text_document(doc)} for doc in documents])
    results = (hybrid_search(index, documents, args.query, top_k=args.top_k)
               if args.method == "legacy" else
               retrieve(index, documents, args.query, method=args.method, dense=dense, top_k=args.top_k))
    llm = generate_answer(args.query, results) if args.llm else {}
    print(json.dumps({
        "query": args.query,
        "method": args.method,
        "top_k": len(results),
        "answer": llm.get("answer"),
        "llm_error": llm.get("error"),
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
