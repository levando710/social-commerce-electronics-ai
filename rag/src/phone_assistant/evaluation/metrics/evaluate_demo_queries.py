"""Run a small fixed-query check for the upgraded RAG retriever."""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
import json
from pathlib import Path
import sys

from phone_assistant.retrieval.search.hybrid_search import hybrid_search


ROOT = MODULE_ROOT
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
QUERIES = [
    "iphone unlocked renewed good camera",
    "samsung galaxy unlocked good battery",
    "điện thoại quốc tế pin tốt",
    "cheap nokia basic phone",
    "review iphone battery health",
    "android phone with good camera and image",
]


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "documents.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "bm25_index.json")
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "demo_query_results.json")
    args = parser.parse_args()

    documents = read_jsonl(args.corpus.resolve())
    index = json.loads(args.index.resolve().read_text(encoding="utf-8"))
    report = []
    for query in QUERIES:
        results = hybrid_search(index, documents, query, top_k=3)
        report.append({
            "query": query,
            "top_results": [
                {
                    "title": item["title"],
                    "score": item["score"],
                    "bm25_score": item["bm25_score"],
                    "rating": item["average_rating"],
                    "review_count": item["review_count"],
                    "has_image": bool(item["product_images"]),
                    "evidence": item["evidence"][:260],
                }
                for item in results
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
