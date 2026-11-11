"""Build a BM25 text index over prepared RAG documents."""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
import json
from pathlib import Path

from phone_assistant.retrieval.search.bm25 import build_index


ROOT = MODULE_ROOT


def read_documents(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "documents.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "bm25_index.json")
    args = parser.parse_args()

    documents = read_documents(args.corpus.resolve())
    index = build_index(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "documents": index["doc_count"],
        "avg_doc_length": round(index["avgdl"], 2),
        "vocabulary": len(index["idf"]),
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
