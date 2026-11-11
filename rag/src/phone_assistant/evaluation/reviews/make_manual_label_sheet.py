"""Create a manual relevance labeling sheet from retrieval top results.

The output CSV is meant to be edited by humans. Fill `manual_relevant` with:
- 1: relevant
- 0: not relevant
- blank: not labeled yet
"""
from __future__ import annotations
from phone_assistant.common.project_layout import MODULE_ROOT

import argparse
import csv
from pathlib import Path


ROOT = MODULE_ROOT
DEFAULT_METHODS = ["bm25", "semantic_tfidf", "hybrid", "intent_hybrid"]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "query_id",
            "query",
            "method",
            "rank",
            "doc_id",
            "title",
            "brand",
            "rating",
            "review_count",
            "auto_relevant",
            "manual_relevant",
            "note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-results", type=Path, default=ROOT / 'artifacts/evaluation' / "results" / "retrieval_top_results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation' / "manual_relevance_labels.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=30)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    args = parser.parse_args()

    rows = read_csv(args.top_results.resolve())
    chosen_queries = []
    seen_queries = set()
    for row in rows:
        qid = row["query_id"]
        if qid not in seen_queries:
            chosen_queries.append(qid)
            seen_queries.add(qid)
        if len(chosen_queries) >= args.max_queries:
            break
    chosen_query_set = set(chosen_queries)
    method_set = set(args.methods)

    output = []
    seen_items = set()
    for row in rows:
        if row["query_id"] not in chosen_query_set:
            continue
        if row["method"] not in method_set:
            continue
        if int(row["rank"]) > args.top_k:
            continue
        key = (row["query_id"], row["method"], row["rank"], row["doc_id"])
        if key in seen_items:
            continue
        seen_items.add(key)
        output.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "method": row["method"],
            "rank": row["rank"],
            "doc_id": row["doc_id"],
            "title": row["title"],
            "brand": row["brand"],
            "rating": row["rating"],
            "review_count": row["review_count"],
            "auto_relevant": row["is_relevant"],
            "manual_relevant": "",
            "note": "",
        })

    write_csv(args.output.resolve(), output)
    print(f"Wrote {len(output)} rows to {args.output.resolve()}")
    print("Fill manual_relevant with 1/0, then run scripts/evaluate_manual_labels.py")


if __name__ == "__main__":
    main()

