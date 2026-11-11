"""Evaluate retrieval metrics from a manually labeled CSV sheet."""
from __future__ import annotations
from phone_assistant.common.project_layout import MODULE_ROOT

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = MODULE_ROOT
DEFAULT_K_VALUES = (1, 3, 5, 10)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dcg(binary_relevance):
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(binary_relevance))


def metrics_for_hits(hits, total_relevant, k_values):
    output = {}
    for k in k_values:
        top_hits = hits[:k]
        hit_count = sum(top_hits)
        ideal = [1] * min(total_relevant, k)
        ideal_dcg = dcg(ideal)
        first_hit = 0
        for idx, hit in enumerate(top_hits, start=1):
            if hit:
                first_hit = idx
                break
        output[f"precision@{k}"] = hit_count / k if k else 0.0
        output[f"recall@{k}"] = hit_count / total_relevant if total_relevant else 0.0
        output[f"ndcg@{k}"] = dcg(top_hits) / ideal_dcg if ideal_dcg else 0.0
        output[f"mrr@{k}"] = 1 / first_hit if first_hit else 0.0
        output[f"coverage@{k}"] = 1.0 if hit_count else 0.0
    return output


def mean(rows, key):
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def write_markdown(path: Path, summary: dict, k_values) -> None:
    lines = [
        "# Manual relevance evaluation",
        "",
        "Bao cao nay tinh metric tu nhan `manual_relevant` trong file CSV gan nhan thu cong.",
        "",
        f"- So query da gan nhan: {summary['labeled_queries']}",
        f"- So dong da gan nhan: {summary['labeled_rows']}",
        "",
        "## Metric trung binh",
        "",
    ]
    headers = ["method"]
    for k in k_values:
        headers.extend([f"P@{k}", f"R@{k}", f"NDCG@{k}", f"MRR@{k}", f"Cov@{k}"])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for method, metrics in summary["mean_metrics"].items():
        row = [method]
        for k in k_values:
            row.extend([
                f"{metrics[f'precision@{k}']:.3f}",
                f"{metrics[f'recall@{k}']:.3f}",
                f"{metrics[f'ndcg@{k}']:.3f}",
                f"{metrics[f'mrr@{k}']:.3f}",
                f"{metrics[f'coverage@{k}']:.3f}",
            ])
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / 'artifacts/evaluation' / "manual_relevance_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / 'artifacts/evaluation' / "manual_results")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = read_csv(args.labels.resolve())
    labeled = [row for row in rows if str(row.get("manual_relevant", "")).strip() in {"0", "1"}]
    if not labeled:
        raise SystemExit("No manual labels found. Fill manual_relevant with 1/0 first.")

    grouped = defaultdict(list)
    relevant_by_query = defaultdict(set)
    for row in labeled:
        key = (row["query_id"], row["method"])
        grouped[key].append(row)
        if row["manual_relevant"].strip() == "1":
            relevant_by_query[row["query_id"]].add(row["doc_id"])

    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= args.top_k)
    per_query = []
    for (query_id, method), group in grouped.items():
        group.sort(key=lambda row: int(row["rank"]))
        hits = [1 if row["manual_relevant"].strip() == "1" else 0 for row in group[:args.top_k]]
        total_relevant = max(1, len(relevant_by_query[query_id]))
        per_query.append({
            "query_id": query_id,
            "method": method,
            **metrics_for_hits(hits, total_relevant, k_values),
        })

    methods = sorted({row["method"] for row in per_query})
    mean_metrics = {
        method: {
            key: mean([row for row in per_query if row["method"] == method], key)
            for k in k_values
            for key in (
                f"precision@{k}",
                f"recall@{k}",
                f"ndcg@{k}",
                f"mrr@{k}",
                f"coverage@{k}",
            )
        }
        for method in methods
    }
    summary = {
        "labeled_queries": len({row["query_id"] for row in labeled}),
        "labeled_rows": len(labeled),
        "mean_metrics": mean_metrics,
        "per_query": per_query,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manual_retrieval_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "manual_retrieval_report.md", summary, k_values)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

