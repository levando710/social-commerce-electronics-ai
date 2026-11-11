"""Evaluate RAG retrieval baselines with standard IR metrics.

The script uses criterion-based relevance labels from evaluation/queries.json.
It is designed for fast offline experiments: no scraping, no training, no LLM API.
"""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from pathlib import Path
import sys
import unicodedata

from phone_assistant.retrieval.search.bm25 import build_index, score
from phone_assistant.retrieval.search.hybrid_search import hybrid_search, intent_hybrid_search
from phone_assistant.retrieval.search.semantic_search import build_semantic_index, semantic_results


ROOT = MODULE_ROOT
DEFAULT_K_VALUES = (1, 3, 5, 10)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(value):
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def doc_haystack(doc):
    parts = [
        doc.get("title", ""),
        doc.get("brand", ""),
        doc.get("text", ""),
        " ".join(doc.get("categories") or []),
        json.dumps(doc.get("details") or {}, ensure_ascii=False),
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def contains_any(haystack, terms):
    return any(normalize_text(term) in haystack for term in terms)


def is_relevant(doc, relevance):
    haystack = doc_haystack(doc)

    for group in relevance.get("must_any", []):
        if not contains_any(haystack, group):
            return False

    exclude_terms = relevance.get("exclude_any", [])
    if exclude_terms and contains_any(haystack, exclude_terms):
        return False

    min_rating = relevance.get("min_rating")
    if min_rating is not None:
        rating = doc.get("average_rating")
        if rating is None or float(rating) < float(min_rating):
            return False

    min_review_count = relevance.get("min_review_count")
    if min_review_count is not None and int(doc.get("review_count") or 0) < int(min_review_count):
        return False

    if relevance.get("require_review") and int(doc.get("review_count") or 0) <= 0:
        return False

    if relevance.get("require_product_image") and not doc.get("product_image_paths"):
        return False

    if relevance.get("require_review_image") and not doc.get("review_image_paths"):
        return False

    return True


def bm25_search(index, documents_by_id, query, top_k):
    rows = []
    for doc_id, raw_score in score(index, query)[:top_k]:
        doc = documents_by_id.get(doc_id)
        if not doc:
            continue
        rows.append({
            "doc_id": doc_id,
            "score": raw_score,
            "title": doc.get("title"),
            "brand": doc.get("brand"),
            "average_rating": doc.get("average_rating"),
            "review_count": doc.get("review_count"),
            "product_images": doc.get("product_image_paths", [])[:2],
            "review_images": doc.get("review_image_paths", [])[:2],
        })
    return rows


def strip_visual_evidence(doc):
    stripped = dict(doc)
    stripped["visual_evidence"] = []
    lines = []
    for line in str(doc.get("text") or "").splitlines():
        if not line.startswith("Visual evidence:"):
            lines.append(line)
    stripped["text"] = "\n".join(lines)
    return stripped


def dcg(binary_relevance):
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(binary_relevance))


def metrics_for_ranking(ranked_ids, relevant_ids, k_values):
    relevant_set = set(relevant_ids)
    total_relevant = len(relevant_set)
    output = {}

    for k in k_values:
        top_ids = ranked_ids[:k]
        hits = [1 if doc_id in relevant_set else 0 for doc_id in top_ids]
        hit_count = sum(hits)
        ideal_hits = [1] * min(total_relevant, k)
        ideal_dcg = dcg(ideal_hits)

        first_hit_rank = 0
        for idx, hit in enumerate(hits, start=1):
            if hit:
                first_hit_rank = idx
                break

        output[f"precision@{k}"] = hit_count / k if k else 0.0
        output[f"recall@{k}"] = hit_count / total_relevant if total_relevant else 0.0
        output[f"ndcg@{k}"] = dcg(hits) / ideal_dcg if ideal_dcg else 0.0
        output[f"mrr@{k}"] = 1 / first_hit_rank if first_hit_rank else 0.0
        output[f"coverage@{k}"] = 1.0 if hit_count else 0.0

    return output


def mean_metric(rows, key):
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary, k_values):
    lines = [
        "# Báo cáo đánh giá retrieval",
        "",
        "Báo cáo này so sánh BM25 baseline với hybrid retriever hiện tại.",
        "Nhãn liên quan được xác định theo bộ tiêu chí trong `evaluation/queries.json`.",
        "Bước đánh giá này không gọi LLM; mục tiêu là đo chất lượng truy xuất evidence.",
        "",
        "## Dữ liệu đánh giá",
        "",
        f"- Số document: {summary['dataset']['documents']}",
        f"- Số truy vấn đánh giá: {summary['dataset']['queries']}",
        f"- Số document liên quan lớn nhất trên một truy vấn: {summary['dataset']['max_relevant']}",
        f"- Số document liên quan nhỏ nhất trên một truy vấn: {summary['dataset']['min_relevant']}",
        "",
        "## Metric trung bình",
        "",
    ]

    headers = ["Phương án"]
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

    lines.extend([
        "",
        "## Mức cải thiện của Hybrid so với BM25",
        "",
        "| Metric | Mức cải thiện |",
        "|---|---:|",
    ])
    for key, value in summary["hybrid_improvement"].items():
        lines.append(f"| {key} | {value:.2f}% |")

    if summary.get("intent_hybrid_improvement"):
        lines.extend([
            "",
            "## Mức cải thiện của Intent Hybrid so với Hybrid",
            "",
            "| Metric | Mức cải thiện |",
            "|---|---:|",
        ])
        for key, value in summary["intent_hybrid_improvement"].items():
            lines.append(f"| {key} | {value:.2f}% |")

    lines.extend([
        "",
        "## Diễn giải dùng trong báo cáo",
        "",
        "- Precision@K đo tỷ lệ sản phẩm liên quan trong K kết quả đầu.",
        "- Recall@K đo tỷ lệ tài liệu liên quan được tìm thấy trong K kết quả đầu.",
        "- NDCG@K đánh giá chất lượng xếp hạng, ưu tiên kết quả liên quan ở vị trí cao.",
        "- MRR đo vị trí xuất hiện của kết quả liên quan đầu tiên.",
        "- Coverage@K đo tỷ lệ truy vấn có ít nhất một kết quả liên quan trong K kết quả đầu.",
        "- Đây là đánh giá retrieval, không phải đánh giá chất lượng sinh câu trả lời của LLM.",
        "- Intent Hybrid bổ sung bước hiểu truy vấn/query rewriting trước khi truy xuất evidence.",
        "",
    ])
    experiment = summary.get("dense_experiment")
    if experiment:
        lines.extend([
            "## Cấu hình neural retrieval", "",
            f"- Fusion: {experiment['fusion']}",
            f"- Sparse input: {experiment['sparse_input']}",
            f"- Labels: {experiment['label_note']}",
            f"- Language: {experiment['language_note']}",
            "", "### Ablation trên cùng bộ truy vấn", "",
        ])
        if 5 in k_values:
            lines.extend(["| So sánh | Δ P@5 | Δ NDCG@5 | Δ MRR@5 |", "|---|---:|---:|---:|"])
            for first, second in (("sparse_dense_text", "sparse_dense_clip"),
                                  ("sparse_dense_clip", "intent_sparse_dense_clip")):
                if first not in summary["mean_metrics"] or second not in summary["mean_metrics"]:
                    continue
                differences = [summary["mean_metrics"][second][key] - summary["mean_metrics"][first][key]
                               for key in ("precision@5", "ndcg@5", "mrr@5")]
                lines.append(f"| {second} − {first} | " + " | ".join(f"{value:+.4f}" for value in differences) + " |")
        lines.extend(["", "### Độ trễ retrieval", "",
                      f"- {experiment['latency_note']}",
                      f"- Warm-up encoders: {experiment['warmup_seconds']:.2f} s.", "",
                      "| Phương án | Mean ms/query |", "|---|---:|"])
        for method, value in experiment["mean_retrieval_latency_ms"].items():
            lines.append(f"| {method} | {value:.2f} |")
        lines.extend(["", "Các chênh lệch trên là kết quả mô tả của tập thử hiện tại, chưa chứng minh ý nghĩa thống kê.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "documents.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "bm25_index.json")
    parser.add_argument("--queries", type=Path, default=ROOT / 'artifacts/evaluation' / "queries.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / 'artifacts/evaluation' / "results")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-index", type=Path, help="Add actual neural text/CLIP baselines from this index")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("top-k must be positive")
    if args.dense_index and args.output_dir.resolve() == (ROOT / 'artifacts/evaluation/results').resolve():
        args.output_dir = ROOT / 'artifacts/evaluation/results_dense'

    documents = read_jsonl(args.corpus.resolve())
    documents_by_id = {doc["doc_id"]: doc for doc in documents}
    text_only_documents = [strip_visual_evidence(doc) for doc in documents]
    text_only_index = build_index(text_only_documents)
    semantic_index = build_semantic_index(documents)
    index = json.loads(args.index.resolve().read_text(encoding="utf-8"))
    queries = json.loads(args.queries.resolve().read_text(encoding="utf-8"))
    dense = None
    if args.dense_index:
        from phone_assistant.retrieval.search.dense_search import DenseRetriever, dense_results, multimodal_search
        dense = DenseRetriever(args.dense_index, documents, device=args.device)
        # Load and warm encoders before latency measurement; model load is separate.
        warm_started = time.perf_counter()
        for channel in dense.arrays:
            dense.search("phone", channel=channel, top_k=1)
        dense_warmup_seconds = time.perf_counter() - warm_started
    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= args.top_k)
    if args.top_k not in k_values:
        k_values = tuple(sorted(set(k_values + (args.top_k,))))

    detail_rows = []
    per_query = []
    relevant_counts = []
    method_names = ["bm25_text_only", "bm25", "semantic_tfidf", "hybrid", "intent_hybrid"]
    if dense:
        method_names += ["dense_text", "sparse_dense_text"]
        if "clip" in dense.arrays:
            method_names += ["dense_clip", "sparse_dense_clip", "intent_sparse_dense_clip"]

    for item in queries:
        relevant_ids = [
            doc["doc_id"]
            for doc in documents
            if is_relevant(doc, item.get("relevance", {}))
        ]
        relevant_counts.append(len(relevant_ids))

        method_results = {
            "bm25_text_only": bm25_search(text_only_index, documents_by_id, item["query"], args.top_k),
            "bm25": bm25_search(index, documents_by_id, item["query"], args.top_k),
            "semantic_tfidf": semantic_results(semantic_index, documents_by_id, item["query"], top_k=args.top_k),
            "hybrid": hybrid_search(index, documents, item["query"], top_k=args.top_k),
            "intent_hybrid": intent_hybrid_search(index, documents, item["query"], top_k=args.top_k, use_llm=False),
        }
        latency = {}
        if dense:
            functions = {
                "dense_text": lambda: dense_results(dense, documents, item["query"], top_k=args.top_k),
                "sparse_dense_text": lambda: multimodal_search(text_only_index, documents, dense,
                    item["query"], top_k=args.top_k, use_clip=False),
            }
            if "clip" in dense.arrays:
                functions.update({
                    "dense_clip": lambda: dense_results(dense, documents, item["query"], channel="clip", top_k=args.top_k),
                    "sparse_dense_clip": lambda: multimodal_search(text_only_index, documents, dense,
                        item["query"], top_k=args.top_k),
                    "intent_sparse_dense_clip": lambda: multimodal_search(text_only_index, documents, dense,
                        item["query"], top_k=args.top_k, use_query_plan=True, use_llm=False),
                })
            for method, function in functions.items():
                # Prevent another method's query cache from making timings unfair.
                dense.query_cache.clear()
                started = time.perf_counter()
                method_results[method] = function()
                latency[method] = (time.perf_counter() - started) * 1000

        for method, results in method_results.items():
            ranked_ids = [row["doc_id"] for row in results]
            metrics = metrics_for_ranking(ranked_ids, relevant_ids, k_values)
            row = {
                "query_id": item["id"],
                "query": item["query"],
                "method": method,
                "relevant_count": len(relevant_ids),
                **metrics,
            }
            if method in latency:
                row["retrieval_latency_ms"] = latency[method]
            per_query.append(row)

            for rank, result in enumerate(results, start=1):
                detail_rows.append({
                    "query_id": item["id"],
                    "query": item["query"],
                    "method": method,
                    "rank": rank,
                    "doc_id": result["doc_id"],
                    "is_relevant": 1 if result["doc_id"] in set(relevant_ids) else 0,
                    "score": result.get("score"),
                    "bm25_score": result.get("bm25_score", result.get("score")),
                    "title": result.get("title"),
                    "brand": result.get("brand"),
                    "rating": result.get("average_rating"),
                    "review_count": result.get("review_count"),
                    "has_product_image": 1 if result.get("product_images") else 0,
                    "has_review_image": 1 if result.get("review_images") else 0,
                })

    mean_metrics = {}
    for method in method_names:
        rows = [row for row in per_query if row["method"] == method]
        mean_metrics[method] = {
            key: mean_metric(rows, key)
            for k in k_values
            for key in (
                f"precision@{k}",
                f"recall@{k}",
                f"ndcg@{k}",
                f"mrr@{k}",
                f"coverage@{k}",
            )
        }

    improvement = {}
    for key, hybrid_value in mean_metrics["hybrid"].items():
        bm25_value = mean_metrics["bm25"].get(key, 0.0)
        if bm25_value:
            improvement[key] = (hybrid_value - bm25_value) / bm25_value * 100
        else:
            improvement[key] = 0.0 if hybrid_value == 0 else 100.0

    intent_improvement = {}
    for key, intent_value in mean_metrics["intent_hybrid"].items():
        hybrid_value = mean_metrics["hybrid"].get(key, 0.0)
        if hybrid_value:
            intent_improvement[key] = (intent_value - hybrid_value) / hybrid_value * 100
        else:
            intent_improvement[key] = 0.0 if intent_value == 0 else 100.0

    summary = {
        "dataset": {
            "documents": len(documents),
            "queries": len(queries),
            "min_relevant": min(relevant_counts) if relevant_counts else 0,
            "max_relevant": max(relevant_counts) if relevant_counts else 0,
            "avg_relevant": sum(relevant_counts) / len(relevant_counts) if relevant_counts else 0,
        },
        "k_values": list(k_values),
        "mean_metrics": mean_metrics,
        "ablation_note": "bm25_text_only strips Visual evidence from documents; bm25 includes visual evidence text; hybrid adds deterministic reranking; intent_hybrid adds query understanding before retrieval.",
        "hybrid_improvement": improvement,
        "intent_hybrid_improvement": intent_improvement,
        "per_query": per_query,
    }
    if dense:
        summary["dense_experiment"] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            "sparse_index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
            "index_manifest": dense.manifest,
            "fusion": "weighted RRF(k=60), BM25=1, text=1, CLIP=0.5; candidates=80/channel",
            "sparse_input": "text-only BM25, excludes Pillow visual tags for image ablation",
            "warmup_seconds": dense_warmup_seconds,
            "mean_retrieval_latency_ms": {
                method: mean_metric([row for row in per_query if row["method"] == method], "retrieval_latency_ms")
                for method in method_names if method.startswith(("dense_", "sparse_", "intent_sparse_"))
            },
            "latency_note": "CPU wall clock, encoders warm, query cache cleared per method; excludes generation",
            "label_note": "Criterion-based relevance, not human judgments; no claims about answer faithfulness",
            "language_note": "CLIP query encoder is English-centric; no claim of multilingual visual alignment",
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if dense:
        for row in per_query:
            row.setdefault("retrieval_latency_ms", "")
    write_csv(args.output_dir / "retrieval_metrics_by_query.csv", per_query)
    write_csv(args.output_dir / "retrieval_top_results.csv", detail_rows)
    write_markdown(args.output_dir / "retrieval_evaluation_report.md", summary, k_values)

    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "dataset": summary["dataset"],
        "mean_metrics": summary["mean_metrics"],
        "hybrid_improvement": summary["hybrid_improvement"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
