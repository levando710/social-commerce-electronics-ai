"""Score provided manual ratings, keeping unreviewed rows and failed answers unscored."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from phone_assistant.recommendation.recsys_data import ROOT, sha256, read_jsonl


def read_ratings(path, expected_pairs):
    ratings, unknown, seen = {}, 0, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "item_id", "manual_relevance"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing query_id/item_id/manual_relevance columns")
        for row in reader:
            pair = (row["query_id"], row["item_id"])
            if pair in seen:
                raise ValueError(f"Duplicate manual query/item row: {pair}")
            seen.add(pair)
            if pair not in expected_pairs:
                raise ValueError(f"Rating outside the prepared review pool: {pair}")
            label = row["manual_relevance"].strip().lower()
            if label in ("", "uncertain"):
                unknown += 1
                continue
            if label not in ("0", "1", "2"):
                raise ValueError("manual_relevance must be 0, 1, 2, uncertain or blank")
            if not row.get("reviewer", "").strip():
                raise ValueError(f"Filled ratings need a reviewer name/code: {pair}")
            ratings[pair] = int(label)
    return ratings, unknown, len(seen)


def score_rankings(rankings, ratings, pools, k=5):
    per_query = {}
    for method, rows in rankings.items():
        per_query[method] = {}
        for row in rows:
            qid = row["id"]
            ids = row["ids"][:k]
            if any((qid, item) not in ratings for item in ids):
                continue  # Unknown never becomes a zero relevance vote.
            grades = [ratings[qid, item] for item in ids]
            complete_pool = all((qid, item) in ratings for item in pools[qid])
            ndcg = None
            if complete_pool:
                ideal = sorted((ratings[qid, item] for item in pools[qid]), reverse=True)[:k]
                dcg = lambda values: sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(values))
                denominator = dcg(ideal)
                ndcg = dcg(grades) / denominator if denominator else None
            per_query[method][qid] = {"intent_id": row["intent_id"], "precision_strict@5": sum(v == 2 for v in grades) / k,
                "precision_including_partial@5": sum(v >= 1 for v in grades) / k, "pooled_ndcg@5": ndcg}
    common = set.intersection(*(set(values) for values in per_query.values())) if per_query else set()
    summary = {}
    for method, values in per_query.items():
        summary[method] = {"fully_rated_top5_queries": len(values), "common_queries": len(common), "metrics_on_common_queries": {}}
        for metric in ("precision_strict@5", "precision_including_partial@5", "pooled_ndcg@5"):
            groups = defaultdict(list)
            for qid in common:
                value = values[qid][metric]
                if value is not None:
                    groups[values[qid]["intent_id"]].append(value)
            summary[method]["metrics_on_common_queries"][metric] = sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None
    return summary, sorted(common)


def answer_rating_summary(path, expected_answers=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "raw_answer", "manual_faithfulness", "manual_relevance", "manual_citation_correctness", "reviewer"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing answer-rating columns")
        rows = list(reader)
    metrics = {key: [] for key in ("manual_faithfulness", "manual_relevance", "manual_citation_correctness")}
    seen = set()
    for row in rows:
        if row["query_id"] in seen:
            raise ValueError("Duplicate answer query row")
        seen.add(row["query_id"])
        if expected_answers is not None:
            original = expected_answers.get(row["query_id"])
            if original is None or row["raw_answer"].replace("\r\n", "\n").strip() != (original.get("answer") or "").replace("\r\n", "\n").strip():
                raise ValueError("Answer-rating row differs from the saved original answer")
        for key in metrics:
            text = row[key].strip().lower()
            if text in ("", "uncertain"):
                continue
            if not row.get("raw_answer", "").strip():
                raise ValueError("A failed/missing answer must not receive a quality rating")
            if text not in ("0", "1", "2") or not row.get("reviewer", "").strip():
                raise ValueError("Answer ratings must be 0/1/2 with reviewer; blanks remain unreviewed")
            metrics[key].append(int(text))
    generated = sum(bool(r.get("answer")) for r in expected_answers.values()) if expected_answers is not None else sum(bool(r.get("raw_answer", "").strip()) for r in rows)
    return {"requested_rows": len(expected_answers) if expected_answers is not None else len(rows),
        "provided_rows": len(rows), "unprovided_rows": len(expected_answers) - len(seen) if expected_answers is not None else 0,
        "generated_rows": generated,
        "ratings": {key: {"rated": len(values), "mean_0_to_2": sum(values) / len(values) if values else None} for key, values in metrics.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / 'artifacts/evaluation/handset_human_review_v1/retrieval_labels.csv')
    parser.add_argument("--answers", type=Path, default=ROOT / 'artifacts/evaluation/handset_human_review_v1/answer_labels.csv')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_handset_manual_progress_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep prior annotation progress; use a new output")
    source = ROOT / 'artifacts/evaluation/results_handset_rag_test_v1/experiment.json'
    experiment = json.loads(source.read_text(encoding="utf-8"))
    rankings = {method: [r for r in experiment["rows"][method] if r["kind"] == "retrieval" and r["id"].endswith("_0")]
        for method in ("bm25", "rewrite_guard")}
    pools = defaultdict(set)
    for values in rankings.values():
        for row in values:
            pools[row["id"]].update(row["ids"][:5])
    pairs = {(qid, item) for qid, items in pools.items() for item in items}
    ratings, unknown, present = read_ratings(args.labels, pairs)
    summary, common = score_rankings(rankings, ratings, pools)
    answers_source = ROOT / 'artifacts/evaluation/results_handset_answers_v1/records.jsonl'
    expected_answers = {r["query_id"]: r for r in read_jsonl(answers_source)}
    record = {"status": "awaiting_human_labels" if not ratings else "partial_or_complete_manual_ratings",
        "source_sha256": sha256(Path(__file__)), "labels_sha256": sha256(args.labels), "answers_sha256": sha256(args.answers),
        "retrieval_experiment_sha256": sha256(source), "pool_pairs": len(pairs), "rows_present": present,
        "rows_missing": len(pairs) - present, "rated_pairs": len(ratings), "uncertain_or_blank_rows": unknown,
        "grade_counts": dict(Counter(ratings.values())), "comparable_queries": common, "methods": summary,
        "answers_source_sha256": sha256(answers_source), "answers": answer_rating_summary(args.answers, expected_answers),
        "limitations": ["Provided manual labels, not independently verified reviewer identities.",
            "Only common fully reviewed top5 queries enter method comparisons; coverage is reported.",
            "NDCG ideal is the judged two-method candidate pool, not the full catalog. No full-catalog recall claim.",
            "Metrics macro-average over available intent groups. Unreviewed/uncertain labels are not negative votes.",
            "No quality scores for missing generated answers; offline catalog summaries are not evaluated as LLM outputs."]}
    args.output.mkdir(parents=True)
    (args.output / "progress.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Tiến độ đánh giá thủ công", "", f"Trạng thái: {record['status']}",
        f"Đã chấm {len(ratings)}/{len(pairs)} cặp; {len(common)} câu đủ nhãn để so sánh công bằng.",
        "", "Nhãn trống/uncertain không được tính là sai. Khi chưa có nhãn, các metric là null.", "",
        "```json", json.dumps(summary, indent=2), "```", ""] + record["limitations"]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "rated_pairs": len(ratings), "pool_pairs": len(pairs), "comparable_queries": len(common)}, indent=2))


if __name__ == "__main__":
    main()
