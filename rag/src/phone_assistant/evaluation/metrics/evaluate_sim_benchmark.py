"""Evaluate paired first-query/history retrieval; tune fusion on validation only."""
from __future__ import annotations
from phone_assistant.common.project_layout import source_path

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from phone_assistant.retrieval.search.bm25 import build_index, score
from phone_assistant.recommendation.recsys_data import InteractionData, ROOT, read_jsonl, sha256


def rank_target(scores, target, blocked):
    values = np.asarray(scores, dtype=np.float64).copy()
    if not np.isfinite(values).all() or target in blocked:
        raise ValueError("Invalid scores or held-out target already seen")
    if blocked:
        values[list(blocked)] = -np.inf
    order = np.lexsort((np.arange(len(values)), -values))
    return int(np.flatnonzero(order == target)[0]) + 1


def aggregate(rows):
    # Personas of one user share the same target; they are not independent users.
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row["rank"])
    result = {"users": len(by_user), "queries": len(rows)}
    for k in (5, 10, 20):
        result[f"hr@{k}"] = float(np.mean([np.mean([r <= k for r in ranks]) for ranks in by_user.values()])) if rows else None
        result[f"ndcg@{k}"] = float(np.mean([np.mean([1 / math.log2(r + 1) if r <= k else 0 for r in ranks]) for ranks in by_user.values()])) if rows else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / 'data/external/benchmarks/phones_v1')
    parser.add_argument("--output", type=Path, default=ROOT / 'artifacts/evaluation/results_sim_phones_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; do not retune on an evaluated test set")
    data = InteractionData(args.benchmark)  # Verifies every frozen artifact, not just train edges.
    documents = read_jsonl(args.benchmark / "documents.jsonl")
    if [d["doc_id"] for d in documents] != data.item_ids:
        raise ValueError("Catalog order mismatch")
    profiles = read_jsonl(args.benchmark / "agent_profiles.jsonl")
    queries = read_jsonl(args.benchmark / "agent_queries.jsonl")
    labels = {r["user_id"]: r for r in read_jsonl(args.benchmark / "evaluator_labels.jsonl")}
    vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, dtype=np.float32)
    vectors = vectorizer.fit_transform([d["text"] for d in documents])
    profile_vectors = {}
    for profile in profiles:
        item_indices = [data.maps["items"][r["item_id"]] for r in profile["history"]]
        # Only prior product metadata; no review text, hidden goals or simulator prompts.
        profile_vectors[profile["user_id"]] = normalize(np.asarray(vectors[item_indices].mean(axis=0)))
    index = build_index(documents)
    weights = (0.0, 0.1, 0.25, 0.5, 1.0)
    elapsed = defaultdict(float)

    def evaluate(split, selected_weights):
        rows = defaultdict(list)
        for query in queries:
            if query["split"] != split:
                continue
            uid = query["user_id"]
            user_index = data.maps["users"][uid]
            blocked = set(data.train_seen[user_index])
            if split == "test":
                blocked.update(data.validation_seen[user_index])
            target = data.maps["items"][labels[uid]["item_id"]]
            started = time.perf_counter()
            bm25_scores = np.zeros(data.n_items, dtype=np.float32)
            for item, value in score(index, query["query"]):
                bm25_scores[data.maps["items"][item]] = value
            bm25_scores /= max(float(bm25_scores.max()), 1e-12)
            profile_scores = np.asarray(vectors @ profile_vectors[uid].T).ravel()
            elapsed[split] += time.perf_counter() - started
            methods = {"bm25": bm25_scores, "history_tfidf": profile_scores}
            methods.update({f"fusion_{weight:g}": bm25_scores + weight * profile_scores for weight in selected_weights})
            for method, values in methods.items():
                rows[method].append({"user_id": uid, "dialogue_id": query["dialogue_id"],
                                     "rank": rank_target(values, target, blocked),
                                     "group": "warm_item" if data.item_degree[target] else "cold_item"})
        return rows

    validation = evaluate("validation", weights)
    validation_metrics = {method: aggregate(rows) for method, rows in validation.items()}
    # Earliest/lower personalization weight for ties. No test measurements used here.
    chosen = max(weights, key=lambda w: (validation_metrics[f"fusion_{w:g}"]["ndcg@10"],
                                        validation_metrics[f"fusion_{w:g}"]["hr@10"], -w))
    test = evaluate("test", [chosen])
    test_metrics = {method: {group: aggregate([r for r in rows if group == "all" or r["group"] == group])
                            for group in ("all", "warm_item", "cold_item")} for method, rows in test.items()}
    result = {"benchmark_manifest_sha256": sha256(args.benchmark / "manifest.json"),
              "source_sha256": {name: sha256(source_path(name)) for name in
                                ("prepare_sim_benchmark.py", "evaluate_sim_benchmark.py", "bm25.py", "recsys_data.py")},
              "selection": {"weights": weights, "chosen_weight": chosen, "criterion": "validation user-macro NDCG@10, then HR@10, then smaller weight"},
              "validation": validation_metrics, "test": test_metrics, "test_ranks": dict(test),
              "scoring_seconds_excluding_initialization_and_sorting": dict(elapsed),
              "packages": {"numpy": np.__version__, "scikit-learn": __import__("sklearn").__version__},
              "limitations": ["First-query offline evaluation, not an interactive Co-Guiding experiment.",
                  "Single hidden target labels: HR/NDCG measure target recovery, not precision of all recommendations.",
                  "Two simulated personas per user are averaged within user, then across users.",
                  "Catalog includes accessories; metadata snapshot is not historically time-verified.",
                  "No GNN or LLM training in this experiment; no claim of human user benefit."]}
    args.output.mkdir(parents=True)
    (args.output / "experiment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Store the actual experiment source alongside hashes to make later edits harmless.
    source_dir = args.output / "source"
    source_dir.mkdir()
    for name in result["source_sha256"]:
        (source_dir / name).write_bytes((source_path(name)).read_bytes())
    domain = json.loads((args.benchmark / "summary.json").read_text(encoding="utf-8"))["domain"]
    lines = [f"# Tìm kiếm theo câu hỏi và lịch sử người dùng — {domain}", "",
             "Đánh giá câu hỏi đầu tiên; không phải thử nghiệm hội thoại nhiều lượt hoặc đánh giá người dùng thật.", "",
             f"Trọng số lịch sử chọn bằng validation: **{chosen:g}**. Test chỉ chạy sau khi chọn.", "",
             "| Phương án | Người test | Câu hỏi | HR@10 | NDCG@10 | HR@20 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, groups in test_metrics.items():
        m = groups["all"]
        lines.append(f"| {method} | {m['users']} | {m['queries']} | {m['hr@10']:.4f} | {m['ndcg@10']:.4f} | {m['hr@20']:.4f} |")
    lines += ["", "Mỗi người có hai hội thoại mô phỏng; tính trung bình trong từng người trước khi tính toàn tập.",
              "Nhãn chỉ có một sản phẩm đích, nên không dùng kết quả này để kết luận các sản phẩm khác là sai.", "",
              "## Giới hạn", ""] + [f"- {item}" for item in result["limitations"]]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
