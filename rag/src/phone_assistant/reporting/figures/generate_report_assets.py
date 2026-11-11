from __future__ import annotations
from phone_assistant.common.project_layout import PROJECT_ROOT

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = PROJECT_ROOT
RAG_DIR = ROOT / "rag"
PRICING_DIR = ROOT / "pricing"

METRICS_PATH = RAG_DIR / 'artifacts/evaluation' / "results" / "retrieval_metrics.json"
COVERAGE_PATH = RAG_DIR / 'artifacts/evaluation' / "results" / "multimodal_coverage.json"
SUMMARY_PATH = RAG_DIR / 'artifacts/runtime/catalog' / "summary.json"
BY_QUERY_PATH = RAG_DIR / 'artifacts/evaluation' / "results" / "retrieval_metrics_by_query.csv"
MODEL_COMPARISON_PATH = PRICING_DIR / "artifacts/reports" / "model_comparison_results.csv"

OUT_DIR = PRICING_DIR / "docs" / "figures"
SUMMARY_OUT = PRICING_DIR / "docs" / "REPORT_FIGURES_AND_STATS.md"


METHOD_LABELS = {
    "bm25_text_only": "BM25 text-only",
    "bm25": "BM25 + visual evidence",
    "semantic_tfidf": "Semantic TF-IDF",
    "hybrid": "Hybrid retrieval",
    "intent_hybrid": "Intent Hybrid",
}

METHOD_COLORS = {
    "bm25_text_only": "#6b7280",
    "bm25": "#2563eb",
    "semantic_tfidf": "#0891b2",
    "hybrid": "#ef4444",
    "intent_hybrid": "#7c3aed",
}

RETRIEVAL_METHODS = ["bm25_text_only", "bm25", "semantic_tfidf", "hybrid", "intent_hybrid"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> Path:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def percent_text(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_percent(text: str) -> float:
    return float(str(text).replace("%", "").replace(",", ".").strip())


def parse_vnd(text: str) -> float:
    return float(str(text).replace("VNĐ", "").replace(",", "").replace(".", "").strip())


def draw_box(ax, xy, text, color="#eef2ff", edge="#4f46e5", width=2.9, height=0.72):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.5)
    return patch


def draw_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color="#334155",
            shrinkA=4,
            shrinkB=4,
        )
    )


def chart_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Pipeline tong quat Multimodal RAG cho tu van do dien tu", fontsize=16, weight="bold", pad=16)

    draw_box(ax, (0.5, 6.4), "Truy van\nnguoi dung", "#ecfeff", "#0891b2")
    draw_box(ax, (4.0, 6.4), "Phan tich y dinh\nva chuan hoa query", "#f0fdf4", "#16a34a")
    draw_box(ax, (7.5, 6.4), "Truy xuat ung vien\nBM25 / Hybrid", "#fff7ed", "#ea580c")
    draw_box(ax, (11.0, 6.4), "LLM tong hop\ncau tra loi", "#fef2f2", "#dc2626")

    draw_arrow(ax, (3.4, 6.76), (4.0, 6.76))
    draw_arrow(ax, (6.9, 6.76), (7.5, 6.76))
    draw_arrow(ax, (10.4, 6.76), (11.0, 6.76))

    sources = [
        ("Text\nTitle, mo ta,\nreview", 0.5, 4.55, "#eff6ff", "#2563eb"),
        ("Metadata\nbrand, rating,\nreview count", 3.9, 4.55, "#f8fafc", "#475569"),
        ("Image\ncaption/tag/\nvisual evidence", 7.25, 4.55, "#f0fdf4", "#16a34a"),
        ("Video/Audio\nframe, OCR,\ntranscript", 10.65, 4.55, "#faf5ff", "#9333ea"),
    ]
    for text, x, y, c, e in sources:
        draw_box(ax, (x, y), text, c, e, width=2.85, height=1.05)
        draw_arrow(ax, (x + 1.42, y + 1.05), (8.95, 6.4))

    draw_box(ax, (1.2, 2.55), "Document store\nva chi muc BM25", "#f8fafc", "#334155", width=3.2, height=0.85)
    draw_box(ax, (5.25, 2.55), "Hop nhat diem\ntext + image + metadata", "#fff7ed", "#ea580c", width=3.2, height=0.85)
    draw_box(ax, (9.3, 2.55), "Reranking va loc\nbang chung", "#fefce8", "#ca8a04", width=3.2, height=0.85)
    draw_arrow(ax, (4.4, 2.98), (5.25, 2.98))
    draw_arrow(ax, (8.45, 2.98), (9.3, 2.98))
    draw_arrow(ax, (10.9, 3.4), (12.4, 6.4))

    ax.text(
        7,
        1.25,
        "Ket qua: danh sach san pham lien quan, evidence tu text/review/anh, va cau tra loi co can cu",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    return savefig("01_multimodal_rag_pipeline.png")


def chart_query_understanding_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Bo sung lop Query Understanding truoc Retrieval", fontsize=16, weight="bold", pad=16)

    steps = [
        ("Cau hoi tu nhien\nco the nhieu nhieu", "#f8fafc", "#475569"),
        ("Gemini / rules\nphan tich y dinh", "#faf5ff", "#7c3aed"),
        ("Query rewrite\nkeywords, tags, filters", "#eff6ff", "#2563eb"),
        ("Hybrid retrieval\ntext + visual evidence", "#fff7ed", "#ea580c"),
        ("LLM tra loi\nkem evidence", "#fef2f2", "#dc2626"),
    ]
    xs = [0.35, 3.05, 5.75, 8.45, 11.15]
    for i, ((text, color, edge), x) in enumerate(zip(steps, xs)):
        draw_box(ax, (x, 4.75), text, color, edge, width=2.25, height=1.0)
        if i < len(steps) - 1:
            draw_arrow(ax, (x + 2.25, 5.25), (xs[i + 1], 5.25))

    draw_box(ax, (1.0, 2.55), "Vi du query:\n'Mua cho me may pin trau,\ndung Zalo, chup chau ro'", "#f8fafc", "#64748b", width=3.7, height=1.0)
    draw_box(ax, (5.15, 2.55), "Y dinh cot loi:\nbattery_good, camera_good,\neasy_use, budget", "#faf5ff", "#7c3aed", width=3.7, height=1.0)
    draw_box(ax, (9.3, 2.55), "Retrieval query:\nsmartphone battery camera\ngood rating review", "#eff6ff", "#2563eb", width=3.7, height=1.0)
    draw_arrow(ax, (4.7, 3.05), (5.15, 3.05))
    draw_arrow(ax, (8.85, 3.05), (9.3, 3.05))
    ax.text(
        7,
        1.25,
        "Muc tieu: giu lai tin hieu quan trong, bo tu nhieu, va bien nhu cau nguoi dung thanh dieu kien truy xuat.",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    return savefig("01b_query_understanding_pipeline.png")


def chart_coverage(coverage: dict) -> Path:
    labels = ["Text documents", "Product images", "Visual evidence", "Reviews", "Review images"]
    counts = [
        coverage["text_documents"],
        coverage["documents_with_product_images"],
        coverage["documents_with_visual_evidence"],
        coverage["documents_with_reviews"],
        coverage["documents_with_review_images"],
    ]
    total = coverage["documents"]
    percents = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(labels))
    bars = ax.barh(y, percents, color=["#0f766e", "#2563eb", "#16a34a", "#f97316", "#64748b"])
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Ty le phu tren tap document (%)")
    ax.set_title("Do phu cac nguon du lieu trong tap RAG", fontsize=15, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    for bar, count, pct in zip(bars, counts, percents):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{count}/{total} ({pct:.1f}%)", va="center")
    return savefig("02_multimodal_data_coverage.png")


def chart_visual_tags(coverage: dict) -> Path:
    tags = coverage.get("top_visual_tags", {})
    top = list(tags.items())[:12]
    labels = [k.replace("_", " ") for k, _ in top][::-1]
    values = [v for _, v in top][::-1]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.barh(labels, values, color="#22c55e")
    ax.set_title("Cac visual tag pho bien trich xuat tu anh san pham", fontsize=15, weight="bold")
    ax.set_xlabel("So luong anh/document")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(values):
        ax.text(v + 5, i, str(v), va="center")
    return savefig("03_top_visual_tags.png")


def chart_retrieval_at5(metrics: dict) -> Path:
    mean = metrics["mean_metrics"]
    measures = ["precision@5", "recall@5", "ndcg@5", "mrr@5", "coverage@5"]
    labels = ["Precision@5", "Recall@5", "NDCG@5", "MRR@5", "Coverage@5"]
    x = np.arange(len(labels))
    width = 0.14

    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = np.linspace(-2 * width, 2 * width, len(RETRIEVAL_METHODS))
    for offset, method in zip(offsets, RETRIEVAL_METHODS):
        values = [mean[method][m] for m in measures]
        ax.bar(x + offset, values, width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Gia tri trung binh")
    ax.set_title("So sanh cac phuong an retrieval tai Top-5", fontsize=15, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    return savefig("04_retrieval_metrics_at5.png")


def chart_improvement(metrics: dict) -> Path:
    improvement = metrics["hybrid_improvement"]
    keys = ["precision@5", "recall@5", "ndcg@5", "mrr@5", "coverage@5"]
    labels = ["Precision@5", "Recall@5", "NDCG@5", "MRR@5", "Coverage@5"]
    values = [improvement[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in values]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_ylabel("Muc cai thien so voi BM25 + visual evidence (%)")
    ax.set_title("Improvement cua Hybrid retrieval tai Top-5", fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.4 if v >= 0 else -0.8), f"{v:.1f}%", ha="center")
    return savefig("05_hybrid_improvement_at5.png")


def chart_metrics_by_k(metrics: dict) -> Path:
    ks = metrics["k_values"]
    mean = metrics["mean_metrics"]
    measures = [("precision", "Precision"), ("ndcg", "NDCG"), ("mrr", "MRR"), ("coverage", "Coverage")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, measures):
        for method in RETRIEVAL_METHODS:
            values = [mean[method][f"{metric}@{k}"] for k in ks]
            ax.plot(ks, values, marker="o", linewidth=2, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Bien thien metric theo Top-K", fontsize=15, weight="bold")
    return savefig("06_retrieval_metrics_by_k.png")


def chart_query_distribution(by_query: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    unique_queries = by_query.drop_duplicates("query_id")
    axes[0].hist(unique_queries["relevant_count"], bins=16, color="#2563eb", alpha=0.85)
    axes[0].set_title("Phan phoi so document lien quan theo query")
    axes[0].set_xlabel("So document lien quan")
    axes[0].set_ylabel("So query")
    axes[0].grid(axis="y", alpha=0.25)

    methods = RETRIEVAL_METHODS
    data = [by_query[by_query["method"] == m]["precision@5"] for m in methods]
    axes[1].boxplot(data, tick_labels=[METHOD_LABELS[m] for m in methods], patch_artist=True)
    axes[1].set_title("Phan tan Precision@5 theo query")
    axes[1].set_ylabel("Precision@5")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Thong ke bo query danh gia retrieval", fontsize=15, weight="bold")
    return savefig("07_query_evaluation_distribution.png")


def chart_pricing_model_comparison(df: pd.DataFrame) -> Path:
    work = df.copy()
    work["MAPE"] = work["MAPE (%)"].map(parse_percent)
    work["HighConf_MAPE"] = work["HighConf MAPE (%)"].map(parse_percent)
    labels = [str(x).replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "").replace("6. ", "") for x in work["Mô hình"]]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.bar(x - width / 2, work["MAPE"], width, label="MAPE toan tap", color="#64748b")
    ax.bar(x + width / 2, work["HighConf_MAPE"], width, label="MAPE nhom do tin cay cao", color="#16a34a")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("So sanh sai so cac mo hinh dinh gia dien thoai", fontsize=15, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, row in work.iterrows():
        ax.text(i - width / 2, row["MAPE"] + 0.6, f"{row['MAPE']:.1f}", ha="center", fontsize=8)
        ax.text(i + width / 2, row["HighConf_MAPE"] + 0.6, f"{row['HighConf_MAPE']:.1f}", ha="center", fontsize=8)
    return savefig("08_pricing_model_mape_comparison.png")


def chart_pricing_runtime(df: pd.DataFrame) -> Path:
    work = df.copy()
    work["Thoi_gian"] = work["Thời gian (s)"].astype(str).str.replace("s", "", regex=False).str.replace(",", ".", regex=False).astype(float)
    labels = [str(x).replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "").replace("6. ", "") for x in work["Mô hình"]]

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.barh(labels[::-1], work["Thoi_gian"].iloc[::-1], color="#f97316")
    ax.set_xlabel("Thoi gian huan luyen / so sanh (giay)")
    ax.set_title("Chi phi thoi gian cua cac mo hinh dinh gia", fontsize=15, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(work["Thoi_gian"].iloc[::-1]):
        ax.text(v + max(work["Thoi_gian"]) * 0.01, i, f"{v:.2f}s", va="center")
    return savefig("09_pricing_model_runtime.png")


def chart_system_layers() -> Path:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Kien truc thuc nghiem ket hop pricing model va Multimodal RAG", fontsize=15, weight="bold")

    layers = [
        ("Nguon du lieu", ["Cho Tot listings", "Amazon Reviews", "Product images", "Review text"]),
        ("Xu ly du lieu", ["Clean/normalize", "LLM extraction", "Image analysis", "Corpus builder"]),
        ("Mo hinh", ["CatBoost pricing", "BM25 index", "Hybrid reranker", "Gemini LLM"]),
        ("Ung dung", ["Dinh gia dien thoai", "Chatbot tu van", "Evidence display", "Bao cao danh gia"]),
    ]
    x_positions = [0.6, 4.0, 7.4, 10.8]
    for idx, (title, items) in enumerate(layers):
        x = x_positions[idx]
        draw_box(ax, (x, 5.75), title, "#e0f2fe", "#0284c7", width=2.75, height=0.62)
        y = 4.75
        for item in items:
            draw_box(ax, (x, y), item, "#f8fafc", "#64748b", width=2.75, height=0.48)
            y -= 0.72
        if idx < len(layers) - 1:
            draw_arrow(ax, (x + 2.78, 3.6), (x_positions[idx + 1], 3.6))
    return savefig("10_system_layers.png")


def build_summary(
    metrics: dict,
    coverage: dict,
    model_df: pd.DataFrame,
    generated: list[Path],
) -> None:
    mean = metrics["mean_metrics"]
    dataset = metrics["dataset"]
    lines: list[str] = []
    lines.append("# Danh muc so do va thong ke co the chen vao bao cao\n")
    lines.append("File nay duoc sinh tu ket qua kiem thu hien co, khong chay lai train hoac cao du lieu.\n")
    lines.append("## 1. Thong ke dataset RAG\n")
    lines.append(f"- So document RAG: **{coverage['documents']:,}**")
    lines.append(f"- So query danh gia: **{dataset['queries']}**")
    lines.append(f"- Document co review: **{coverage['documents_with_reviews']:,}** ({coverage['coverage_percent']['reviews']:.2f}%)")
    lines.append(f"- Document co anh san pham: **{coverage['documents_with_product_images']:,}** ({coverage['coverage_percent']['product_images']:.2f}%)")
    lines.append(f"- Document co visual evidence: **{coverage['documents_with_visual_evidence']:,}** ({coverage['coverage_percent']['visual_evidence']:.2f}%)")
    lines.append(f"- Anh da phan tich thanh cong: **{coverage['image_analysis_success']:,}**")
    lines.append("\n## 2. Ket qua retrieval tai Top-5\n")
    lines.append("| Phuong an | Precision@5 | Recall@5 | NDCG@5 | MRR@5 | Coverage@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in RETRIEVAL_METHODS:
        row = mean[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {percent_text(row['precision@5'])} | "
            f"{percent_text(row['recall@5'])} | {percent_text(row['ndcg@5'])} | "
            f"{percent_text(row['mrr@5'])} | {percent_text(row['coverage@5'])} |"
        )
    lines.append("\n## 3. Muc cai thien cua Hybrid retrieval so voi BM25 + visual evidence tai Top-5\n")
    for key in ["precision@5", "recall@5", "ndcg@5", "mrr@5", "coverage@5"]:
        lines.append(f"- {key}: **{metrics['hybrid_improvement'][key]:.2f}%**")

    if metrics.get("intent_hybrid_improvement"):
        lines.append("\n## 3b. Tac dong cua Query Understanding so voi Hybrid\n")
        for key in ["precision@1", "mrr@5", "recall@5", "precision@5", "ndcg@5"]:
            lines.append(f"- {key}: **{metrics['intent_hybrid_improvement'][key]:.2f}%**")

    lines.append("\n## 4. Ket qua dinh gia dien thoai\n")
    lines.append("| Mo hinh | MAPE toan tap | RMSE | MAPE high-confidence | RMSE high-confidence | Thoi gian |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in model_df.iterrows():
        lines.append(
            f"| {row['Mô hình']} | {row['MAPE (%)']} | {row['RMSE (VNĐ)']} | "
            f"{row['HighConf MAPE (%)']} | {row['HighConf RMSE (VNĐ)']} | {row['Thời gian (s)']} |"
        )

    lines.append("\n## 5. Danh muc hinh anh da sinh\n")
    for path in generated:
        rel = path.relative_to(PRICING_DIR)
        lines.append(f"- `{rel}`")

    lines.append("\n## 6. Goi y chen hinh vao bao cao\n")
    suggestions = [
        ("01_multimodal_rag_pipeline.png", "Chen vao phan nguyen ly/pipeline Multimodal RAG."),
        ("01b_query_understanding_pipeline.png", "Chen vao phan Gemini/LLM hieu truy van va loc nhieu."),
        ("02_multimodal_data_coverage.png", "Chen vao phan mo ta dataset va do phu modality."),
        ("04_retrieval_metrics_at5.png", "Chen vao phan thuc nghiem so sanh retrieval."),
        ("05_hybrid_improvement_at5.png", "Chen ngay sau bang metric de lam ro muc cai thien."),
        ("07_query_evaluation_distribution.png", "Chen vao phan thiet ke bo query danh gia."),
        ("08_pricing_model_mape_comparison.png", "Chen vao phan ket qua mo hinh dinh gia."),
        ("10_system_layers.png", "Chen vao phan kien truc tong the he thong."),
    ]
    for fig, note in suggestions:
        lines.append(f"- `{fig}`: {note}")

    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output()
    metrics = load_json(METRICS_PATH)
    coverage = load_json(COVERAGE_PATH)
    load_json(SUMMARY_PATH)
    by_query = pd.read_csv(BY_QUERY_PATH)
    model_df = pd.read_csv(MODEL_COMPARISON_PATH)

    generated = [
        chart_pipeline(),
        chart_query_understanding_pipeline(),
        chart_coverage(coverage),
        chart_visual_tags(coverage),
        chart_retrieval_at5(metrics),
        chart_improvement(metrics),
        chart_metrics_by_k(metrics),
        chart_query_distribution(by_query),
        chart_pricing_model_comparison(model_df),
        chart_pricing_runtime(model_df),
        chart_system_layers(),
    ]
    build_summary(metrics, coverage, model_df, generated)
    print(f"Generated {len(generated)} figures in {OUT_DIR}")
    print(f"Summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
