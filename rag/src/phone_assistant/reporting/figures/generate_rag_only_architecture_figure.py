from phone_assistant.common.project_layout import PROJECT_ROOT

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = PROJECT_ROOT
OUT = ROOT / "pricing" / "docs" / "figures" / "10_multimodal_rag_architecture_only.png"


def draw_box(ax, x, y, w, h, text, face, edge, fontsize=12, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=1.7,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#0f172a",
        wrap=True,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#0f3b66",
            shrinkA=5,
            shrinkB=5,
        )
    )


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(20, 11.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        10,
        10.45,
        "Kiến trúc thực nghiệm Multimodal RAG cho tư vấn đồ điện tử",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color="#071a36",
    )

    cols = [
        (0.8, "Nguồn dữ liệu", "#e0f2fe", "#0284c7"),
        (5.3, "Xử lý đa phương thức", "#ecfdf5", "#059669"),
        (9.8, "Truy xuất và xếp hạng", "#fff7ed", "#ea580c"),
        (14.3, "Sinh câu trả lời\nvà đánh giá", "#fef2f2", "#dc2626"),
    ]
    col_w = 3.8
    header_h = 0.78
    item_h = 0.72
    item_gap = 0.32

    for x, title, face, edge in cols:
        draw_box(ax, x, 9.0, col_w, header_h, title, face, edge, fontsize=13.2, weight="bold")

    source_items = [
        "Văn bản\nTiêu đề, mô tả",
        "Siêu dữ liệu\nBrand, rating, số review",
        "Review người dùng\nNội dung đánh giá",
        "Hình ảnh sản phẩm",
        "Video / âm thanh\nĐịnh hướng mở rộng",
    ]
    process_items = [
        "Làm sạch và\nchuẩn hóa dữ liệu",
        "Phân tích ảnh\nCaption, tag, visual evidence",
        "Xây dựng corpus\nDocument cấp sản phẩm",
        "Lập chỉ mục\nBM25 / TF-IDF concept",
        "Query Understanding\nLọc nhiễu, tạo query rewrite",
    ]
    retrieval_items = [
        "BM25 text-only",
        "BM25 + visual evidence",
        "Semantic TF-IDF",
        "Hybrid reranker",
        "Intent Hybrid",
    ]
    app_items = [
        "Gemini LLM / VLM\nTổng hợp dựa trên evidence",
        "Chatbot tư vấn\nTrả lời theo nhu cầu",
        "Hiển thị bằng chứng\nSản phẩm, review, ảnh",
        "Đánh giá retrieval\nPrecision, Recall, NDCG, MRR",
        "Báo cáo tính khả thi",
    ]

    groups = [source_items, process_items, retrieval_items, app_items]
    for (x, _, _, _), items in zip(cols, groups):
        y = 7.9
        for item in items:
            draw_box(ax, x, y, col_w, item_h, item, "#f8fafc", "#64748b", fontsize=11.2)
            y -= item_h + item_gap

    # Main horizontal flow
    y_mid = 5.85
    arrow(ax, (4.6, y_mid), (5.3, y_mid))
    arrow(ax, (9.1, y_mid), (9.8, y_mid))
    arrow(ax, (13.6, y_mid), (14.3, y_mid))

    # Emphasize feedback loop from evaluation to query understanding without crossing labels.
    ax.add_patch(
        FancyArrowPatch(
            (16.2, 3.15),
            (7.2, 2.62),
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            linestyle="--",
            color="#0f3b66",
            shrinkA=6,
            shrinkB=6,
        )
    )
    ax.text(
        11.5,
        2.35,
        "Kết quả đánh giá được dùng để tinh chỉnh query understanding và trọng số reranking",
        ha="center",
        va="center",
        fontsize=10.6,
        color="#334155",
    )

    # Bottom output band.
    draw_box(
        ax,
        1.2,
        0.55,
        17.6,
        0.85,
        "Kết quả: danh sách sản phẩm liên quan, bằng chứng từ văn bản / review / hình ảnh, và câu trả lời có căn cứ",
        "#eff6ff",
        "#2563eb",
        fontsize=14,
        weight="bold",
    )
    arrow(ax, (16.2, 2.62), (16.2, 1.43))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
