"""Comparison charts for the report, drawn from the report's own result tables.

Every value is read from the numbered tables of the Markdown report, whose numbers are separately checked against the
result files by ``tools/check_report_numbers.py``; the charts therefore cannot drift from the tables.  Output: PNG for
Word, SVG and PDF as editable vector sources, and ``manifest.json`` listing each figure with the table it comes from.
"""
from phone_assistant.common.project_layout import MODULE_ROOT
import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = MODULE_ROOT
REPORT = ROOT.parent / "docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet); gray marks the baseline.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BASELINE = "#8a8a85"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def number(cell):
    cell = cell.strip().replace("—", "").replace("−", "-")
    return float(cell) if re.fullmatch(r"-?\d+\.\d+|-?\d+", cell) else None


def table_after(lines, caption_phrase):
    """Rows of the first pipe table after the caption that contains ``caption_phrase``.

    Captions are found by their wording, not their number, so inserting a table does not break the charts.
    """
    start = next(index for index, line in enumerate(lines) if line.startswith("**Bảng") and caption_phrase in line)
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[0], [row for row in rows[2:]]


def column(header, name):
    return next(index for index, label in enumerate(header) if label == name)


def hbars(ax, labels, values, errors, colors, xmax=None, fmt="{:.3f}"):
    positions = range(len(labels))
    has_error = any(error for error in errors)
    ax.barh(positions, values, xerr=[error or 0 for error in errors] if has_error else None, color=colors, height=0.62,
            ecolor=INK, error_kw={"elinewidth": 0.8, "capsize": 2.5}, linewidth=0, zorder=3)
    for position, value, error in zip(positions, values, errors):
        ax.text(value + (error or 0) + (xmax or max(values)) * 0.015, position, fmt.format(value), va="center",
                fontsize=8, color=INK)
    ax.set_yticks(list(positions), labels)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlim(0, (xmax or max(values)) * 1.22)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def save(fig, directory, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=200 if extension == "png" else None, bbox_inches="tight")
    plt.close(fig)


def panels(count, width=6.4, height=3.0, sharey=True):
    fig, axes = plt.subplots(1, count, figsize=(width, height), sharey=sharey, constrained_layout=True)
    return fig, (axes if count > 1 else [axes])


RETRIEVAL_NAMES = {"bm25": "BM25", "bm25_visual_tags": "BM25 + thẻ ảnh", "dense_text": "Vector văn bản",
                   "dense_clip": "Vector ảnh (CLIP)", "clip": "Vector ảnh (CLIP)", "hybrid": "Hybrid (đề xuất)"}
RECSYS_NAMES = {"Popularity": "Popularity", "lightgcn": "LightGCN", "vbpr": "VBPR", "bm3": "BM3",
                "graph_text_only": "Graph text-only", "graph_multimodal": "Graph đa phương thức",
                "graph_multimodal_gcl": "Graph đa phương thức + GCL"}


def retrieval_chart(lines, directory, caption, name, title, metrics):
    header, rows = table_after(lines, caption)
    labels = [RETRIEVAL_NAMES[row[0]] for row in rows]
    colors = [BASELINE if row[0] == "bm25" else SLOTS[index % len(SLOTS)] for index, row in enumerate(rows)]
    fig, axes = panels(len(metrics), width=6.4, height=2.7)
    for ax, metric in zip(axes, metrics):
        values = [number(row[column(header, metric)]) for row in rows]
        hbars(ax, labels, values, [None] * len(values), colors, fmt="{:.3f}")
        ax.set_title(metric, loc="left", fontsize=9, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, directory, name)


def recsys_chart(lines, directory, name, title, tables):
    order = list(RECSYS_NAMES)
    fig, axes = panels(len(tables), width=7.0 if len(tables) > 1 else 6.0, height=3.4, sharey=True)
    for ax, (caption, panel) in zip(axes, tables):
        header, rows = table_after(lines, caption)
        by_name = {row[0]: row for row in rows if row[0] in RECSYS_NAMES}
        present = [key for key in order if key in by_name]
        values = [number(by_name[key][1]) for key in present]
        errors = [number(by_name[key][2]) for key in present]
        colors = [BASELINE if key == "Popularity" else SLOTS[order.index(key) - 1] for key in present]
        positions = [order.index(key) for key in present]
        ax.barh(positions, values, xerr=[e or 0 for e in errors], color=colors, height=0.62, ecolor=INK,
                error_kw={"elinewidth": 0.8, "capsize": 2.5}, linewidth=0, zorder=3)
        top = max(v + (e or 0) for v, e in zip(values, errors))
        for position, value, error in zip(positions, values, errors):
            ax.text(value + (error or 0) + top * 0.03, position, f"{value:.4f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(order)), [RECSYS_NAMES[key] for key in order])
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlim(0, top * 1.45)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(panel, loc="left", fontsize=9, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.text(0.01, -0.03, "Thanh sai số: độ lệch chuẩn theo ba seed; Popularity không có seed.", fontsize=8, color=MUTED)
    save(fig, directory, name)


def coupling_chart(lines, directory):
    header, rows = table_after(lines, "Ảnh hưởng của hai hướng Co-Guiding")
    labels = [row[0].replace("unguarded", "không guard").replace("guarded", "có guard").replace("/", " · ") for row in rows]
    fig, axes = panels(2, width=6.6, height=3.4)
    for ax, (metric, title) in zip(axes, (("Feed HR@10", "Gợi ý trong feed (HR@10)"),
                                          ("Truy hồi NDCG@10", "Truy hồi sản phẩm (NDCG@10)"))):
        values = [number(row[column(header, metric)]) for row in rows]
        colors = [SLOTS[0] if "không guard" in label else SLOTS[1] for label in labels]
        hbars(ax, labels, values, [None] * len(values), colors, xmax=max(values), fmt="{:.3f}")
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    fig.suptitle("Co-Guiding ở thời điểm suy luận: xanh = không guard, cam = có guard", x=0.01, ha="left", fontsize=10)
    save(fig, directory, "fig_coguiding_modes")


def spec_chart(lines, directory):
    header, rows = table_after(lines, "Bóc tách thông số trên tập kiểm tra")
    conditions = [("Quy tắc", "Quy tắc: đúng", BASELINE), ("Chữ", "Chữ: đúng", SLOTS[0]),
                  ("Chữ + lân cận", "Chữ + lân cận: đúng", SLOTS[2]), ("Chữ + ảnh", "Chữ + ảnh: đúng", SLOTS[1])]
    fig, ax = plt.subplots(figsize=(6.6, 3.5), constrained_layout=True)
    width = 0.2
    for offset, (label, key, color) in enumerate(conditions):
        values = [number(row[column(header, key)]) for row in rows]
        positions = [index + (offset - 1.5) * width for index in range(len(rows))]
        ax.bar(positions, values, width=width * 0.9, color=color, label=label, zorder=3, linewidth=0)
        for position, value in zip(positions, values):
            ax.text(position, value + 0.012, f"{value:.2f}", ha="center", fontsize=6.5, color=INK, rotation=90)
    ax.set_xticks(range(len(rows)), [f"{row[0]}\n(n={row[1]})" for row in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("Tỷ lệ trả lời đúng")
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
    save(fig, directory, "fig_spec_extraction")


def generation_chart(lines, directory):
    header, rows = table_after(lines, "Đánh giá sinh mở rộng theo nhánh đầu vào")
    names = {"text_only": "Chỉ văn bản", "text_image": "Văn bản + ảnh"}
    metrics = [("Độ bao phủ judge", "Độ bao phủ của judge"), ("Faithfulness micro (thô)", "Faithfulness thô")]
    fig, axes = panels(2, width=6.0, height=2.2)
    for ax, (key, title) in zip(axes, metrics):
        values = [number(row[column(header, key)]) for row in rows]
        hbars(ax, [names[row[0]] for row in rows], values, [None] * len(values), [SLOTS[0], SLOTS[1]], xmax=1.0,
              fmt="{:.3f}")
        ax.set_xlim(0, 1.15)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    fig.suptitle("Đánh giá sinh trên 52 câu mỗi nhánh (judge cùng nhà cung cấp)", x=0.01, ha="left", fontsize=10)
    save(fig, directory, "fig_generation_branches")


def latency_chart(lines, directory):
    header, rows = table_after(lines, "Độ trễ từng pha")
    names = {"search": "Tìm kiếm (vector đã cache)", "text": "Mã hóa truy vấn (MiniLM)",
             "clip": "Mã hóa truy vấn (CLIP)", "anchored": "Xếp hạng anchored"}
    fig, ax = plt.subplots(figsize=(6.2, 2.5), constrained_layout=True)
    median = [number(row[column(header, "Trung vị ms")]) for row in rows]
    p95 = [number(row[column(header, "p95 ms")]) for row in rows]
    positions = range(len(rows))
    ax.barh([p - 0.2 for p in positions], median, height=0.36, color=SLOTS[0], label="Trung vị", zorder=3)
    ax.barh([p + 0.2 for p in positions], p95, height=0.36, color=SLOTS[1], label="p95", zorder=3)
    for position, a, b in zip(positions, median, p95):
        ax.text(a + 1, position - 0.2, f"{a:.1f}", va="center", fontsize=8)
        ax.text(b + 1, position + 0.2, f"{b:.1f}", va="center", fontsize=8)
    ax.set_yticks(list(positions), [names[row[0]] for row in rows])
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlim(0, max(p95) * 1.2)
    ax.set_xlabel("ms (CPU, từng pha riêng lẻ)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    save(fig, directory, "fig_latency")


FIGURES = [
    ("fig_retrieval_electronics", "bảng cùng tên", "Truy hồi trên Electronics"),
    ("fig_retrieval_vogue", "bảng cùng tên", "Truy hồi first-turn trên VOGUE"),
    ("fig_recsys_datasets", "bảng cùng tên", "Gợi ý trên ba bộ dữ liệu"),
    ("fig_coguiding_modes", "bảng cùng tên", "Chế độ Co-Guiding"),
    ("fig_spec_extraction", "bảng cùng tên", "Bóc tách thông số theo điều kiện bằng chứng"),
    ("fig_generation_branches", "bảng cùng tên", "Đánh giá sinh theo nhánh đầu vào"),
    ("fig_latency", "bảng cùng tên", "Độ trễ từng pha"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research_figures/results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lines = args.report.read_text(encoding="utf-8").splitlines()
    retrieval_chart(lines, args.output, "Hiệu năng truy hồi trên tập kiểm tra Electronics", "fig_retrieval_electronics",
                    "Truy hồi trên tập kiểm tra Electronics (76 truy vấn)", ["R@10", "NDCG@10", "MRR@10"])
    retrieval_chart(lines, args.output, "Hiệu năng truy hồi first-turn trên VOGUE", "fig_retrieval_vogue",
                    "Truy hồi first-turn trên VOGUE (60 hội thoại)", ["R@10", "NDCG@10", "MRR@10"])
    recsys_chart(lines, args.output, "fig_recsys_datasets", "HR@10 trên ba bộ dữ liệu gợi ý (ba seed)",
                 [("Kết quả gợi ý Electronics trung bình", "Electronics"), ("Kết quả gợi ý trên Amazon Baby_Products", "Baby_Products"), ("Kết quả gợi ý trên MIND-small", "MIND-small")])
    coupling_chart(lines, args.output)
    spec_chart(lines, args.output)
    generation_chart(lines, args.output)
    latency_chart(lines, args.output)
    manifest = [{"figure": name, "source_table": table, "title": title,
                 "files": [f"{name}.{extension}" for extension in ("png", "svg", "pdf")]} for name, table, title in FIGURES]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(args.output))


if __name__ == "__main__":
    main()
