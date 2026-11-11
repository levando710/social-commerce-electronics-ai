"""Standalone publication figures from saved results; unavailable experiments are not fabricated."""
import json
from pathlib import Path
import sys

from phone_assistant.recommendation.recsys_data import ROOT, sha256

sys.path.insert(0, str(ROOT / 'artifacts/cache/report_tools'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = ROOT / "docs/research_figures/extensions"
FIGURE_DPI = 180
FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 4.2
FONT_POINTS = 10
METHODS = ("graph_multimodal", "graph_multimodal_gcl")
COLORS = ("#2a78d6", "#eb6834")  # same validated blue and orange as the result charts
BASELINE = "#8a8a85"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def save(figure, name, sources, manifest):
    figure.tight_layout()
    for extension in ("png", "svg", "pdf"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
    manifest[name] = {"inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "outputs_sha256": {extension: sha256(OUTPUT / f"{name}.{extension}") for extension in ("png", "svg", "pdf")}}
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": FONT_POINTS, "font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    manifest = {"source_sha256": sha256(Path(__file__)), "matplotlib": matplotlib.__version__, "figures": {}}
    sparse = ROOT / 'artifacts/evaluation/results_electronics_robustness_analysis_v1/analysis.json'
    full = ROOT / 'artifacts/evaluation/results_electronics_multimodal_v1/experiment.json'
    if sparse.exists():
        data, baseline = json.loads(sparse.read_text()), json.loads(full.read_text())
        figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
        for ax, metric in zip(axes, ("hr@10", "ndcg@10")):
            for method, color, label in zip(METHODS, COLORS, ("Graph đa phương thức", "Graph đa phương thức + GCL")):
                points = []
                for fraction, row in sorted(data["aggregated"].items(), key=lambda pair: float(pair[0])):
                    cells = [c for c in data["cells"] if str(c["fraction"]) == fraction]
                    x = float(np.mean([c["actual_retain_fraction"] for c in cells]))
                    points.append((x, row["methods"][method][metric]))
                    ax.scatter([x] * len(cells), [c["methods"][method][metric] for c in cells], color=color, alpha=.35, s=20)
                points.append((1.0, baseline["summary"][method]["test"]["all"][metric]["mean"]))
                ax.plot([p[0] for p in points], [p[1] for p in points], "o-", color=color, label=label)
            ax.set_xlabel("Tỷ lệ cạnh huấn luyện được giữ (thực tế)")
            ax.set_ylabel(metric.upper()); ax.grid(color=GRID, linewidth=.8); ax.set_axisbelow(True); ax.legend(fontsize=8, frameon=False)
        figure.suptitle("Electronics: độ nhạy theo lượng tương tác huấn luyện (ba seed, hai mặt nạ cố định)")
        save(figure, "feedback_robustness", [sparse, full], manifest["figures"])
    anchored = ROOT / 'artifacts/evaluation/results_electronics_anchored_analysis_v1/analysis.json'
    if anchored.exists():
        data = json.loads(anchored.read_text())
        figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
        methods = list(data["summary"])
        labels = ["Detached", "Joint CE", "Joint CE + JS"]
        for ax, metric in zip(axes, ("hr@10", "ndcg@10")):
            bars = ax.bar(labels, [data["summary"][m]["query_test"][metric]["mean"] for m in methods], color=COLORS[0], width=.55)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            ax.axhline(data["frozen_text_baseline"][metric], linestyle="--", color=BASELINE, label="Văn bản cố định (đối chứng)")
            ax.set_ylabel("Query " + metric.upper()); ax.set_ylim(0, data["frozen_text_baseline"][metric] * 1.22)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
            ax.grid(axis="y", color=GRID, linewidth=.8); ax.set_axisbelow(True)
        figure.suptitle(f"Truy hồi query giữ nền ngữ nghĩa; alpha chọn theo validation = {data['selected_alpha']}")
        save(figure, "anchored_query_retrieval", [anchored], manifest["figures"])
    generation = ROOT / 'artifacts/evaluation/results_rag_generation_analysis_v1/analysis.json'
    if generation.exists():
        data = json.loads(generation.read_text()); stages = ("planned", "generated", "judged")
        figure, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
        bar_width = .35; positions = np.arange(len(stages))
        for offset, arm, color, label in zip((-.5, .5), ("text_only", "text_image"), COLORS, ("Chỉ văn bản", "Văn bản + ảnh")):
            bars = ax.bar(positions + offset * bar_width, [data["by_arm"][arm][stage] for stage in stages], width=bar_width, color=color, label=label)
            ax.bar_label(bars, padding=3)
        ax.set_xticks(positions, ["Dự kiến", "Sinh được", "Chấm hợp lệ"])
        ax.set_ylabel("Số câu trả lời"); ax.set_ylim(0, 7); ax.legend(frameon=False)
        ax.grid(axis="y", color=GRID, linewidth=.8); ax.set_axisbelow(True)
        ax.set_title("Độ bao phủ của pilot sinh: lỗi API vẫn nằm trong mẫu số")
        save(figure, "generation_coverage", [generation], manifest["figures"])
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figures": list(manifest["figures"]), "directory": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
