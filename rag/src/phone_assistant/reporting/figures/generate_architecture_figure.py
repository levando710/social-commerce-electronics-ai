"""Own editable vector figure of the overall architecture: four blocks and their data flows (SVG and draw.io)."""
from phone_assistant.common.project_layout import MODULE_ROOT
from phone_assistant.reporting.figures.diagram_style import box, build_drawio, build_svg, edge, frame
import argparse
from pathlib import Path

ROOT = MODULE_ROOT
WIDTH, HEIGHT = 1300, 1470
TITLE = "Kiến trúc tổng thể của hệ thống"
FRAMES = [
    frame(60, 90, 1180, 130, "Dữ liệu đầu vào"),
    frame(60, 270, 780, 430, "Khối 1. Biểu diễn tri thức và chỉ mục"),
    frame(880, 270, 360, 430, "Khối 3. Hệ gợi ý trên đồ thị"),
    frame(60, 850, 640, 430, "Khối 2. Truy hồi lai"),
    frame(740, 850, 500, 430, "Khối 4. Co-Guiding"),
]
BOXES = {
    "catalog": box(100, 135, 330, 60, ["Catalog, thông số, đánh giá"]),
    "image": box(520, 135, 240, 60, ["Ảnh sản phẩm"]),
    "inter": box(900, 135, 310, 60, ["Tương tác huấn luyện"]),
    "norm": box(115, 330, 300, 60, ["Chuẩn hóa trường dữ liệu"]),
    "minilm": box(95, 440, 200, 80, ["Vector văn bản", "MiniLM"]),
    "hier": box(330, 440, 290, 80, ["Phân cấp", "nhóm → thuộc tính → giá trị"]),
    "clip": box(655, 440, 165, 80, ["Vector ảnh", "CLIP ViT-B/32"]),
    "index": box(330, 600, 290, 70, ["Chỉ mục lai", "BM25 + vector"]),
    "graph": box(910, 350, 300, 110, ["Đồ thị dị thể", "Người dùng · sản phẩm", "Nội dung · thông số"]),
    "gnn": box(910, 560, 300, 80, ["GNN + học tương phản", "(GCL)"]),
    "query": box(100, 750, 260, 70, ["Truy vấn và ngữ cảnh", "hội thoại"]),
    "understand": box(130, 920, 200, 60, ["Hiểu truy vấn"]),
    "rrf": box(150, 1050, 340, 70, ["Truy hồi và hợp nhất hạng", "(RRF)"]),
    "guard": box(110, 1170, 420, 80, ["Xếp hạng lại theo ràng buộc", "satisfied / unknown / violated"]),
    "r2r": box(740, 930, 300, 80, ["RecSys → RAG", "Sở thích làm prior"]),
    "c2r": box(740, 1130, 320, 80, ["RAG → RecSys", "Bằng chứng định hướng feed"]),
    "answer": box(180, 1350, 280, 70, ["Sinh câu trả lời", "có viện dẫn"]),
    "feed": box(900, 1350, 240, 60, ["Danh sách gợi ý"]),
}
EDGES = [
    edge([(412, 195), (412, 330)], source="catalog", target="norm"),
    edge([(195, 390), (195, 440)], source="norm", target="minilm"),
    edge([(380, 390), (380, 440)], source="norm", target="hier"),
    edge([(700, 195), (700, 440)], source="image", target="clip"),
    edge([(160, 520), (160, 635), (330, 635)], source="minilm", target="index"),
    edge([(760, 520), (760, 635), (620, 635)], source="clip", target="index"),
    edge([(840, 400), (910, 400)], "vector, phân cấp", source="norm", target="graph", label_at=(834, 384, "end")),
    edge([(1170, 195), (1170, 350)], source="inter", target="graph"),
    edge([(1060, 460), (1060, 560)], source="graph", target="gnn"),
    edge([(300, 820), (300, 920)], source="query", target="understand"),
    edge([(300, 980), (300, 1050)], source="understand", target="rrf"),
    edge([(450, 670), (450, 1050)], source="index", target="rrf"),
    edge([(320, 1120), (320, 1170)], source="rrf", target="guard"),
    edge([(960, 640), (960, 930)], dashed=True, source="gnn", target="r2r"),
    edge([(740, 970), (560, 970), (560, 1085), (490, 1085)], "prior sở thích", dashed=True, source="r2r", target="rrf",
         label_at=(620, 955, "middle")),
    edge([(530, 1190), (740, 1190)], dashed=True, source="guard", target="c2r"),
    edge([(320, 1250), (320, 1350)], "bằng chứng có nguồn", source="guard", target="answer", label_at=(335, 1315, "start")),
    edge([(900, 1210), (900, 1350)], dashed=True, source="c2r", target="feed"),
    edge([(1210, 600), (1272, 600), (1272, 1380), (1140, 1380)], source="gnn", target="feed"),
]
NOTES = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research_figures/architecture")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "kien_truc_tong_the.svg").write_text(
        build_svg(WIDTH, HEIGHT, TITLE, BOXES, EDGES, NOTES, FRAMES), encoding="utf-8")
    build_drawio(WIDTH, HEIGHT, TITLE, BOXES, EDGES, FRAMES).write(
        args.output / "kien_truc_tong_the.drawio", encoding="utf-8", xml_declaration=True)
    print(str(args.output))


if __name__ == "__main__":
    main()
