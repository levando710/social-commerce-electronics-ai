"""Own editable vector figure of the multimodal evidence pipeline (SVG and draw.io).

Solid boxes are implemented and evaluated in this study (text and images).  Dashed boxes are designed extensions that
are NOT implemented (video through keyframes, audio through speech-to-text).
"""
from phone_assistant.common.project_layout import MODULE_ROOT
from phone_assistant.reporting.figures.diagram_style import box, build_drawio, build_svg, edge, frame
import argparse
from pathlib import Path

ROOT = MODULE_ROOT
WIDTH, HEIGHT = 1300, 760
TITLE = "Pipeline bằng chứng đa phương thức"
FRAMES = [
    frame(45, 100, 210, 520, "1. Nguồn dữ liệu"),
    frame(295, 100, 210, 520, "2. Bộ mã hóa"),
    frame(545, 100, 210, 520, "3. Chỉ mục"),
    frame(795, 100, 210, 520, "4. Truy vấn"),
    frame(1045, 100, 210, 520, "5. Đầu ra"),
]
BOXES = {
    "text": box(65, 150, 170, 90, ["Văn bản và", "thông số"]),
    "image": box(65, 270, 170, 90, ["Ảnh sản phẩm"]),
    "video": box(65, 390, 170, 90, ["Video", "(chưa cài đặt)"], dashed=True),
    "audio": box(65, 510, 170, 90, ["Âm thanh", "(chưa cài đặt)"], dashed=True),
    "minilm": box(315, 150, 170, 90, ["Mã hóa văn bản", "MiniLM"]),
    "clip": box(315, 270, 170, 90, ["Mã hóa ảnh", "CLIP ViT-B/32"]),
    "frames": box(315, 390, 170, 90, ["Lấy khung hình,", "mã hóa CLIP", "(dự kiến)"], dashed=True),
    "asr": box(315, 510, 170, 90, ["Chuyển giọng nói", "thành văn bản", "(dự kiến)"], dashed=True),
    "index": box(565, 205, 170, 130, ["Chỉ mục lai", "đa phương thức", "BM25, vector", "và RRF"]),
    "future": box(565, 450, 170, 120, ["Bằng chứng video", "và âm thanh", "(dự kiến)"], dashed=True),
    "query": box(815, 150, 170, 110, ["Hiểu truy vấn", "Ý định, ràng buộc", "Viết lại truy vấn"]),
    "guard": box(815, 330, 170, 130, ["Xếp hạng lại theo", "ràng buộc:", "satisfied /", "unknown / violated"]),
    "answer": box(1065, 150, 170, 130, ["Câu trả lời", "có viện dẫn", "Từ chối khi thiếu", "bằng chứng"]),
    "recommender": box(1065, 330, 170, 90, ["Hệ gợi ý đồ thị", "GCL, Co-Guiding"]),
}
EDGES = [
    edge([(235, 195), (315, 195)], source="text", target="minilm"),
    edge([(235, 315), (315, 315)], source="image", target="clip"),
    edge([(235, 435), (315, 435)], dashed=True, source="video", target="frames"),
    edge([(235, 555), (315, 555)], dashed=True, source="audio", target="asr"),
    edge([(485, 195), (525, 195), (525, 262), (565, 262)], source="minilm", target="index"),
    edge([(485, 315), (525, 315), (525, 298), (565, 298)], source="clip", target="index"),
    edge([(485, 435), (525, 435), (525, 495), (565, 495)], dashed=True, source="frames", target="future"),
    edge([(485, 555), (525, 555), (525, 535), (565, 535)], dashed=True, source="asr", target="future"),
    edge([(650, 450), (650, 330)], "dự kiến", dashed=True, source="future", target="index", label_at=(662, 396, "start")),
    edge([(815, 205), (775, 205), (775, 262), (735, 262)], source="query", target="index"),
    edge([(735, 298), (775, 298), (775, 390), (815, 390)], source="index", target="guard"),
    edge([(900, 260), (900, 330)], "ràng buộc", source="query", target="guard", label_at=(912, 302, "start")),
    edge([(985, 360), (1025, 360), (1025, 215), (1065, 215)], source="guard", target="answer"),
    edge([(985, 405), (1065, 405)], source="guard", target="recommender"),
]
NOTES = ["Nét liền: đã cài đặt và đánh giá (văn bản, ảnh). Nét đứt: hướng mở rộng chưa cài đặt (video, âm thanh); không kết quả nào phụ thuộc.",
         "Ảnh catalog không xác nhận tình trạng của một thiết bị đã qua sử dụng."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research_figures/multimodal_pipeline")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "multimodal_pipeline.svg").write_text(
        build_svg(WIDTH, HEIGHT, TITLE, BOXES, EDGES, NOTES, FRAMES), encoding="utf-8")
    build_drawio(WIDTH, HEIGHT, TITLE, BOXES, EDGES, FRAMES).write(
        args.output / "multimodal_pipeline.drawio", encoding="utf-8", xml_declaration=True)
    print(str(args.output))


if __name__ == "__main__":
    main()
