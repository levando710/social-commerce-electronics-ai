"""Own editable vector figure generated from the measured hierarchical graph schema."""
from phone_assistant.common.project_layout import MODULE_ROOT
from phone_assistant.reporting.figures.diagram_style import box, build_drawio, build_svg, edge, frame
import argparse
import json
from pathlib import Path

ROOT = MODULE_ROOT
WIDTH, HEIGHT = 1300, 760


def vn(number):
    return f"{number:,}".replace(",", ".")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / 'artifacts/evaluation/results_connected_multimodal_v2/experiment.json')
    parser.add_argument("--output", type=Path, default=ROOT / "docs/connected_recsys_figures")
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    run = next(r for r in experiment["runs"] if r["method"] == "graph_multimodal_gcl")
    counts = run["graph_schema"]["nodes"]
    edges = run["graph_schema"]["undirected_edges"]
    layers = experiment["config"]["layers"]
    frames = [
        frame(30, 100, 560, 400, "Cấu trúc đồ thị"),
        frame(620, 100, 650, 400, "Lan truyền và học"),
    ]
    boxes = {
        "user": box(50, 190, 160, 90, ["Người dùng", vn(counts["user"])]),
        "item": box(250, 190, 150, 90, ["Sản phẩm", vn(counts["item"])]),
        "text": box(440, 140, 130, 90, ["Nút văn bản", vn(counts["text"]), "MiniLM"]),
        "image": box(440, 250, 130, 90, ["Nút ảnh", vn(counts["image"]), "CLIP"]),
        "spec": box(440, 360, 130, 100, ["Giá trị", "thông số", vn(counts["spec"])]),
        "attribute": box(250, 370, 150, 90, ["Thuộc tính", vn(counts["spec_attribute"])]),
        "group": box(50, 370, 160, 90, ["Nhóm thông số", vn(counts["spec_group"])]),
        "propagation": box(650, 140, 270, 100, ["Lan truyền đồ thị", f"{layers} lớp, ma trận kề", "chuẩn hóa đối xứng"]),
        "rank": box(990, 140, 250, 100, ["Vector người dùng", "và sản phẩm", "Xếp hạng toàn catalog"]),
        "views": box(650, 290, 270, 90, ["Hai khung nhìn", "bỏ cạnh ngẫu nhiên"]),
        "loss": box(650, 410, 270, 70, ["BPR + chính quy hóa + GCL"]),
        "guard": box(30, 530, 1240, 90, ["Đồ thị huấn luyện không chứa cặp người dùng–sản phẩm bị giữ lại; "
                                         "không dùng điểm hay nhãn đánh giá làm đặc trưng văn bản;",
                                         "không suy luận các thông số còn thiếu."]),
    }
    links = [
        edge([(210, 235), (250, 235)], source="user", target="item"),
        edge([(400, 205), (420, 205), (420, 185), (440, 185)], source="item", target="text"),
        edge([(400, 235), (425, 235), (425, 295), (440, 295)], source="item", target="image"),
        edge([(400, 265), (412, 265), (412, 410), (440, 410)], source="item", target="spec"),
        edge([(325, 280), (325, 370)], "thuộc tính", source="item", target="attribute", label_at=(335, 330, "start")),
        edge([(210, 415), (250, 415)], source="group", target="attribute"),
        edge([(400, 430), (440, 430)], source="attribute", target="spec"),
        edge([(570, 185), (600, 185), (600, 175), (650, 175)], source="text", target="propagation"),
        edge([(570, 295), (610, 295), (610, 205), (650, 205)], source="image", target="propagation"),
        edge([(570, 410), (625, 410), (625, 225), (650, 225)], source="spec", target="propagation"),
        edge([(920, 190), (990, 190)], "điểm số", source="propagation", target="rank"),
        edge([(785, 240), (785, 290)], "tăng cường", source="propagation", target="views", label_at=(797, 270, "start")),
        edge([(785, 380), (785, 410)], source="views", target="loss"),
    ]
    notes = ["Các quan hệ của đồ thị được xem là vô hướng khi lan truyền. Ảnh catalog không xác nhận tình trạng thiết bị đã qua sử dụng.",
             f"{vn(edges['user_item'])} cạnh người dùng–sản phẩm huấn luyện; Co-Guiding được đánh giá riêng."]
    title = "Đồ thị đa phương thức với phân cấp thông số tường minh"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hierarchical_multimodal_graph.svg").write_text(
        build_svg(WIDTH, HEIGHT, title, boxes, links, notes, frames), encoding="utf-8")
    build_drawio(WIDTH, HEIGHT, title, boxes, links, frames).write(
        args.output / "hierarchical_multimodal_graph.drawio", encoding="utf-8", xml_declaration=True)
    print(str(args.output))


if __name__ == "__main__":
    main()
