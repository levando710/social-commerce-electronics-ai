"""Generate original editable SVG/draw.io diagrams from the completed graph schema."""
from phone_assistant.common.project_layout import MODULE_ROOT
from html import escape
import json
import xml.etree.ElementTree as ET

ROOT = MODULE_ROOT


def main():
    report = json.loads((ROOT / 'artifacts/evaluation/results_recsys/experiment.json').read_text(encoding="utf-8"))
    run = next(run for run in report["runs"] if run["method"] == "graph_multimodal_gcl")
    counts, edges = run["graph_schema"]["nodes"], run["graph_schema"]["undirected_edges"]
    boxes = {
        "users": (40, 150, 220, 90, f"Users: {counts['user']:,}", "#e8f1ff"),
        "items": (340, 150, 220, 90, f"Items: {counts['item']:,}", "#e8f1ff"),
        "text": (650, 35, 240, 80, f"Metadata text: {counts['text']:,}", "#e5f6ef"),
        "image": (650, 155, 240, 80, f"CLIP images: {counts['image']:,}", "#fff0dc"),
        "spec": (650, 275, 240, 80, f"Spec values: {counts['spec']:,}", "#f0eafe"),
        "propagate": (1000, 150, 240, 110, "Normalized propagation / 2 layers", "#e8f1ff"),
        "views": (1000, 340, 240, 80, "Two edge-dropout views", "#fff0dc"),
        "loss": (1000, 490, 240, 80, "BPR + regularization + InfoNCE", "#f0eafe"),
        "rank": (1320, 150, 220, 110, "User-item scores / full-catalog ranking", "#e5f6ef"),
        "guard": (40, 340, 520, 100, "Train edges only / metadata without reviews or ratings", "#f5f6f8"),
    }
    links = [("users", "items", f"train: {edges['user_item']:,}"),
             ("items", "text", "item-text"), ("items", "image", "item-image"),
             ("items", "spec", "item-spec"), ("text", "propagate", "text projection"),
             ("image", "propagate", "image projection"), ("spec", "propagate", "spec embedding"),
             ("propagate", "rank", "export vectors"), ("propagate", "views", "augmentation"),
             ("views", "loss", "node alignment")]
    out = ROOT / "docs/recsys_figures"
    out.mkdir(parents=True, exist_ok=True)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="650" viewBox="0 0 1600 650">',
           '<rect width="1600" height="650" fill="white"/>',
           '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L9,3 L0,6" fill="#516078"/></marker></defs>',
           '<text x="40" y="30" font-family="Arial" font-size="23">Multimodal heterogeneous graph + GCL — implemented RecSys branch</text>']
    for source, target, label in links:
        x, y, w, h, _, _ = boxes[source]
        tx, ty, tw, th, _, _ = boxes[target]
        if ty > y + h and tx == x:
            x1, y1, x2, y2 = x + w / 2, y + h, tx + tw / 2, ty
        else:
            x1, y1, x2, y2 = x + w, y + h / 2, tx, ty + th / 2
        svg.extend([f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#516078" stroke-width="2" marker-end="url(#arrow)"/>',
                    f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-8}" text-anchor="middle" font-family="Arial" font-size="12">{escape(label)}</text>'])
    for x, y, w, h, label, color in boxes.values():
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{color}" stroke="#7e90aa"/>')
        for row, text in enumerate(label.split(" / ")):
            svg.append(f'<text x="{x+w/2}" y="{y+h/2+row*22-5}" text-anchor="middle" font-family="Arial" font-size="17">{escape(text)}</text>')
    svg.extend(['<text x="40" y="610" font-family="Arial" font-size="16">Prototype: static catalog, train-only interactions. RAG/Co-Guiding and full specs hierarchy are not implemented here.</text>', '</svg>'])
    (out / "heterogeneous_graph_gcl.svg").write_text("\n".join(svg), encoding="utf-8")
    document = ET.Element("mxfile", host="app.diagrams.net")
    diagram = ET.SubElement(document, "diagram", name="Implemented RecSys graph")
    model = ET.SubElement(diagram, "mxGraphModel", dx="1600", dy="650", grid="1", pageWidth="1600", pageHeight="650")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    for name, (x, y, w, h, label, color) in boxes.items():
        cell = ET.SubElement(root, "mxCell", id=name, value=label.replace(" / ", "\n"),
            style=f"rounded=1;whiteSpace=wrap;html=0;fillColor={color};fontSize=17;", vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), attrib={"as": "geometry"})
    for index, (source, target, label) in enumerate(links):
        cell = ET.SubElement(root, "mxCell", id=f"edge{index}", value=label, source=source, target=target,
                             style="edgeStyle=orthogonalEdgeStyle;endArrow=block;html=0;", edge="1", parent="1")
        ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
    ET.ElementTree(document).write(out / "heterogeneous_graph_gcl.drawio", encoding="utf-8", xml_declaration=True)
    print(f"Generated editable SVG and draw.io source in {out}")


if __name__ == "__main__":
    main()
