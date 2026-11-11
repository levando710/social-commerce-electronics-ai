"""Generate an original editable diagram of the implemented handset RAG serving path."""
from html import escape
import xml.etree.ElementTree as ET

from phone_assistant.recommendation.recsys_data import ROOT


def main():
    output = ROOT / "docs/handset_figures"
    output.mkdir(parents=True, exist_ok=True)
    boxes = {
        "query": (40, 260, ["User query / follow-up", "Vietnamese or English"], "#e8f1ff"),
        "state": (340, 260, ["Dialogue state", "Brand, capacity, soft needs"], "#e8f1ff"),
        "retrieval": (640, 260, ["BM25 + text + CLIP", "Rank fusion; 80 candidates"], "#e5f6ef"),
        "guard": (940, 260, ["Explicit fact constraints", "Unknown / suspect != verified"], "#fff0dc"),
        "context": (1240, 260, ["Evidence + product IDs", "Exact field quotes / images"], "#e5f6ef"),
        "generation": (1540, 260, ["External LLM", "Citation normalization"], "#f0eafe"),
        "metadata": (340, 70, ["Phone catalog metadata", "No review/rating input"], "#f5f6f8"),
        "encoders": (640, 70, ["Frozen MiniLM / CLIP", "Metadata / actual photos"], "#e5f6ef"),
        "graph": (340, 500, ["Train-history recommender", "Separate offline/browser profiles"], "#e8f1ff"),
        "feed": (940, 500, ["Query-conditioned feed", "Seen / fact filters + reward"], "#fff0dc"),
        "evaluation": (1240, 500, ["Offline quality evaluation", "Automatic vs human labels"], "#f5f6f8"),
        "answer": (1540, 500, ["Answer + cited products", "Failures remain failures"], "#f0eafe"),
    }
    edges = [("query", "state", "parse / carry state"), ("state", "retrieval", "query rewrite"),
        ("retrieval", "guard", "candidates"), ("guard", "context", "eligible evidence"),
        ("context", "generation", "grounded context"), ("generation", "answer", "generated text"),
        ("metadata", "encoders", "allowlisted metadata"), ("encoders", "retrieval", "vectors"),
        ("metadata", "guard", "field evidence"), ("graph", "retrieval", "optional Rec to RAG"),
        ("graph", "feed", "base feed"), ("guard", "feed", "optional RAG to Rec"),
        ("answer", 'artifacts/evaluation', "saved answers / claims")]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1820" height="720" viewBox="0 0 1820 720">',
        '<rect width="1820" height="720" fill="white"/>',
        '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L9,3 L0,6" fill="#516078"/></marker></defs>',
        '<text x="40" y="32" font-family="Arial" font-size="24">Handset-only conversational RAG and optional inference coupling</text>']
    for source, target, label in edges:
        sx, sy, _, _ = boxes[source]; tx, ty, _, _ = boxes[target]
        if source == "metadata" and target == "guard":
            svg += ['<polyline points="460,160 460,200 1060,200 1060,260" fill="none" stroke="#516078" stroke-width="2" marker-end="url(#a)"/>',
                    '<text x="1030" y="188" text-anchor="middle" font-family="Arial" font-size="12">field evidence</text>']
            continue
        if sx == tx:
            x1, y1, x2, y2 = sx + 120, sy + (90 if ty > sy else 0), tx + 120, ty + (0 if ty > sy else 90)
        else:
            x1, y1, x2, y2 = sx + (240 if tx > sx else 0), sy + 45, tx + (0 if tx > sx else 240), ty + 45
        label_x = (x1 + x2) / 2 + (55 if sx == tx else 0)
        label_y = sy - 12 if sy == ty else (y1 + y2) / 2 + (20 if sx == tx else -9)
        svg += [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#516078" stroke-width="2" marker-end="url(#a)"/>',
                f'<text x="{label_x}" y="{label_y}" text-anchor="middle" font-family="Arial" font-size="12">{escape(label)}</text>']
    for x, y, lines, color in boxes.values():
        svg.append(f'<rect x="{x}" y="{y}" width="240" height="90" rx="9" fill="{color}" stroke="#7e90aa"/>')
        for i, line in enumerate(lines):
            svg.append(f'<text x="{x+120}" y="{y+36+22*i}" text-anchor="middle" font-family="Arial" font-size="15">{escape(line)}</text>')
    svg += ['<text x="40" y="675" font-family="Arial" font-size="15">Catalog criteria and functional checks are diagnostics. Human labels remain separate. Source-faithful text can still repeat incorrect metadata.</text>', '</svg>']
    (output / "handset_conversational_rag.svg").write_text("\n".join(svg), encoding="utf-8")
    document = ET.Element("mxfile", host="app.diagrams.net")
    diagram = ET.SubElement(document, "diagram", name="Handset conversational RAG")
    model = ET.SubElement(diagram, "mxGraphModel", pageWidth="1820", pageHeight="720")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0"); ET.SubElement(root, "mxCell", id="1", parent="0")
    for name, (x, y, lines, color) in boxes.items():
        cell = ET.SubElement(root, "mxCell", id=name, value="\n".join(lines), style=f"rounded=1;whiteSpace=wrap;html=0;fillColor={color};fontSize=15;", vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width="240", height="90", attrib={"as": "geometry"})
    for i, (source, target, label) in enumerate(edges):
        cell = ET.SubElement(root, "mxCell", id=f"e{i}", value=label, source=source, target=target, style="edgeStyle=orthogonalEdgeStyle;endArrow=block;html=0;", edge="1", parent="1")
        ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
    ET.ElementTree(document).write(output / "handset_conversational_rag.drawio", encoding="utf-8", xml_declaration=True)
    print(output)


if __name__ == "__main__":
    main()
