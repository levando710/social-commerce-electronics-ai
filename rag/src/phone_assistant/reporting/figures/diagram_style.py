"""Shared drawing style for the report's block diagrams (SVG and draw.io).

Style: white background, black outlines, serif text, titled group frames, right-angled arrows (solid for the main
flow, dashed for feedback or planned parts) and edge labels on a white halo so that a line never strikes through text.
"""
from html import escape
import xml.etree.ElementTree as ET

FONT = "Times New Roman, Liberation Serif, serif"
INK, FRAME_INK, ARROW = "#1a1a1a", "#444444", "#1a1a1a"
BOX_FONT, LABEL_FONT, CAPTION_FONT, TITLE_FONT, LINE_HEIGHT = 21, 18, 21, 30, 27


def box(x, y, w, h, lines, dashed=False):
    return {"x": x, "y": y, "w": w, "h": h, "lines": lines, "dashed": dashed}


def frame(x, y, w, h, caption):
    return {"x": x, "y": y, "w": w, "h": h, "caption": caption}


def edge(points, label="", dashed=False, source=None, target=None, label_at=None):
    return {"points": points, "label": label, "dashed": dashed, "source": source, "target": target,
            "label_at": label_at}


def label_position(points):
    """Beside the longest segment: above a horizontal one, to the right of a vertical one."""
    segments = list(zip(points, points[1:]))
    (x1, y1), (x2, y2) = max(segments, key=lambda s: abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]))
    if y1 == y2:
        return (x1 + x2) / 2, y1 - 10, "middle"
    return x1 + 10, (y1 + y2) / 2 + 6, "start"


def build_svg(width, height, title, boxes, edges, notes=(), frames=()):
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">'
           f'<path d="M0,0 L0,6 L9,3 z" fill="{ARROW}"/></marker></defs>',
           f'<text x="{width / 2}" y="52" text-anchor="middle" font-family="{FONT}" font-size="{TITLE_FONT}">{escape(title)}</text>']
    for item in frames:
        svg.append(f'<rect x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" height="{item["h"]}" fill="none" '
                   f'stroke="{FRAME_INK}" stroke-width="1.5"/>')
        svg.append(f'<text x="{item["x"] + 18}" y="{item["y"] + 30}" font-family="{FONT}" font-size="{CAPTION_FONT}" '
                   f'fill="{FRAME_INK}">{escape(item["caption"])}</text>')
    for item in edges:
        points = " ".join(f"{x},{y}" for x, y in item["points"])
        dash = ' stroke-dasharray="9 6"' if item["dashed"] else ""
        svg.append(f'<polyline points="{points}" fill="none" stroke="{ARROW}" stroke-width="2"{dash} marker-end="url(#arrow)"/>')
    for item in boxes.values():
        dash = ' stroke-dasharray="9 6"' if item["dashed"] else ""
        svg.append(f'<rect x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" height="{item["h"]}" fill="white" '
                   f'stroke="{INK}" stroke-width="1.6"{dash}/>')
        lines = item["lines"]
        top = item["y"] + item["h"] / 2 - (len(lines) - 1) * LINE_HEIGHT / 2 + 7
        for index, line in enumerate(lines):
            svg.append(f'<text x="{item["x"] + item["w"] / 2}" y="{top + index * LINE_HEIGHT}" text-anchor="middle" '
                       f'font-family="{FONT}" font-size="{BOX_FONT}">{escape(line)}</text>')
    for item in edges:
        if item["label"]:
            x, y, anchor = item["label_at"] or label_position(item["points"])
            if item["label_at"] and len(item["label_at"]) == 2:
                anchor = "middle"
            svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{LABEL_FONT}" '
                       f'fill="{FRAME_INK}" stroke="white" stroke-width="6" paint-order="stroke" '
                       f'stroke-linejoin="round">{escape(item["label"])}</text>')
    for index, note in enumerate(notes):
        svg.append(f'<text x="40" y="{height - 50 + index * 26}" font-family="{FONT}" font-size="{LABEL_FONT + 1}">{escape(note)}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def build_drawio(width, height, name, boxes, edges, frames=()):
    document = ET.Element("mxfile", host="app.diagrams.net")
    diagram = ET.SubElement(document, "diagram", name=name)
    model = ET.SubElement(diagram, "mxGraphModel", grid="1", pageWidth=str(width), pageHeight=str(height))
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    for index, item in enumerate(frames):
        style = (f"rounded=0;whiteSpace=wrap;html=0;fillColor=none;strokeColor={FRAME_INK};fontSize={CAPTION_FONT};"
                 "fontFamily=Times New Roman;verticalAlign=top;align=left;spacingLeft=14;spacingTop=6;")
        cell = ET.SubElement(root, "mxCell", id=f"frame{index}", value=item["caption"], style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(item["x"]), y=str(item["y"]), width=str(item["w"]),
                      height=str(item["h"]), attrib={"as": "geometry"})
    for key, item in boxes.items():
        style = (f"rounded=0;whiteSpace=wrap;html=0;fillColor=#ffffff;strokeColor={INK};fontSize={BOX_FONT};"
                 "fontFamily=Times New Roman;")
        if item["dashed"]:
            style += "dashed=1;"
        cell = ET.SubElement(root, "mxCell", id=key, value="\n".join(item["lines"]), style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(item["x"]), y=str(item["y"]), width=str(item["w"]),
                      height=str(item["h"]), attrib={"as": "geometry"})
    for index, item in enumerate(edges):
        if not (item["source"] and item["target"]):
            continue
        style = f"edgeStyle=orthogonalEdgeStyle;endArrow=block;strokeColor={ARROW};html=0;fontFamily=Times New Roman;"
        if item["dashed"]:
            style += "dashed=1;"
        cell = ET.SubElement(root, "mxCell", id=f"edge{index}", value=item["label"], source=item["source"],
                             target=item["target"], style=style, edge="1", parent="1")
        ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
    return ET.ElementTree(document)
