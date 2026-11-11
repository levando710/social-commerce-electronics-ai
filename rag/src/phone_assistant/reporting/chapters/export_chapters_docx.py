"""Export the editable Markdown draft to Word with native OMML equations, not formula screenshots."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Cm, Emu, Pt
from lxml import etree

from phone_assistant.recommendation.recsys_data import ROOT, sha256

REPORT_TOOL_DIRECTORY = ROOT / 'artifacts/cache/report_tools'
DEFAULT_MATH_XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
BODY_FONT_POINTS = 13
LONG_TOKEN_CHARACTERS = 18
CODE_FONT_POINTS = 10
TABLE_FONT_POINTS = 11
BODY_LINE_SPACING = 1.3
LEFT_MARGIN_CM = 2.5
RIGHT_MARGIN_CM = 2
VERTICAL_MARGIN_CM = 2
MAXIMUM_HEADING_LEVEL = 4
PAGE_WIDTH_CM, PAGE_HEIGHT_CM = 21, 29.7
FIGURE_WIDTH_CM = 16
INLINE_TOKEN = re.compile(r"\*\*.*?\*\*|\*[^*\n]+\*|`[^`]+`|\\\(.*?\\\)|\[[^\]]+\]\([^)]+\)")
DISPLAY_BLOCK = re.compile(r"^\$\$[ \t]*\n(.*?)\n[ \t]*\$\$[ \t]*$", re.M | re.S)
INLINE_DOLLAR_MATH = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)")
NUMBERED_ITEM = re.compile(r"\d+\.\s+(.*)")


def normalize_markdown(text):
    """Accept the report dialect: multi-line $$ blocks and inline $...$ math become the exporter's one-line forms."""
    text = DISPLAY_BLOCK.sub(lambda m: "$$ " + " ".join(part.strip() for part in m.group(1).splitlines()) + " $$", text)
    lines = [line if line.lstrip().startswith("$$") else INLINE_DOLLAR_MATH.sub(lambda m: "\\(" + m.group(1) + "\\)", line)
             for line in text.splitlines()]
    return "\n".join(lines)


COVER_BORDER_EMU = (5937738, 9378629)
COVER_ANCHOR = (
    '<wp:anchor xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" '
    'distL="0" distR="0" simplePos="0" relativeHeight="251659264" behindDoc="1" locked="0" layoutInCell="1" '
    'allowOverlap="1"><wp:simplePos x="0" y="0"/><wp:positionH relativeFrom="margin"><wp:posOffset>-104922</wp:posOffset>'
    '</wp:positionH><wp:positionV relativeFrom="page"><wp:posOffset>258201</wp:posOffset></wp:positionV></wp:anchor>')


def add_cover(document, specification_path):
    """Cover page: ornamental border behind the text, school logo, titles, member table, place and year."""
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    folder = specification_path.parent

    def line(text="", size=14, bold=False, space_after=0):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_after = Pt(space_after)
        if text:
            run = paragraph.add_run(text)
            run.font.name, run.font.size, run.bold = "Times New Roman", Pt(size), bold
        return paragraph

    holder = line()
    picture = holder.add_run().add_picture(str(folder / specification["border"]), width=Emu(COVER_BORDER_EMU[0]),
                                           height=Emu(COVER_BORDER_EMU[1]))
    drawing = picture._inline.getparent()
    anchor = parse_xml(COVER_ANCHOR)
    for child in list(picture._inline):
        if child.tag.split("}")[1] in ("extent", "effectExtent", "docPr", "cNvGraphicFramePr", "graphic"):
            anchor.append(child)
        if child.tag.endswith("}extent"):
            anchor.append(OxmlElement("wp:wrapNone"))
    drawing.replace(picture._inline, anchor)
    line(specification["institution"], 14, True)
    line(specification["faculty"], 14, True)
    line("_" * 35, 14)
    line(space_after=6)
    logo = line()
    logo.add_run().add_picture(str(folder / specification["logo"]), width=Cm(3.4))
    line(space_after=18)
    line(specification["document"], 18, True, 12)
    line(specification["course"], 14, True, 24)
    line(specification["topic"], 14, True, 36)
    line(specification["scope"], 13, False, 18)
    line(specification["members_caption"], 13, True, 6)
    rows = specification["members"]
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = 1
    for row_index, values in enumerate(rows):
        for cell, value in zip(table.rows[row_index].cells, values):
            cell.text = ""
            run = cell.paragraphs[0].add_run(value)
            run.font.name, run.font.size, run.bold = "Times New Roman", Pt(13), row_index == 0
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
    line(space_after=60)
    line(specification["place_year"], 14, False)
    document.add_page_break()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.docx")
    parser.add_argument("--math-xsl", type=Path, default=DEFAULT_MATH_XSL)
    parser.add_argument("--cover", type=Path, default=ROOT.parent / "docs/reports/cover/cover.json",
                        help="cover page specification; skipped when the file does not exist")
    args = parser.parse_args()
    sys.path.insert(0, str(REPORT_TOOL_DIRECTORY))
    from latex2mathml.converter import convert
    if not args.math_xsl.exists():
        raise FileNotFoundError("Supply the installed Office MathML-to-OMML stylesheet through --math-xsl")
    transform = etree.XSLT(etree.parse(str(args.math_xsl)))
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(PAGE_WIDTH_CM), Cm(PAGE_HEIGHT_CM)
    section.left_margin, section.right_margin = Cm(LEFT_MARGIN_CM), Cm(RIGHT_MARGIN_CM)
    section.top_margin = section.bottom_margin = Cm(VERTICAL_MARGIN_CM)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Times New Roman", Pt(BODY_FONT_POINTS)
    normal.paragraph_format.line_spacing = BODY_LINE_SPACING
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3", "Heading 4"):
        document.styles[name].font.name = "Times New Roman"
    document.core_properties.title = "Chương 1–2: Multimodal GNN–GCL–RAG"
    document.core_properties.author = ""
    counts = {"display_equations": 0, "inline_equations": 0, "tables": 0, "figures": 0}
    if args.cover.exists():
        add_cover(document, args.cover)
        section.different_first_page_header_footer = True

    def add_math(paragraph, latex, display=False):
        latex = latex.replace(r"\mathbb1", r"\mathbf{1}")
        root = etree.fromstring(convert(latex).encode("utf-8"))
        converted = transform(root).getroot()
        if converted is None: raise ValueError(f"Failed equation conversion: {latex}")
        paragraph._p.append(deepcopy(converted))
        counts["display_equations" if display else "inline_equations"] += 1

    def inline(paragraph, text):
        cursor = 0
        for match in INLINE_TOKEN.finditer(text):
            paragraph.add_run(text[cursor:match.start()])
            token = match.group()
            if token.startswith(r"\("):
                add_math(paragraph, token[2:-2])
            elif token.startswith("**"):
                paragraph.add_run(token[2:-2]).bold = True
            elif token.startswith("*"):
                paragraph.add_run(token[1:-1].replace("`", "")).italic = True
            elif token.startswith("`"):
                run = paragraph.add_run(token[1:-1]); run.font.name = "Consolas"; run.font.size = Pt(CODE_FONT_POINTS)
            else:
                link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
                relationship = paragraph.part.relate_to(link.group(2), RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
                element = OxmlElement("w:hyperlink"); element.set(qn("r:id"), relationship)
                run, properties, style = OxmlElement("w:r"), OxmlElement("w:rPr"), OxmlElement("w:rStyle")
                style.set(qn("w:val"), "Hyperlink"); properties.append(style); run.append(properties)
                value = OxmlElement("w:t"); value.text = link.group(1); run.append(value)
                element.append(run); paragraph._p.append(element)
            cursor = match.end()
        paragraph.add_run(text[cursor:])

    lines = normalize_markdown(args.input.read_text(encoding="utf-8")).splitlines()
    index, figure_inserted = 0, False
    counts["code_blocks"] = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("```"):
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code = document.add_paragraph()
                code.paragraph_format.line_spacing, code.paragraph_format.space_after = 1.0, Pt(0)
                run = code.add_run(lines[index])
                run.font.name, run.font.size = "Consolas", Pt(CODE_FONT_POINTS)
                index += 1
            counts["code_blocks"] += 1
            index += 1
            continue
        if not line or line.startswith("<!--"):
            index += 1; continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            picture = (args.input.parent / image_match.group(2)).resolve()
            document.add_picture(str(picture), width=Cm(FIGURE_WIDTH_CM))
            document.paragraphs[-1].paragraph_format.keep_with_next = True
            if image_match.group(1):
                document.add_paragraph(image_match.group(1), style="Caption")
            counts["figures"] += 1
        elif line.startswith("$$"):
            formula = line.removeprefix("$$").removesuffix("$$").strip()
            tag = re.search(r"\\tag\{([^}]+)\}", formula)
            if tag: formula = formula[:tag.start()] + formula[tag.end():]
            paragraph = document.add_paragraph(); paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_together = True
            add_math(paragraph, formula, display=True)
            if tag: paragraph.add_run("    (" + tag.group(1) + ")")
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells): rows.append(cells)
                index += 1
            table = document.add_table(rows=1, cols=len(rows[0])); table.style = "Table Grid"
            for offset, cells in enumerate(rows):
                target = table.rows[0].cells if offset == 0 else table.add_row().cells
                table.rows[offset]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
                if len(cells) != len(target): raise ValueError("Inconsistent Markdown table")
                for cell, text in zip(target, cells):
                    inline(cell.paragraphs[0], text)
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(TABLE_FONT_POINTS)
                        if offset == 0:
                            run.bold = True
            table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            counts["tables"] += 1
            continue
        elif line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            text = line[depth:].strip()
            paragraph = document.add_heading(level=0 if index == 0 else min(depth, MAXIMUM_HEADING_LEVEL))
            inline(paragraph, text)
        elif line.startswith("- "):
            inline(document.add_paragraph(style="List Bullet"), line[2:])
        elif NUMBERED_ITEM.fullmatch(line):
            inline(document.add_paragraph(style="List Number"), NUMBERED_ITEM.fullmatch(line).group(1))
        else:
            paragraph = document.add_paragraph()
            # A justified line that must break before a long unbreakable token (path, hash, identifier) is stretched
            # into wide gaps, so such paragraphs stay left-aligned.
            visible = re.sub(r"\[([^\]]+)\]\([^)]+\)", lambda link: link.group(1), line).replace("`", "")
            long_token = any(len(token) >= LONG_TOKEN_CHARACTERS for token in visible.split())
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if long_token else WD_ALIGN_PARAGRAPH.JUSTIFY
            inline(paragraph, line)
        index += 1
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page = OxmlElement("w:fldSimple"); page.set(qn("w:instr"), "PAGE"); footer._p.append(page)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)
    # Structural verification of actual native equation elements in the saved Word file.
    import zipfile
    with zipfile.ZipFile(args.output) as archive:
        xml = etree.fromstring(archive.read("word/document.xml"))
    equations = xml.xpath("count(//m:oMath)", namespaces={"m": "http://schemas.openxmlformats.org/officeDocument/2006/math"})
    if int(equations) != counts["display_equations"] + counts["inline_equations"]:
        raise ValueError("Saved Word equation count mismatch")
    report = {**counts, "native_omml_equations": int(equations), "source_markdown_sha256": sha256(args.input),
        "export_source_sha256": sha256(Path(__file__)), "math_xsl_sha256": sha256(args.math_xsl),
        "bibliography_sha256": sha256(args.input.parent / "references.bib") if (args.input.parent / "references.bib").exists() else None,
        "latex2mathml_version": "3.81.1", "output_sha256": sha256(args.output),
        "scope": "Editable review draft; not institution-approved formatting or completed research evidence."}
    args.output.with_suffix(".export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
