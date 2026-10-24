"""Build docs/System_Reference.pdf from docs/SYSTEM_REFERENCE.md.

Run from the TradingAgents folder (reportlab is not part of the app's environment):
    uv run --no-project --with reportlab python docs/build_reference_pdf.py
Handles the Markdown the reference uses: headings, paragraphs, bullet and numbered
lists, tables, code blocks, bold, italic and inline code.
"""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (ListFlowable, ListItem, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = Path(__file__).parent
INK, MUTED, LINE, SOFT, ACCENT = (colors.HexColor(c) for c in ("#0f172a", "#475569", "#e2e8f0", "#f1f5f9", "#2563eb"))
ss = getSampleStyleSheet()
H = {1: ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=INK,
                       spaceBefore=4, spaceAfter=8),
     2: ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=INK,
                       spaceBefore=12, spaceAfter=5),
     3: ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ACCENT,
                       spaceBefore=8, spaceAfter=3)}
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.8, textColor=INK,
                      spaceAfter=5)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.8, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.4, leading=9.2, backColor=SOFT,
                      borderPadding=(4, 5, 4, 5), spaceBefore=3, spaceAfter=7)


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", text)
    return text.replace("−", "-").replace("≤", "&lt;=").replace("≥", "&gt;=")


def table(rows: list[list[str]], width: float):
    n = len(rows[0])
    data = [[Paragraph(inline(c), CELLB if r == 0 else CELL) for c in row + [""] * (n - len(row))]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[width / n] * n, repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                           ("BACKGROUND", (0, 0), (-1, 0), SOFT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def render(md: str, width: float) -> list:
    story, lines, i = [], md.splitlines(), 0
    para: list[str] = []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), BODY))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(block), CODE))
        elif m := re.match(r"^(#{1,3}) (.*)", line):
            flush()
            level = len(m.group(1))
            if level == 2 and story:
                story.append(Spacer(1, 4))
            story.append(Paragraph(inline(m.group(2)), H[level]))
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += [table(rows, width), Spacer(1, 6)]
            continue
        elif re.match(r"^\s*([-*]|\d+\.) ", line):
            flush()
            items, numbered = [], bool(re.match(r"^\s*\d+\.", line))
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.) ", lines[i]):
                text = re.sub(r"^\s*([-*]|\d+\.) ", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and not re.match(r"^\s*([-*]|\d+\.) ", lines[i]):
                    text += " " + lines[i].strip()
                    i += 1
                items.append(ListItem(Paragraph(inline(text), BODY), leftIndent=12))
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet", leftIndent=12,
                                      bulletFontSize=8, start="1" if numbered else "•"))
            continue
        elif line.strip() == "---":
            flush()
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.7 * inch, 0.45 * inch, "Portfolio app: system reference")
    canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    out = HERE / "System_Reference.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.75 * inch, title="Portfolio app: system reference")
    story = render((HERE / "SYSTEM_REFERENCE.md").read_text(encoding="utf-8"), 7.1 * inch)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    main()
