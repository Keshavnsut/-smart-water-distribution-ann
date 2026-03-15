from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

src = Path("Smart_Water_Distribution_Report_Draft.md")
out = Path("Smart_Water_Distribution_Report.pdf")

text = src.read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()

styles = getSampleStyleSheet()
style_body = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Times-Roman",
    fontSize=11,
    leading=15,
    alignment=TA_JUSTIFY,
)
style_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=18, leading=22)
style_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=14, leading=18)
style_h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Times-Bold", fontSize=12, leading=16)

story = []

def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )

for raw in lines:
    line = raw.strip("\n")
    if not line.strip():
        story.append(Spacer(1, 8))
        continue

    if line.startswith("# "):
        story.append(Paragraph(esc(line[2:].strip()), style_h1))
        story.append(Spacer(1, 8))
        continue
    if line.startswith("## "):
        story.append(Paragraph(esc(line[3:].strip()), style_h2))
        story.append(Spacer(1, 6))
        continue
    if line.startswith("### "):
        story.append(Paragraph(esc(line[4:].strip()), style_h3))
        story.append(Spacer(1, 4))
        continue

    txt = line.strip()
    if txt.startswith("- "):
        txt = "• " + txt[2:]

    story.append(Paragraph(esc(txt), style_body))


doc = SimpleDocTemplate(
    str(out),
    pagesize=A4,
    leftMargin=50,
    rightMargin=50,
    topMargin=50,
    bottomMargin=50,
)
doc.build(story)
print(f"Generated: {out.resolve()}")
