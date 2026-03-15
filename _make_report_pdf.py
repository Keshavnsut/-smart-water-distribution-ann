from pathlib import Path
from fpdf import FPDF

src = Path("Smart_Water_Distribution_Report_Draft.md")
out = Path("Smart_Water_Distribution_Report.pdf")

text = src.read_text(encoding="utf-8")
lines = text.splitlines()

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_margins(15, 15, 15)
pdf.add_page()
pdf.set_font("Times", size=12)


def safe_write(line: str, h: int = 6):
    if not line:
        pdf.ln(4)
        return

    # If a line has very long tokens (e.g., URLs), hard-wrap by character chunks.
    tokens = line.split(" ")
    rebuilt = []
    for token in tokens:
        if len(token) <= 80:
            rebuilt.append(token)
        else:
            chunks = [token[i:i+70] for i in range(0, len(token), 70)]
            rebuilt.extend(chunks)
    wrapped = " ".join(rebuilt)
    pdf.multi_cell(0, h, wrapped)


for raw in lines:
    line = raw.rstrip()

    if not line:
        pdf.ln(4)
        continue

    if line.startswith("# "):
        pdf.set_font("Times", "B", 18)
        safe_write(line[2:], h=10)
        pdf.ln(2)
        pdf.set_font("Times", size=12)
        continue

    if line.startswith("## "):
        pdf.set_font("Times", "B", 15)
        safe_write(line[3:], h=8)
        pdf.ln(1)
        pdf.set_font("Times", size=12)
        continue

    if line.startswith("### "):
        pdf.set_font("Times", "B", 13)
        safe_write(line[4:], h=7)
        pdf.set_font("Times", size=12)
        continue

    if line.lstrip().startswith("- "):
        bullet_text = "- " + line.lstrip()[2:]
        safe_write(bullet_text, h=6)
        continue

    safe_write(line, h=6)

pdf.output(str(out))
print(f"Generated: {out.resolve()}")
