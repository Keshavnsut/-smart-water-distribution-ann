import re
from pathlib import Path

pdf_path = Path("Smart_Water_Distribution_Report_Style_Matched.pdf")
text = ""

try:
    import pypdf  # type: ignore

    reader = pypdf.PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
except Exception:
    try:
        import PyPDF2  # type: ignore

        reader = PyPDF2.PdfReader(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        print(f"PDF_READ_ERROR: {exc}")
        raise SystemExit(1)

urls = sorted(set(re.findall(r"https?://[^\s\)\]>\"]+", text)))
print(f"URL_COUNT: {len(urls)}")
for url in urls:
    print(url)
