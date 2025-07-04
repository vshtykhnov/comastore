import fitz, pathlib

src = pathlib.Path("raw_pdf")          # где лежат PDF
dst = pathlib.Path("pages")            # куда класть PNG
dst.mkdir(exist_ok=True)

for pdf in src.glob("*.pdf"):
    doc = fitz.open(pdf)
    for i, page in enumerate(doc, 1):
        img = page.get_pixmap(dpi=300)
        img.save(dst / f"{pdf.stem}_p{i:02d}.png")
print("ГОТОВО")
