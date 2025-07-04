import fitz
from pathlib import Path


def main():
    """Convert all PDFs in the raw_pdf directory to PNG images (72 dpi)."""
    src_dir = Path("raw_pdf")  # source directory with PDFs
    dst_dir = Path("pages")    # destination directory for PNGs
    dst_dir.mkdir(exist_ok=True)

    for pdf_path in src_dir.glob("*.pdf"):
        doc = fitz.open(pdf_path)
        for page_number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=72)
            img_name = f"{pdf_path.stem}_p{page_number:02d}.png"
            pix.save(dst_dir / img_name)
    print("Done")


if __name__ == "__main__":
    main()
