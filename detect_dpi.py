#!/usr/bin/env python
"""detect_dpi.py

Print the effective DPI (dots per inch) for each page in every PDF inside
`raw_pdf/`, or (optionally) read the DPI metadata of PNG/JPG images inside
`pages/`.

Usage:
  python detect_dpi.py              # analyse PDFs in raw_pdf/
  python detect_dpi.py --images     # read metadata from images in pages/
"""
import argparse
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

RAW_PDF_DIR = Path("raw_pdf")
IMAGES_DIR = Path("pages")


def dpi_from_pdf(pdf_path: Path):
    """Yield (page_number, dpi_x, dpi_y) for each page in the given PDF."""
    doc = fitz.open(pdf_path)
    for page_number, page in enumerate(doc, start=1):
        # Render page at the default 1-to-1 matrix (1 px = 1 pt, i.e. 72 dpi).
        pix = page.get_pixmap()  # no dpi arg ⇒ default scale 1.0
        # Page rectangle dimensions are in points (1 pt = 1/72 inch)
        dpi_x = pix.width * 72 / page.rect.width
        dpi_y = pix.height * 72 / page.rect.height
        yield page_number, round(dpi_x), round(dpi_y)


def dpi_from_image(img_path: Path):
    """Return DPI tuple stored in PNG/JPEG metadata (may be None)."""
    with Image.open(img_path) as img:
        return img.info.get("dpi")


def main():
    parser = argparse.ArgumentParser(description="Detect DPI of PDFs or images.")
    parser.add_argument(
        "--images",
        action="store_true",
        help="Inspect PNG/JPG files in pages/ instead of PDFs in raw_pdf/",
    )
    args = parser.parse_args()

    if args.images:
        if not IMAGES_DIR.exists():
            print("pages/ directory not found – nothing to do")
            return
        for img_path in sorted(IMAGES_DIR.glob("*.*")):
            dpi = dpi_from_image(img_path)
            print(f"{img_path.name}: {dpi} (metadata)")
    else:
        if not RAW_PDF_DIR.exists():
            print("raw_pdf/ directory not found – nothing to do")
            return
        for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
            for page_no, dpi_x, dpi_y in dpi_from_pdf(pdf_path):
                print(f"{pdf_path.name} – page {page_no}: {dpi_x}×{dpi_y} dpi (calculated)")


if __name__ == "__main__":
    main() 