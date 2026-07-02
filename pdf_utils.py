"""
pdf_utils.py
Converts PDF pages to PIL Images using PyMuPDF (fitz).
Pure Python — no external binaries required.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator
from PIL import Image
import fitz  # PyMuPDF
import io


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 200,
) -> Iterator[tuple[int, Image.Image]]:
    """
    Yield (page_number, PIL.Image) tuples for every page of a PDF.

    Args:
        pdf_path: Path to the PDF file.
        dpi:      Rendering resolution (higher = better quality but slower).

    Yields:
        (1-based page number, RGB PIL Image)
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    # Scale factor: fitz default is 72 dpi, so multiply by dpi/72
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_bytes = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            yield page_index + 1, pil_image
    finally:
        doc.close()


def count_pdf_pages(pdf_path: str | Path) -> int:
    """Return the total number of pages in a PDF without rendering them."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
