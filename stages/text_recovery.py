"""Stage 5.5: Text Recovery — backfill Jmail text, extract PDF text layers, OCR."""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import fitz
import numpy as np

logger = logging.getLogger(__name__)


def classify_page_pixels(pixels: np.ndarray) -> str:
    """Classify a grayscale page image by pixel distribution."""
    white_pct = (pixels > 230).mean()
    black_pct = (pixels < 25).mean()
    mid_pct = 1.0 - white_pct - black_pct

    if black_pct > 0.90:
        return "redacted"
    if mid_pct > 0.40:
        return "photo"
    if white_pct > 0.98:
        return "blank"
    return "text"


def extract_text_from_pdf(pdf_bytes: bytes,
                          min_words_per_page: int = 50
                          ) -> Optional[tuple[str, str, Optional[str]]]:
    """Try to extract text from a PDF's embedded text layer.
    Returns (text, "pdf_text_layer", None) if sufficient, else None."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None

    if doc.page_count == 0:
        doc.close()
        return None

    pages_text = []
    total_words = 0
    for page in doc:
        text = page.get_text().strip()
        pages_text.append(text)
        total_words += len(text.split())
    doc.close()

    avg_words = total_words / len(pages_text) if pages_text else 0
    if avg_words >= min_words_per_page:
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
        return (full_text, "pdf_text_layer", None)
    return None
