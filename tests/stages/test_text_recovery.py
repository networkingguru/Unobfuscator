import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from stages.text_recovery import classify_page_pixels


def test_classify_all_black_as_redacted():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    assert classify_page_pixels(pixels) == "redacted"

def test_classify_all_white_as_blank():
    pixels = np.full((100, 100), 240, dtype=np.uint8)
    assert classify_page_pixels(pixels) == "blank"

def test_classify_photo_as_photo():
    pixels = np.linspace(50, 200, 10000, dtype=np.uint8).reshape(100, 100)
    assert classify_page_pixels(pixels) == "photo"

def test_classify_text_page_as_text():
    pixels = np.full((100, 100), 240, dtype=np.uint8)
    pixels[10:12, 10:60] = 5
    pixels[20:22, 10:70] = 5
    pixels[30:32, 10:50] = 5
    assert classify_page_pixels(pixels) == "text"

def test_classify_mostly_redacted_with_some_white():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[0:5, :] = 240
    assert classify_page_pixels(pixels) == "redacted"


# ---------------------------------------------------------------------------
# Task 4: PDF Text Layer Extraction
# ---------------------------------------------------------------------------

import fitz

from stages.text_recovery import extract_text_from_pdf


def _make_text_pdf(text: str, num_pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        rect = page.rect + fitz.Rect(72, 72, -72, -72)
        page.insert_textbox(rect, text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def test_extract_text_from_text_pdf():
    pdf = _make_text_pdf("This is a test document with enough words. " * 10)
    text, source, tags = extract_text_from_pdf(pdf, min_words_per_page=10)
    assert len(text.split()) > 50
    assert source == "pdf_text_layer"
    assert tags is None

def test_extract_text_sparse_pdf_returns_none():
    pdf = _make_text_pdf("Hi")
    result = extract_text_from_pdf(pdf, min_words_per_page=50)
    assert result is None

def test_extract_text_empty_pdf():
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    result = extract_text_from_pdf(pdf_bytes, min_words_per_page=50)
    assert result is None

def test_extract_text_corrupt_pdf():
    result = extract_text_from_pdf(b"not a pdf", min_words_per_page=50)
    assert result is None
