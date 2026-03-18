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


# ---------------------------------------------------------------------------
# Task 5: OCR Engine
# ---------------------------------------------------------------------------

from PIL import Image
from stages.text_recovery import ocr_pdf, ocr_image, _is_tesseract_available


def test_ocr_pdf_skips_redacted_pages():
    doc = fitz.open()
    p0 = doc.new_page()
    p0.insert_text((72, 72), "Hello world " * 20, fontsize=12)
    p1 = doc.new_page()
    rect = fitz.Rect(0, 0, p1.rect.width, p1.rect.height)
    p1.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
    pdf_bytes = doc.tobytes()
    doc.close()

    text, source, tags_json = ocr_pdf(pdf_bytes)
    tags = json.loads(tags_json)
    assert tags["1"] == "redacted"
    assert source == "ocr"

def test_ocr_pdf_skips_photo_pages():
    doc = fitz.open()
    page = doc.new_page(width=100, height=100)
    gradient = np.linspace(50, 200, 100 * 100, dtype=np.uint8).reshape(100, 100)
    rgb = np.stack([gradient, gradient, gradient], axis=2)
    img_pil = Image.fromarray(rgb, "RGB")
    import io
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    page.insert_image(page.rect, stream=buf.getvalue())
    pdf_bytes = doc.tobytes()
    doc.close()

    text, source, tags_json = ocr_pdf(pdf_bytes)
    tags = json.loads(tags_json)
    assert tags["0"] == "photo"

def test_ocr_image_skips_all_black():
    img = Image.new("L", (100, 100), color=0)
    text, tag = ocr_image(img)
    assert tag == "redacted"
    assert text == ""

def test_is_tesseract_available_returns_bool():
    result = _is_tesseract_available()
    assert isinstance(result, bool)

def test_ocr_image_returns_text_for_text_image():
    if not _is_tesseract_available():
        pytest.skip("Tesseract not installed")
    img = Image.new("L", (400, 100), color=240)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Hello World Test Document", fill=0)
    text, tag = ocr_image(img)
    assert tag == "text"
