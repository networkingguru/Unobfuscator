# Text Recovery Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover missing document text for ~95k docs via Jmail shard fix, PDF text layer extraction, and Tesseract OCR.

**Architecture:** A new Phase 5.5 stage (`stages/text_recovery.py`) runs between PDF processing and output generation. It first backfills text from the correct Jmail parquet shard, then extracts text from PDFs/images via PyMuPDF and Tesseract. Recovered text is fingerprinted inline and stored, so it participates in unredaction matching on the next daemon cycle.

**Tech Stack:** PyMuPDF (fitz), pytesseract, Pillow, numpy, DuckDB, SQLite

**Spec:** `docs/superpowers/specs/2026-03-18-text-recovery-design.md`

---

### Task 1: Database Schema Migration

**Files:**
- Modify: `core/db.py:1-98` (SCHEMA string and init_db)
- Test: `tests/core/test_db.py`

- [ ] **Step 1: Write failing tests for new columns and helpers**

In `tests/core/test_db.py`, add:

```python
from core.db import (
    get_docs_needing_text_recovery, get_docs_needing_backfill,
    update_extracted_text, mark_ocr_processed
)


def test_new_columns_exist_after_init(conn):
    """text_source, ocr_processed, page_tags columns must exist."""
    row = conn.execute(
        "SELECT text_source, ocr_processed, page_tags FROM documents LIMIT 0"
    ).fetchone()
    # No error means columns exist


def test_update_extracted_text_stores_text_and_source(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    update_extracted_text(conn, SAMPLE_DOC["id"], "Recovered text", "ocr")
    conn.commit()
    row = conn.execute(
        "SELECT extracted_text, text_source FROM documents WHERE id = ?",
        (SAMPLE_DOC["id"],)
    ).fetchone()
    assert row["extracted_text"] == "Recovered text"
    assert row["text_source"] == "ocr"


def test_update_extracted_text_stores_page_tags(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    tags = '{"0": "text", "1": "photo"}'
    update_extracted_text(conn, SAMPLE_DOC["id"], "text", "ocr", page_tags=tags)
    conn.commit()
    row = conn.execute(
        "SELECT page_tags FROM documents WHERE id = ?", (SAMPLE_DOC["id"],)
    ).fetchone()
    assert row["page_tags"] == tags


def test_mark_ocr_processed(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    mark_ocr_processed(conn, SAMPLE_DOC["id"])
    conn.commit()
    row = conn.execute(
        "SELECT ocr_processed FROM documents WHERE id = ?", (SAMPLE_DOC["id"],)
    ).fetchone()
    assert row["ocr_processed"] == 1


def test_get_docs_needing_text_recovery(conn):
    doc_no_text = {**SAMPLE_DOC, "id": "no_text", "extracted_text": ""}
    doc_has_text = {**SAMPLE_DOC, "id": "has_text", "extracted_text": "hello world"}
    upsert_document(conn, doc_no_text)
    upsert_document(conn, doc_has_text)
    conn.commit()
    docs = get_docs_needing_text_recovery(conn)
    ids = [d["id"] for d in docs]
    assert "no_text" in ids
    assert "has_text" not in ids


def test_get_docs_needing_text_recovery_excludes_ocr_processed(conn):
    doc = {**SAMPLE_DOC, "id": "processed", "extracted_text": ""}
    upsert_document(conn, doc)
    mark_ocr_processed(conn, "processed")
    conn.commit()
    docs = get_docs_needing_text_recovery(conn)
    ids = [d["id"] for d in docs]
    assert "processed" not in ids


def test_get_docs_needing_backfill(conn):
    doc = {**SAMPLE_DOC, "id": "backfill_me", "extracted_text": "",
           "release_batch": "VOL00001"}
    upsert_document(conn, doc)
    mark_text_processed(conn, "backfill_me")
    conn.commit()
    docs = get_docs_needing_backfill(conn, known_batches={"VOL00001"})
    ids = [d["id"] for d in docs]
    assert "backfill_me" in ids


def test_get_docs_needing_backfill_excludes_unknown_batch(conn):
    doc = {**SAMPLE_DOC, "id": "unknown_batch", "extracted_text": "",
           "release_batch": "MYSTERY"}
    upsert_document(conn, doc)
    mark_text_processed(conn, "unknown_batch")
    conn.commit()
    docs = get_docs_needing_backfill(conn, known_batches={"VOL00001"})
    ids = [d["id"] for d in docs]
    assert "unknown_batch" not in ids


def test_migration_sets_text_source_for_existing_docs(conn):
    """Existing docs with text should get text_source='jmail' after migration."""
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    # Simulate migration
    from core.db import _migrate_text_recovery_columns
    _migrate_text_recovery_columns(conn)
    conn.commit()
    row = conn.execute(
        "SELECT text_source FROM documents WHERE id = ?", (SAMPLE_DOC["id"],)
    ).fetchone()
    assert row["text_source"] == "jmail"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/core/test_db.py -v -k "new_columns or update_extracted or mark_ocr or needing_text or needing_backfill or migration_sets" 2>&1 | tail -20`
Expected: Multiple FAIL / ImportError

- [ ] **Step 3: Add new columns to SCHEMA and implement helpers**

In `core/db.py`, update the SCHEMA string — add three columns to the `documents` table definition:

```python
# Add after the existing pdf_processed column in SCHEMA:
    text_source TEXT,
    ocr_processed BOOLEAN DEFAULT 0,
    page_tags TEXT
```

Add the migration function and new helpers after the existing `get_all_recovery_groups` function:

```python
def _migrate_text_recovery_columns(conn) -> None:
    """Add text_source, ocr_processed, page_tags columns if missing.

    Also backfill text_source='jmail' for existing docs that already have text.
    """
    for col, typedef in [
        ("text_source", "TEXT"),
        ("ocr_processed", "BOOLEAN DEFAULT 0"),
        ("page_tags", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # Column already exists
    # Backfill text_source for existing docs with text
    conn.execute("""
        UPDATE documents SET text_source = 'jmail'
        WHERE extracted_text IS NOT NULL AND extracted_text != ''
          AND text_source IS NULL
    """)


def update_extracted_text(conn, doc_id: str, text: str, text_source: str,
                          page_tags: str = None) -> None:
    """Store recovered text and its provenance."""
    conn.execute("""
        UPDATE documents
        SET extracted_text = ?, text_source = ?, page_tags = ?
        WHERE id = ?
    """, (text, text_source, page_tags, doc_id))


def mark_ocr_processed(conn, doc_id: str) -> None:
    """Mark a document as having been through OCR/text-recovery."""
    conn.execute("UPDATE documents SET ocr_processed = 1 WHERE id = ?", (doc_id,))


def get_docs_needing_text_recovery(conn, limit: int = 100) -> list[dict]:
    """Return docs where OCR has not been attempted and text is empty."""
    rows = conn.execute("""
        SELECT id, source, release_batch, original_filename, pdf_url
        FROM documents
        WHERE ocr_processed = 0
          AND (extracted_text IS NULL OR extracted_text = '')
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_docs_needing_backfill(conn, known_batches: set,
                              limit: int = 1000) -> list[dict]:
    """Return docs that were processed but got empty text from a known shard."""
    if not known_batches:
        return []
    placeholders = ",".join("?" * len(known_batches))
    rows = conn.execute(f"""
        SELECT id, release_batch
        FROM documents
        WHERE text_processed = 1
          AND (extracted_text IS NULL OR extracted_text = '')
          AND text_source IS NULL
          AND release_batch IN ({placeholders})
        LIMIT ?
    """, (*known_batches, limit)).fetchall()
    return [dict(r) for r in rows]
```

Update `init_db` to call migration after schema creation:

```python
def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate_text_recovery_columns(conn)
        conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/core/test_db.py -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/db.py tests/core/test_db.py
git commit -m "feat: add text_source, ocr_processed, page_tags columns with migration and helpers"
```

---

### Task 2: Shard-Aware Jmail API Fix

**Files:**
- Modify: `core/api.py:11-98`
- Test: `tests/core/test_api.py`

- [ ] **Step 1: Write failing tests for shard resolution**

In `tests/core/test_api.py`, add:

```python
from core.api import resolve_shard, SHARD_MAP, DEFAULT_SHARD


def test_resolve_shard_known_batch():
    assert resolve_shard("VOL00008-2") == "VOL00008"
    assert resolve_shard("VOL00008-OFFICIAL-DOJ-LATEST") == "VOL00008"
    assert resolve_shard("VOL00009") == "VOL00009"
    assert resolve_shard("DataSet11") == "DataSet11"


def test_resolve_shard_unknown_batch_returns_other():
    assert resolve_shard("VOL00001") == "other"
    assert resolve_shard("DOJ-COURT") == "other"
    assert resolve_shard("batch-3") == "other"


@patch("core.api.duckdb.connect")
def test_fetch_documents_text_batch_uses_resolved_shard(mock_connect):
    """URL must use the resolved shard, not the raw batch_id."""
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        columns=["id", "extracted_text"]
    )
    fetch_documents_text_batch(["doc1.pdf"], "VOL00001")
    query = mock_conn.execute.call_args_list[-1][0][0]
    # Should hit other.parquet, not VOL00001.parquet
    assert "other.parquet" in query
    assert "VOL00001.parquet" not in query


@patch("core.api.duckdb.connect")
def test_fetch_documents_text_batch_vol00008_2_uses_vol00008_shard(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        columns=["id", "extracted_text"]
    )
    fetch_documents_text_batch(["doc1.pdf"], "VOL00008-2")
    query = mock_conn.execute.call_args_list[-1][0][0]
    assert "VOL00008.parquet" in query
    assert "VOL00008-2.parquet" not in query


@patch("core.api.duckdb.connect")
def test_fetch_document_text_uses_resolved_shard(mock_connect):
    """Singular fetch_document_text must also use shard resolution."""
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        columns=["id", "extracted_text"]
    )
    fetch_document_text(doc_id="doc1.pdf", batch_id="DOJ-COURT")
    query = mock_conn.execute.call_args_list[-1][0][0]
    assert "other.parquet" in query
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/core/test_api.py -v -k "resolve_shard or uses_resolved" 2>&1 | tail -20`
Expected: ImportError for `resolve_shard`

- [ ] **Step 3: Implement shard mapping and update API functions**

In `core/api.py`, add after the URL constants (line 15):

```python
# Jmail shards text across 5 files; most batches live in "other.parquet".
SHARD_MAP = {
    "VOL00008": "VOL00008",
    "VOL00008-2": "VOL00008",
    "VOL00008-OFFICIAL-DOJ-LATEST": "VOL00008",
    "VOL00009": "VOL00009",
    "VOL00010": "VOL00010",
    "DataSet11": "DataSet11",
}
DEFAULT_SHARD = "other"


def resolve_shard(batch_id: str) -> str:
    """Map a release batch to its Jmail documents-full shard filename."""
    return SHARD_MAP.get(batch_id, DEFAULT_SHARD)
```

Update `fetch_document_text` (line 51-63) — replace the hardcoded URL:

```python
def fetch_document_text(doc_id: str, batch_id: str) -> Optional[str]:
    """Return extracted text for a single document ID within a batch."""
    shard = resolve_shard(batch_id)
    url = f"https://data.jmail.world/v1/documents-full/{shard}.parquet"
    with duckdb.connect() as conn:
        conn.execute("SET force_download=true")
        df = conn.execute(f"""
            SELECT id, extracted_text
            FROM read_parquet('{url}')
            WHERE id = $1
        """, [doc_id]).fetchdf()
    if df.empty:
        return None
    return df.iloc[0]["extracted_text"]
```

Update `fetch_documents_text_batch` (line 83) — replace the hardcoded URL:

```python
    shard = resolve_shard(batch_id)
    url = f"https://data.jmail.world/v1/documents-full/{shard}.parquet"
```

Replace line 83 (`url = f"https://data.jmail.world/v1/documents-full/{batch_id}.parquet"`) with the two lines above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/core/test_api.py -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/api.py tests/core/test_api.py
git commit -m "fix: resolve Jmail shard before querying documents-full parquet"
```

---

### Task 3: Page Classification (Pixel Histogram Pre-Filter)

**Files:**
- Create: `stages/text_recovery.py`
- Test: `tests/stages/test_text_recovery.py`

- [ ] **Step 1: Write failing tests for page classification**

Create `tests/stages/test_text_recovery.py`:

```python
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from stages.text_recovery import classify_page_pixels


def test_classify_all_black_as_redacted():
    """A page that is >90% black pixels should be tagged 'redacted'."""
    pixels = np.zeros((100, 100), dtype=np.uint8)  # all black
    assert classify_page_pixels(pixels) == "redacted"


def test_classify_all_white_as_blank():
    """A page that is >98% white should be tagged 'blank'."""
    pixels = np.full((100, 100), 240, dtype=np.uint8)  # all white
    assert classify_page_pixels(pixels) == "blank"


def test_classify_photo_as_photo():
    """A page with >40% mid-tone pixels should be tagged 'photo'."""
    # Gradient from 50 to 200 — lots of mid-tones
    pixels = np.linspace(50, 200, 10000, dtype=np.uint8).reshape(100, 100)
    assert classify_page_pixels(pixels) == "photo"


def test_classify_text_page_as_text():
    """A page with mostly white background and some dark text pixels."""
    pixels = np.full((100, 100), 240, dtype=np.uint8)  # white background
    # Add some "text" — dark pixels scattered in rows
    pixels[10:12, 10:60] = 5   # dark text line
    pixels[20:22, 10:70] = 5   # another text line
    pixels[30:32, 10:50] = 5
    assert classify_page_pixels(pixels) == "text"


def test_classify_mostly_redacted_with_some_white():
    """95% black with 5% white — still redacted."""
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[0:5, :] = 240  # 5% white strip at top
    assert classify_page_pixels(pixels) == "redacted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v -k "classify" 2>&1 | tail -15`
Expected: ImportError

- [ ] **Step 3: Create `stages/text_recovery.py` with classify_page_pixels**

Create `stages/text_recovery.py`:

```python
"""Stage 5.5: Text Recovery — backfill Jmail text, extract PDF text layers, OCR.

Runs between PDF processing (Phase 5) and output generation (Phase 6).
Recovers text for documents that have no extracted_text, then builds
fingerprints inline so recovered text participates in unredaction matching.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def classify_page_pixels(pixels: np.ndarray) -> str:
    """Classify a grayscale page image by pixel distribution.

    Args:
        pixels: 2D numpy array of uint8 grayscale pixel values.

    Returns:
        One of: "redacted", "photo", "blank", "text"
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v -k "classify" 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add stages/text_recovery.py tests/stages/test_text_recovery.py
git commit -m "feat: add page pixel classification for OCR pre-filter"
```

---

### Task 4: PDF Text Layer Extraction

**Files:**
- Modify: `stages/text_recovery.py`
- Test: `tests/stages/test_text_recovery.py`

- [ ] **Step 1: Write failing tests for text layer extraction**

Add to `tests/stages/test_text_recovery.py`:

```python
import fitz  # PyMuPDF

from stages.text_recovery import extract_text_from_pdf


def _make_text_pdf(text: str, num_pages: int = 1) -> bytes:
    """Create an in-memory PDF with text on each page."""
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_text_from_text_pdf():
    """A PDF with embedded text should return the text and 'pdf_text_layer' source."""
    pdf = _make_text_pdf("This is a test document with enough words to pass the "
                         "threshold. " * 10)
    text, source, tags = extract_text_from_pdf(pdf, min_words_per_page=10)
    assert len(text.split()) > 50
    assert source == "pdf_text_layer"
    assert tags is None  # No OCR needed, no page classification


def test_extract_text_sparse_pdf_returns_none():
    """A PDF with very few words per page should return None (needs OCR)."""
    pdf = _make_text_pdf("Hi")
    result = extract_text_from_pdf(pdf, min_words_per_page=50)
    assert result is None


def test_extract_text_empty_pdf():
    """A PDF with no text should return None."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    result = extract_text_from_pdf(pdf_bytes, min_words_per_page=50)
    assert result is None


def test_extract_text_corrupt_pdf():
    """Corrupt PDF bytes should return None, not raise."""
    result = extract_text_from_pdf(b"not a pdf", min_words_per_page=50)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v -k "extract_text" 2>&1 | tail -15`
Expected: ImportError

- [ ] **Step 3: Implement extract_text_from_pdf**

Add to `stages/text_recovery.py`:

```python
import fitz


def extract_text_from_pdf(pdf_bytes: bytes,
                          min_words_per_page: int = 50
                          ) -> Optional[tuple[str, str, Optional[str]]]:
    """Try to extract text from a PDF's embedded text layer.

    Returns (text, "pdf_text_layer", None) if text layer is sufficient,
    or None if the text layer is too sparse and OCR is needed.
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v -k "extract_text" 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add stages/text_recovery.py tests/stages/test_text_recovery.py
git commit -m "feat: add PDF text layer extraction with word-count threshold"
```

---

### Task 5: OCR Engine with Pre-Filter

**Files:**
- Modify: `stages/text_recovery.py`
- Test: `tests/stages/test_text_recovery.py`

- [ ] **Step 1: Write failing tests for OCR**

Add to `tests/stages/test_text_recovery.py`:

```python
from PIL import Image
from stages.text_recovery import ocr_pdf, ocr_image, _is_tesseract_available


def test_ocr_pdf_skips_redacted_pages():
    """Pages that are all-black should be tagged 'redacted' and not OCR'd."""
    # Create a 2-page PDF: one text page, one all-black page
    doc = fitz.open()
    # Page 0: text page
    p0 = doc.new_page()
    p0.insert_text((72, 72), "Hello world " * 20, fontsize=12)
    # Page 1: all black
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
    """Pages with high mid-tone content should be tagged 'photo'."""
    doc = fitz.open()
    page = doc.new_page(width=100, height=100)
    # Insert a gradient-like image (simulates a photo)
    gradient = np.linspace(50, 200, 100 * 100, dtype=np.uint8).reshape(100, 100)
    # Convert to RGB for PDF insertion
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


def test_ocr_image_returns_text_for_text_image():
    """A simple white image with text rendered on it should return OCR text."""
    if not _is_tesseract_available():
        pytest.skip("Tesseract not installed")
    # Create a white image with text drawn on it
    img = Image.new("L", (400, 100), color=240)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Hello World Test Document", fill=0)
    text, tag = ocr_image(img)
    # Tesseract should find something; exact output varies
    assert tag == "text"


def test_ocr_image_skips_all_black():
    """An all-black image should be tagged 'redacted' with empty text."""
    img = Image.new("L", (100, 100), color=0)
    text, tag = ocr_image(img)
    assert tag == "redacted"
    assert text == ""


def test_is_tesseract_available_returns_bool():
    result = _is_tesseract_available()
    assert isinstance(result, bool)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v -k "ocr" 2>&1 | tail -15`
Expected: ImportError

- [ ] **Step 3: Implement OCR functions**

Add to `stages/text_recovery.py`:

```python
import shutil
from PIL import Image


def _is_tesseract_available() -> bool:
    """Check if the tesseract binary is installed."""
    return shutil.which("tesseract") is not None


def ocr_image(img: Image.Image) -> tuple[str, str]:
    """OCR a single PIL Image after pixel classification.

    Returns (text, tag) where tag is one of: text, redacted, photo, blank.
    """
    gray = img.convert("L") if img.mode != "L" else img
    pixels = np.array(gray)
    tag = classify_page_pixels(pixels)

    if tag != "text":
        return ("", tag)

    import pytesseract
    text = pytesseract.image_to_string(gray).strip()
    return (text, tag)


def ocr_pdf(pdf_bytes: bytes) -> tuple[str, str, str]:
    """OCR a PDF, classifying and OCR'ing each page individually.

    Returns (full_text, "ocr", page_tags_json).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts = []
    page_tags = {}

    for page_num, page in enumerate(doc):
        pixmap = page.get_pixmap(colorspace=fitz.csGRAY)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.h, pixmap.w
        )
        tag = classify_page_pixels(pixels)
        page_tags[str(page_num)] = tag

        if tag == "text":
            img = Image.frombytes("L", (pixmap.w, pixmap.h), pixmap.samples)
            import pytesseract
            text = pytesseract.image_to_string(img).strip()
            page_texts.append(text)
        else:
            page_texts.append("")

    doc.close()
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
    tags_json = json.dumps(page_tags)
    return (full_text, "ocr", tags_json)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v 2>&1 | tail -20`
Expected: All PASS (OCR text tests may skip if Tesseract not installed)

- [ ] **Step 5: Commit**

```bash
git add stages/text_recovery.py tests/stages/test_text_recovery.py
git commit -m "feat: add OCR engine with pixel histogram pre-filter"
```

---

### Task 6: Text Recovery Orchestrator (Phase 5.5)

**Files:**
- Modify: `stages/text_recovery.py`
- Test: `tests/stages/test_text_recovery.py`

- [ ] **Step 1: Write failing tests for the orchestrator**

Add to `tests/stages/test_text_recovery.py`:

```python
from core.db import (
    init_db, get_connection, upsert_document, mark_text_processed,
    upsert_fingerprint, get_all_fingerprints, update_extracted_text,
    mark_ocr_processed
)
from stages.text_recovery import run_text_recovery


@pytest.fixture
def db_conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def test_run_text_recovery_backfills_from_jmail(db_conn):
    """Docs with empty text in a known shard should get text backfilled."""
    doc = {
        "id": "EFTA00000001.pdf", "source": "doj",
        "release_batch": "VOL00001", "original_filename": "EFTA00000001.pdf",
        "page_count": 1, "size_bytes": 1000,
        "description": "Test", "extracted_text": "",
    }
    upsert_document(db_conn, doc)
    mark_text_processed(db_conn, doc["id"])
    db_conn.commit()

    with patch("stages.text_recovery.fetch_documents_text_batch") as mock_fetch:
        mock_fetch.return_value = {"EFTA00000001.pdf": "Backfilled text content here"}
        count = run_text_recovery(db_conn, redaction_markers=["[REDACTED]"])

    assert count > 0
    row = db_conn.execute(
        "SELECT extracted_text, text_source FROM documents WHERE id = ?",
        (doc["id"],)
    ).fetchone()
    assert row["extracted_text"] == "Backfilled text content here"
    assert row["text_source"] == "jmail"


def test_run_text_recovery_builds_fingerprint_after_backfill(db_conn):
    """Backfilled docs should have a fingerprint built inline."""
    doc = {
        "id": "EFTA00000001.pdf", "source": "doj",
        "release_batch": "VOL00001", "original_filename": "EFTA00000001.pdf",
        "page_count": 1, "size_bytes": 1000,
        "description": "Test",
        "extracted_text": "",
    }
    upsert_document(db_conn, doc)
    mark_text_processed(db_conn, doc["id"])
    db_conn.commit()

    long_text = "word " * 100  # Enough for shingles
    with patch("stages.text_recovery.fetch_documents_text_batch") as mock_fetch:
        mock_fetch.return_value = {"EFTA00000001.pdf": long_text}
        run_text_recovery(db_conn, redaction_markers=["[REDACTED]"])

    fps = get_all_fingerprints(db_conn)
    assert any(f["doc_id"] == "EFTA00000001.pdf" for f in fps)


def test_run_text_recovery_skips_already_backfilled(db_conn):
    """Docs that already have text_source set should not be re-backfilled."""
    doc = {
        "id": "already_done", "source": "doj",
        "release_batch": "VOL00001", "original_filename": "done.pdf",
        "page_count": 1, "size_bytes": 1000,
        "description": "Test", "extracted_text": "existing text",
    }
    upsert_document(db_conn, doc)
    mark_text_processed(db_conn, doc["id"])
    db_conn.commit()

    with patch("stages.text_recovery.fetch_documents_text_batch") as mock_fetch:
        mock_fetch.return_value = {}
        run_text_recovery(db_conn, redaction_markers=["[REDACTED]"])

    # Should not have been queried
    mock_fetch.assert_not_called()


def test_run_text_recovery_ocrs_pdf_with_no_text(db_conn, tmp_path):
    """A doc with a cached PDF but no text should be OCR'd."""
    cache_dir = tmp_path / "pdf_cache" / "VOL00001"
    cache_dir.mkdir(parents=True)

    # Create a PDF with text rendered as an image (simulating a scan)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Scanned document text " * 20, fontsize=12)
    pdf_path = cache_dir / "scan.pdf"
    doc.save(str(pdf_path))
    doc.close()

    db_doc = {
        "id": "scan.pdf", "source": "doj",
        "release_batch": "VOL00001", "original_filename": "scan.pdf",
        "page_count": 1, "size_bytes": 1000,
        "description": "Test", "extracted_text": "",
    }
    upsert_document(db_conn, db_doc)
    mark_text_processed(db_conn, db_doc["id"])
    db_conn.commit()

    with patch("stages.text_recovery.fetch_documents_text_batch") as mock_fetch:
        mock_fetch.return_value = {}  # Not in Jmail
        with patch("stages.text_recovery._CACHE_DIR", tmp_path / "pdf_cache"):
            run_text_recovery(db_conn, redaction_markers=["[REDACTED]"])

    row = db_conn.execute(
        "SELECT extracted_text, text_source, ocr_processed FROM documents WHERE id = ?",
        ("scan.pdf",)
    ).fetchone()
    # Should have extracted text from the text layer
    assert row["extracted_text"] != ""
    assert row["ocr_processed"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v -k "run_text_recovery" 2>&1 | tail -15`
Expected: ImportError

- [ ] **Step 3: Implement `run_text_recovery` orchestrator**

Add to `stages/text_recovery.py`:

```python
from core.api import fetch_documents_text_batch, resolve_shard, SHARD_MAP, DEFAULT_SHARD
from core.db import (
    get_docs_needing_backfill, get_docs_needing_text_recovery,
    update_extracted_text, mark_ocr_processed, upsert_fingerprint
)
from stages.indexer import clean_text, shingle, build_fingerprint

_CACHE_DIR = Path(__file__).resolve().parent.parent / "pdf_cache"


def _build_and_store_fingerprint(conn, doc_id: str, text: str,
                                  redaction_markers: list[str]) -> None:
    """Build a MinHash fingerprint from text and store it."""
    cleaned = clean_text(text, redaction_markers)
    shingles = shingle(cleaned)
    if not shingles:
        return
    sig = build_fingerprint(cleaned)
    upsert_fingerprint(conn, doc_id, sig, len(shingles))


def _resolve_pdf_path(doc: dict) -> Optional[Path]:
    """Find the local PDF/image file for a document."""
    batch = doc.get("release_batch", "")
    filename = doc.get("original_filename", "")
    if batch and filename:
        local = _CACHE_DIR / batch / filename
        if local.is_file():
            return local
    return None


def _download_pdf_to_cache(doc: dict) -> Optional[Path]:
    """Download a PDF from its URL to the local cache. Returns path or None."""
    url = doc.get("pdf_url")
    if not url:
        return None
    batch = doc.get("release_batch", "")
    filename = doc.get("original_filename", "")
    if not batch or not filename:
        return None

    import httpx
    cache_path = _CACHE_DIR / batch / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cookies = {}
        if "justice.gov" in url:
            cookies["justiceGovAgeVerified"] = "true"
        resp = httpx.get(url, timeout=30, follow_redirects=True, cookies=cookies)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return cache_path
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return None


def _process_single_doc(conn, doc: dict, redaction_markers: list[str],
                        min_words_per_page: int) -> bool:
    """Attempt text extraction for a single document. Returns True if text recovered."""
    doc_id = doc["id"]
    filename = doc.get("original_filename", "")

    # Find the file
    file_path = _resolve_pdf_path(doc)
    if file_path is None:
        file_path = _download_pdf_to_cache(doc)
    if file_path is None:
        mark_ocr_processed(conn, doc_id)
        return False

    is_image = filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))

    if is_image:
        if not _is_tesseract_available():
            mark_ocr_processed(conn, doc_id)
            return False
        try:
            img = Image.open(file_path)
        except Exception as e:
            logger.warning("Failed to open image %s: %s", file_path, e)
            mark_ocr_processed(conn, doc_id)
            return False
        text, tag = ocr_image(img)
        tags_json = json.dumps({"0": tag})
        if text.strip():
            update_extracted_text(conn, doc_id, text, "ocr", page_tags=tags_json)
            _build_and_store_fingerprint(conn, doc_id, text, redaction_markers)
        else:
            # Store tags even if no text recovered (for future use)
            conn.execute(
                "UPDATE documents SET page_tags = ? WHERE id = ?",
                (tags_json, doc_id)
            )
        mark_ocr_processed(conn, doc_id)
        return bool(text.strip())

    # PDF file
    try:
        pdf_bytes = file_path.read_bytes()
    except Exception as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        mark_ocr_processed(conn, doc_id)
        return False

    # Step 1: Try text layer
    result = extract_text_from_pdf(pdf_bytes, min_words_per_page=min_words_per_page)
    if result is not None:
        text, source, tags = result
        update_extracted_text(conn, doc_id, text, source, page_tags=tags)
        _build_and_store_fingerprint(conn, doc_id, text, redaction_markers)
        mark_ocr_processed(conn, doc_id)
        return True

    # Step 2: OCR
    if not _is_tesseract_available():
        logger.warning("Tesseract not installed — skipping OCR for %s", doc_id)
        mark_ocr_processed(conn, doc_id)
        return False

    try:
        text, source, tags_json = ocr_pdf(pdf_bytes)
    except Exception as e:
        logger.warning("OCR failed for %s: %s", doc_id, e)
        mark_ocr_processed(conn, doc_id)
        return False

    if text.strip():
        update_extracted_text(conn, doc_id, text, source, page_tags=tags_json)
        _build_and_store_fingerprint(conn, doc_id, text, redaction_markers)
    else:
        conn.execute(
            "UPDATE documents SET page_tags = ? WHERE id = ?",
            (tags_json, doc_id)
        )
    mark_ocr_processed(conn, doc_id)
    return bool(text.strip())


def run_text_recovery(conn, redaction_markers: list[str],
                      min_words_per_page: int = 50) -> int:
    """Run the full text recovery pipeline. Returns count of docs recovered."""
    recovered = 0

    # --- Step 1: Jmail shard backfill ---
    # Every batch resolves to a shard (known or "other"), so collect all
    # distinct batch names from the DB as candidates for backfill.
    db_batches = conn.execute(
        "SELECT DISTINCT release_batch FROM documents "
        "WHERE release_batch IS NOT NULL"
    ).fetchall()
    backfill_batches = {row[0] for row in db_batches}

    docs = get_docs_needing_backfill(conn, known_batches=backfill_batches)
    if docs:
        # Group by shard for efficient fetching
        by_shard: dict[str, list[dict]] = {}
        for doc in docs:
            shard = resolve_shard(doc["release_batch"])
            by_shard.setdefault(shard, []).append(doc)

        for shard, shard_docs in by_shard.items():
            doc_ids = [d["id"] for d in shard_docs]
            batch_id = shard_docs[0]["release_batch"]  # Any batch for this shard
            try:
                text_map = fetch_documents_text_batch(doc_ids, batch_id)
            except Exception as e:
                logger.warning("Failed to fetch text for shard %s: %s", shard, e)
                continue

            for doc in shard_docs:
                text = text_map.get(doc["id"])
                if text and text.strip():
                    update_extracted_text(conn, doc["id"], text, "jmail")
                    _build_and_store_fingerprint(
                        conn, doc["id"], text, redaction_markers
                    )
                    recovered += 1
            conn.commit()

    # --- Step 2 & 3: Text layer extraction + OCR ---
    while True:
        batch_docs = get_docs_needing_text_recovery(conn, limit=100)
        if not batch_docs:
            break
        for doc in batch_docs:
            if _process_single_doc(conn, doc, redaction_markers, min_words_per_page):
                recovered += 1
            conn.commit()

    return recovered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/stages/test_text_recovery.py -v 2>&1 | tail -25`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add stages/text_recovery.py tests/stages/test_text_recovery.py
git commit -m "feat: add text recovery orchestrator with Jmail backfill and OCR"
```

---

### Task 7: Wire Phase 5.5 into Daemon Loop

**Files:**
- Modify: `unobfuscator.py:1-50,73-177,293-316,362-411`
- Test: `tests/test_daemon.py`

- [ ] **Step 1: Write failing test for Phase 5.5 in daemon cycle**

Add to `tests/test_daemon.py` (or create if minimal):

```python
import pytest
from unittest.mock import patch, MagicMock, call
from core.db import init_db, get_connection


def test_run_one_cycle_calls_text_recovery(tmp_path):
    """Phase 5.5 must be called between PDF processing and output generation."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = get_connection(db_path)
    cfg = {
        "redaction_markers": ["[REDACTED]"],
        "matching": {"min_overlap_chars": 200, "similarity_threshold": 0.70,
                     "email_header_min_matches": 2},
        "output_dir": str(tmp_path / "output"),
        "ocr": {"min_words_per_page": 50},
    }

    with patch("unobfuscator.run_indexer_batch"), \
         patch("unobfuscator.run_phase0_email_fastpath", return_value=[]), \
         patch("unobfuscator.run_phase2_lsh_candidates", return_value=[]), \
         patch("unobfuscator.run_phase3_verify_and_group"), \
         patch("unobfuscator.run_merger", return_value=0), \
         patch("unobfuscator.dequeue", return_value=None), \
         patch("unobfuscator.process_pdf_for_document"), \
         patch("unobfuscator.run_text_recovery") as mock_tr, \
         patch("unobfuscator.run_output_generator", return_value=0), \
         patch("unobfuscator.get_pending_pdf_documents", return_value=[]):
        from unobfuscator import _run_one_cycle
        _run_one_cycle(conn, cfg)
        mock_tr.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/test_daemon.py -v -k "text_recovery" 2>&1 | tail -15`
Expected: FAIL (import error or assert)

- [ ] **Step 3: Integrate Phase 5.5 into `_run_one_cycle`**

In `unobfuscator.py`, add the import at the top (after the other stage imports, around line 33):

```python
from stages.text_recovery import run_text_recovery
```

In `_run_one_cycle`, add Phase 5.5 after the PDF processing block (after line 156) and before the output generation block (line 158):

```python
    # Phase 5.5: Text recovery (Jmail backfill + OCR)
    if not _shutdown_requested:
        _set_activity("Stage 4.5 Text Recovery: backfill + OCR")
        logger.info("Stage 4.5: starting text recovery")
        min_wpp = cfg_get(cfg, "ocr.min_words_per_page", default=50)
        tr_count = run_text_recovery(conn, redaction_markers=markers,
                                     min_words_per_page=min_wpp)
        logger.info("Stage 4.5: recovered text for %d documents", tr_count)
```

Update the idle-loop sleep guard (around line 296-299) to also check for pending OCR work:

```python
            pending_ocr = conn.execute(
                "SELECT COUNT(*) FROM documents "
                "WHERE ocr_processed = 0 AND (extracted_text IS NULL OR extracted_text = '')"
            ).fetchone()[0]
            if stats.get("pending", 0) == 0 and pending_pdfs == 0 and pending_ocr == 0 and not _shutdown_requested:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/test_daemon.py -v 2>&1 | tail -20`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add unobfuscator.py tests/test_daemon.py
git commit -m "feat: wire Phase 5.5 text recovery into daemon loop"
```

---

### Task 8: Backfill CLI Command and Status Extension

**Files:**
- Modify: `unobfuscator.py:362-411,528-545`
- Test: manual CLI test

- [ ] **Step 1: Add `backfill` CLI command**

In `unobfuscator.py`, add after the `summary` command (before `if __name__`):

```python
@cli.command()
@click.pass_context
def backfill(ctx):
    """One-shot: backfill text from Jmail shards and run OCR on docs missing text."""
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    markers = cfg_get(cfg, "redaction_markers", default=[])
    min_wpp = cfg_get(cfg, "ocr.min_words_per_page", default=50)

    init_db(db_path)
    conn = get_connection(db_path)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    console.print("[dim]Running text recovery (Jmail backfill + OCR)...[/dim]")
    count = run_text_recovery(conn, redaction_markers=markers,
                              min_words_per_page=min_wpp)
    console.print(f"[green]Text recovery complete: {count} documents recovered.[/green]")
    conn.close()
```

- [ ] **Step 2: Extend `status` command with text recovery stats**

In the `status` command, add after the `pdf_done` / `pdf_total` queries (around line 386-389):

```python
    ocr_done = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE ocr_processed=1"
    ).fetchone()[0]
    ocr_pending = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE ocr_processed=0 "
        "AND (extracted_text IS NULL OR extracted_text = '')"
    ).fetchone()[0]
    text_sources = conn.execute(
        "SELECT text_source, COUNT(*) FROM documents "
        "WHERE text_source IS NOT NULL GROUP BY text_source"
    ).fetchall()
    source_str = ", ".join(f"{r[0]}: {r[1]:,}" for r in text_sources)
```

Add a row to the status table (after the PDF Processor row):

```python
    t.add_row("Text Recovery", f"{ocr_done:,} processed, {ocr_pending:,} pending")
    if source_str:
        t.add_row("Text Sources", source_str)
```

- [ ] **Step 3: Run the full test suite**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/ -v 2>&1 | tail -30`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add unobfuscator.py
git commit -m "feat: add backfill CLI command and text recovery status display"
```

---

### Task 9: Config and Dependencies

**Files:**
- Modify: `config.yaml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add OCR config section**

In `config.yaml`, add after the `redaction_markers` section:

```yaml

ocr:
  min_words_per_page: 50
```

- [ ] **Step 2: Add Python dependencies**

In `requirements.txt`, add:

```
pytesseract==0.3.13
Pillow==11.1.0
```

- [ ] **Step 3: Install dependencies**

Run: `pip3 install --break-system-packages pytesseract Pillow 2>&1 | tail -3`

Also ensure Tesseract is installed:
Run: `which tesseract || brew install tesseract`

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python -m pytest tests/ -v 2>&1 | tail -30`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add config.yaml requirements.txt
git commit -m "chore: add pytesseract, Pillow deps and ocr config section"
```

---

### Task 10: Integration Test with Live Data

**Files:**
- Test: manual verification

- [ ] **Step 1: Check current state**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && sqlite3 data/unobfuscator.db "SELECT COUNT(*) FROM documents WHERE extracted_text IS NULL OR extracted_text = ''"`

Record the count of docs without text.

- [ ] **Step 2: Run backfill command**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python unobfuscator.py backfill 2>&1 | tail -20`

This will run the Jmail shard backfill (should recover ~37k docs) and OCR (will process docs with cached PDFs).

- [ ] **Step 3: Verify results**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && sqlite3 data/unobfuscator.db "SELECT text_source, COUNT(*) FROM documents WHERE text_source IS NOT NULL GROUP BY text_source"`

Expected: `jmail` count should have increased significantly. May also see `pdf_text_layer` and/or `ocr` entries.

Run: `cd /Users/brianhill/Scripts/Unobfuscator && sqlite3 data/unobfuscator.db "SELECT COUNT(*) FROM documents WHERE extracted_text IS NULL OR extracted_text = ''"`

Expected: Count should be significantly lower than Step 1.

- [ ] **Step 4: Verify fingerprints were built**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && sqlite3 data/unobfuscator.db "SELECT COUNT(*) FROM document_fingerprints"`

Expected: Count should have increased from before backfill.

- [ ] **Step 5: Check status command**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python unobfuscator.py status`

Expected: Should show "Text Recovery" and "Text Sources" rows.

- [ ] **Step 6: Verify page tags for OCR'd docs**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && sqlite3 data/unobfuscator.db "SELECT page_tags FROM documents WHERE page_tags IS NOT NULL LIMIT 3"`

Expected: JSON strings like `{"0": "text", "1": "redacted"}`.

---

### Out of Scope

- **VOL00009 archive.org download** — handled by existing `download_datasets.py` pattern, separate from this plan. Once downloaded to `pdf_cache/VOL00009/`, text recovery will automatically pick up those files for OCR.
- **House Oversight JPG download** — source investigation deferred per spec. Text recovery will process any files already in `pdf_cache/`.
