"""Stage 5.5: Text Recovery — backfill Jmail text, extract PDF text layers, OCR."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import duckdb

import fitz
import numpy as np
from PIL import Image

from core.api import resolve_shard
from core.db import (
    get_docs_needing_backfill, get_docs_needing_text_recovery,
    update_extracted_text, mark_ocr_processed, upsert_fingerprint
)
from stages.indexer import clean_text, shingle, build_fingerprint

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "pdf_cache"


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


def _is_tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_image(img: Image.Image) -> tuple[str, str]:
    """OCR a single PIL Image after pixel classification."""
    gray = img.convert("L") if img.mode != "L" else img
    pixels = np.array(gray)
    tag = classify_page_pixels(pixels)
    if tag != "text":
        return ("", tag)
    import pytesseract
    text = pytesseract.image_to_string(gray).strip()
    return (text, tag)


def ocr_pdf(pdf_bytes: bytes) -> tuple[str, str, str]:
    """OCR a PDF, classifying each page. Returns (text, "ocr", page_tags_json)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts = []
    page_tags = {}
    for page_num, page in enumerate(doc):
        pixmap = page.get_pixmap(colorspace=fitz.csGRAY)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w)
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


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------

def _build_and_store_fingerprint(conn, doc_id: str, text: str,
                                  redaction_markers: list[str]) -> None:
    cleaned = clean_text(text, redaction_markers)
    shingles = shingle(cleaned)
    if not shingles:
        return
    sig = build_fingerprint(cleaned)
    upsert_fingerprint(conn, doc_id, sig, len(shingles))


def _resolve_pdf_path(doc: dict) -> Optional[Path]:
    batch = doc.get("release_batch", "")
    filename = doc.get("original_filename", "")
    if batch and filename:
        local = _CACHE_DIR / batch / filename
        if local.is_file():
            return local
    return None


def _download_pdf_to_cache(doc: dict) -> Optional[Path]:
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
    doc_id = doc["id"]
    filename = doc.get("original_filename", "")

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
        try:
            text, tag = ocr_image(img)
        finally:
            img.close()
        tags_json = json.dumps({"0": tag})
        if text.strip():
            update_extracted_text(conn, doc_id, text, "ocr", page_tags=tags_json)
            _build_and_store_fingerprint(conn, doc_id, text, redaction_markers)
        else:
            conn.execute("UPDATE documents SET page_tags = ? WHERE id = ?", (tags_json, doc_id))
        mark_ocr_processed(conn, doc_id)
        return bool(text.strip())

    # PDF file
    try:
        pdf_bytes = file_path.read_bytes()
    except Exception as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        mark_ocr_processed(conn, doc_id)
        return False

    result = extract_text_from_pdf(pdf_bytes, min_words_per_page=min_words_per_page)
    if result is not None:
        text, source, tags = result
        update_extracted_text(conn, doc_id, text, source, page_tags=tags)
        _build_and_store_fingerprint(conn, doc_id, text, redaction_markers)
        mark_ocr_processed(conn, doc_id)
        return True

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
        conn.execute("UPDATE documents SET page_tags = ? WHERE id = ?", (tags_json, doc_id))
    mark_ocr_processed(conn, doc_id)
    return bool(text.strip())


def run_text_recovery(conn, redaction_markers: list[str],
                      min_words_per_page: int = 50) -> int:
    """Run the full text recovery pipeline. Returns count of docs recovered."""
    recovered = 0

    # --- Step 1: Jmail shard backfill ---
    # Download each needed shard parquet once to a temp file, then query
    # locally in small batches.  Remote parquet queries via DuckDB load
    # the full extracted_text column into memory (~1-2 GB per shard),
    # so re-downloading per batch is prohibitively expensive.
    _BACKFILL_QUERY_BATCH = 500

    db_batches = conn.execute(
        "SELECT DISTINCT release_batch FROM documents WHERE release_batch IS NOT NULL"
    ).fetchall()
    backfill_batches = {row[0] for row in db_batches}

    # Determine which shards actually have docs needing backfill.
    all_backfill_docs = get_docs_needing_backfill(
        conn, known_batches=backfill_batches, limit=100_000
    )
    shards_needed: dict[str, list[dict]] = {}
    for doc in all_backfill_docs:
        shard = resolve_shard(doc["release_batch"])
        shards_needed.setdefault(shard, []).append(doc)
    del all_backfill_docs

    for shard, shard_docs in shards_needed.items():
        url = f"https://data.jmail.world/v1/documents-full/{shard}.parquet"
        logger.info("Backfill: downloading %s shard (%d docs to check)...",
                     shard, len(shard_docs))

        # Download shard to a temp file so we can query locally
        # without re-downloading for each batch.
        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet",
                                              delete=False) as tmp:
                tmp_path = tmp.name
            with duckdb.connect() as ddb:
                ddb.execute("SET force_download=true")
                ddb.execute(f"""
                    COPY (SELECT id, extracted_text
                          FROM read_parquet('{url}'))
                    TO '{tmp_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """)
        except Exception as e:
            logger.warning("Failed to download shard %s: %s", shard, e)
            for doc in shard_docs:
                conn.execute(
                    "UPDATE documents SET text_source = 'backfill_miss' "
                    "WHERE id = ?", (doc["id"],)
                )
            conn.commit()
            continue

        logger.info("Backfill: shard %s downloaded, querying locally...", shard)

        # Query the local file in batches to keep memory bounded.
        try:
            for i in range(0, len(shard_docs), _BACKFILL_QUERY_BATCH):
                batch = shard_docs[i:i + _BACKFILL_QUERY_BATCH]
                doc_ids = [d["id"] for d in batch]
                ids_str = ", ".join(f"'{did}'" for did in doc_ids)
                with duckdb.connect() as ddb:
                    rows = ddb.execute(f"""
                        SELECT id, extracted_text
                        FROM read_parquet('{tmp_path}')
                        WHERE id IN ({ids_str})
                    """).fetchall()
                text_map = {r[0]: r[1] for r in rows}
                del rows

                for doc in batch:
                    text = text_map.get(doc["id"])
                    if text and isinstance(text, str) and text.strip():
                        update_extracted_text(conn, doc["id"], text, "jmail")
                        _build_and_store_fingerprint(
                            conn, doc["id"], text, redaction_markers
                        )
                        mark_ocr_processed(conn, doc["id"])
                        recovered += 1
                    else:
                        conn.execute(
                            "UPDATE documents SET text_source = 'backfill_miss' "
                            "WHERE id = ?", (doc["id"],)
                        )
                conn.commit()
                del text_map
                logger.info("Backfill: %s — processed %d/%d docs",
                            shard, min(i + _BACKFILL_QUERY_BATCH, len(shard_docs)),
                            len(shard_docs))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

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
