"""Stage 4: PDF Processor — download PDFs and strip soft/overlay redactions.

A "soft redaction" is one where a black rectangle was drawn over text in Acrobat
but the underlying text was never removed from the PDF stream.
PyMuPDF can extract the hidden text by reading the page stream directly.

Logic reference: PIPELINE.md — Phase 5
"""

import json
import logging
import os
import fitz  # PyMuPDF
import httpx
from pathlib import Path
from typing import Optional
from core.db import get_doc_group, get_document_for_pdf, append_soft_redaction_text, mark_pdf_processed
from core.queue import enqueue

logger = logging.getLogger(__name__)

_AGE_VERIFY_COOKIE = ("justiceGovAgeVerified", "true")
_CACHE_DIR = Path(__file__).resolve().parent.parent / "pdf_cache"


def _download_pdf(url: str) -> httpx.Response:
    """Download a PDF, handling the DOJ age-verification gate if needed."""
    cookies = {}
    if "justice.gov" in url:
        cookies[_AGE_VERIFY_COOKIE[0]] = _AGE_VERIFY_COOKIE[1]
    response = httpx.get(url, timeout=30, follow_redirects=True, cookies=cookies)
    response.raise_for_status()
    return response


def extract_soft_redactions(pdf_bytes: bytes) -> list[dict]:
    """Detect and extract text hidden under black rectangle overlays.

    Returns list of dicts: [{page: int, rect: tuple, text: str}, ...]
    Returns empty list if no soft redactions found or PDF is invalid.

    Logic reference: PIPELINE.md — Phase 5, steps 1–2
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return []

    recovered = []
    for page_num, page in enumerate(doc):
        # Find all filled black rectangles on the page (redaction overlays)
        drawings = page.get_drawings()
        for drawing in drawings:
            if drawing.get("fill") != (0.0, 0.0, 0.0):
                continue  # Not a black fill
            rect = fitz.Rect(drawing["rect"])
            if rect.width < 5 or rect.height < 5:
                continue  # Too small to be a redaction

            # Extract text from the PDF stream in that rectangle area
            clip_rect = fitz.Rect(drawing["rect"])
            hidden_text = page.get_text("text", clip=clip_rect).strip()
            if hidden_text:
                recovered.append({
                    "page": page_num,
                    "rect": tuple(rect),
                    "text": hidden_text
                })

    doc.close()
    return recovered


def process_pdf_for_document(conn, doc_id: int) -> None:
    """Download a document's PDF, check for soft redactions, queue merge job if found.

    Logic reference: PIPELINE.md — Phase 5, steps 1–3
    """
    doc = get_document_for_pdf(conn, doc_id)
    if not doc or not doc["pdf_url"]:
        return  # No PDF URL — skip silently

    # Check local pdf_cache before downloading from the network.
    # Cached files live at pdf_cache/{release_batch}/{original_filename}.
    pdf_bytes = None
    batch = doc.get("release_batch", "")
    filename = doc.get("original_filename", "")
    if batch and filename:
        local_path = _CACHE_DIR / batch / filename
        if local_path.is_file():
            pdf_bytes = local_path.read_bytes()
            logger.debug("Loaded PDF from local cache: %s", local_path)

    if pdf_bytes is None:
        try:
            response = _download_pdf(doc["pdf_url"])
            pdf_bytes = response.content
        except Exception as e:
            logger.warning("Failed to download PDF for doc %s (%s): %s", doc_id, doc.get("pdf_url"), e)
            # Mark as processed so permanently broken URLs don't block the queue.
            # If the URL becomes available later, reset pdf_processed via a manual query.
            mark_pdf_processed(conn, doc_id)
            return

    soft_redactions = extract_soft_redactions(pdf_bytes)

    if soft_redactions:
        recovered_text = "\n".join(r["text"] for r in soft_redactions)
        append_soft_redaction_text(conn, doc_id, recovered_text)

        group_id = get_doc_group(conn, doc_id)
        if group_id is not None:
            enqueue(conn, stage="merge", payload={"group_id": group_id}, priority=50)

    mark_pdf_processed(conn, doc_id)
