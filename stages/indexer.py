"""Stage 1: Indexer — fetch documents from Jmail and build MinHash fingerprints.

Logic reference: PIPELINE.md — Phase 1 (Fingerprinting)
"""

import re
import numpy as np
from typing import Optional
from datasketch import MinHash
from core.api import fetch_documents_metadata, fetch_documents_text_batch
from core.db import (
    upsert_document, upsert_fingerprint, mark_text_processed
)


def clean_text(text: str, redaction_markers: list[str]) -> str:
    """Remove redaction markers, normalize whitespace, lowercase."""
    for marker in redaction_markers:
        text = text.replace(marker, " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shingle(text: str, window: int = 8) -> list[str]:
    """Slice text into overlapping n-word windows (shingles)."""
    words = text.split()
    if len(words) < window:
        return []
    return [" ".join(words[i:i + window]) for i in range(len(words) - window + 1)]


def build_fingerprint(text: str, num_perm: int = 128) -> bytes:
    """Generate a MinHash signature for the given text.

    Returns raw bytes of the uint64 hashvalues array.
    Deserialize in matcher.py with: np.frombuffer(sig, dtype=np.uint64)
    """
    m = MinHash(num_perm=num_perm)
    for s in shingle(text):
        m.update(s.encode("utf-8"))
    return m.hashvalues.tobytes()


def index_document(conn, doc: dict, redaction_markers: list[str],
                   num_perm: int = 128) -> None:
    """Store a document in the DB and generate its MinHash fingerprint."""
    upsert_document(conn, doc)

    text = doc.get("extracted_text") or ""
    if not text.strip():
        mark_text_processed(conn, doc["id"])
        return

    cleaned = clean_text(text, redaction_markers)
    shingles = shingle(cleaned)
    if not shingles:
        mark_text_processed(conn, doc["id"])
        return

    sig = build_fingerprint(cleaned, num_perm=num_perm)
    upsert_fingerprint(conn, doc["id"], sig, len(shingles))
    mark_text_processed(conn, doc["id"])


def run_indexer_batch(conn, batch_id: Optional[str],
                      redaction_markers: list[str],
                      num_perm: int = 128) -> int:
    """Fetch all documents for a batch from Jmail and index them. Returns count."""
    docs = fetch_documents_metadata(batch_id=batch_id)
    if not docs:
        return 0
    doc_ids = [meta["id"] for meta in docs]
    text_map = fetch_documents_text_batch(doc_ids, batch_id) if batch_id else {}
    count = 0
    for meta in docs:
        doc = {
            **meta,
            "extracted_text": text_map.get(meta["id"]) or "",
            "pdf_url": meta.get("source_url"),
        }
        index_document(conn, doc, redaction_markers, num_perm)
        conn.commit()
        count += 1
    return count
