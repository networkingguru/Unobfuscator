"""Stage 5: Output Generator — produce highlighted PDFs with footnote pages.

Only generates a file when at least one redaction was recovered.
Yellow highlight = cross-document merge recovery (Stage 3).
Green highlight  = soft redaction removal (Stage 4).

Logic reference: PIPELINE.md — Phase 6
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

from core.db import (
    get_merge_result, get_documents_by_ids,
    mark_output_generated, get_pending_output_groups
)

# Highlight colours as RGB tuples for PyMuPDF
YELLOW = (1.0, 1.0, 0.0)
GREEN = (0.0, 1.0, 0.0)

_JMAIL_DOC_BASE = "https://data.jmail.world/v1/documents"


def build_output_path(output_dir: str, source: str, batch: str, doc_id: int) -> str:
    """Build output file path and create parent directories."""
    path = Path(output_dir) / source / batch / f"{doc_id}_merged.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _apply_highlights(page: fitz.Page, segments: list[dict]) -> None:
    """Search for each recovered segment on the page and add a highlight annotation.

    Yellow for Stage 3 (cross-doc merge), Green for Stage 4 (soft redaction removal).
    """
    for seg in segments:
        text = seg.get("text", "").strip()
        stage = seg.get("stage", "merge")
        if not text:
            continue
        colour = GREEN if stage == "pdf" else YELLOW
        for rect in page.search_for(text):
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=colour)
            annot.update()


def _build_footnote_page(
    doc: fitz.Document,
    source_docs: list[dict],
    recovered_count: int,
    soft_recovered_count: int,
) -> None:
    """Append a footnote page listing all source documents and recovery stats."""
    page = doc.new_page()
    today = date.today().isoformat()

    method_parts = []
    if recovered_count - soft_recovered_count > 0:
        method_parts.append("cross-document merge")
    if soft_recovered_count > 0:
        method_parts.append("soft redaction removal")
    method = " + ".join(method_parts) if method_parts else "cross-document merge"

    lines = [
        "=" * 60,
        f"SOURCES — Unobfuscator v1.0 — {today}",
        "=" * 60,
        "",
        "This document was reconstructed from the following sources:",
        "",
    ]
    for i, src in enumerate(source_docs, 1):
        lines.append(
            f"[{i}] Document ID {src['id']} — {src['source']} "
            f"{src['release_batch']} — {src['original_filename']}"
        )
        lines.append(f"    {_JMAIL_DOC_BASE}/{src['id']}")
        lines.append("")

    lines += [
        f"Redactions recovered:      {recovered_count}",
        f"Soft redactions removed:   {soft_recovered_count}",
        f"Recovery method: {method}",
        "=" * 60,
    ]

    y = 50
    for line in lines:
        page.insert_text((40, y), line, fontsize=9)
        y += 14


def generate_output_pdf(
    conn,
    group_id: int,
    output_dir: str,
    redaction_markers: list[str],
) -> Optional[str]:
    """Generate a highlighted output PDF for a merge group.

    Returns the output file path, or None if no redactions were recovered.
    Logic reference: PIPELINE.md — Phase 6
    """
    merge_row = get_merge_result(conn, group_id)
    if not merge_row or merge_row["recovered_count"] == 0:
        return None

    merged_text = merge_row["merged_text"]
    source_doc_ids = json.loads(merge_row["source_doc_ids"])
    recovered_count = merge_row["recovered_count"]
    soft_recovered_count = merge_row.get("soft_recovered_count") or 0
    recovered_segments = json.loads(merge_row.get("recovered_segments") or "[]")

    source_docs = get_documents_by_ids(conn, source_doc_ids)
    base_doc = source_docs[0] if source_docs else {}

    output_path = build_output_path(
        output_dir,
        source=base_doc.get("source", "unknown"),
        batch=base_doc.get("release_batch", "unknown"),
        doc_id=source_doc_ids[0] if source_doc_ids else group_id
    )

    pdf = fitz.open()
    page = pdf.new_page()
    rect = fitz.Rect(40, 40, page.rect.width - 40, page.rect.height - 60)
    page.insert_textbox(rect, merged_text, fontsize=10)

    _apply_highlights(page, recovered_segments)
    _build_footnote_page(pdf, source_docs, recovered_count, soft_recovered_count)

    pdf.save(output_path)
    pdf.close()

    mark_output_generated(conn, group_id)
    return output_path


def run_output_generator(conn, output_dir: str, redaction_markers: list[str]) -> int:
    """Generate output PDFs for all pending merge results. Returns count generated."""
    count = 0
    for row in get_pending_output_groups(conn):
        path = generate_output_pdf(conn, row["group_id"], output_dir, redaction_markers)
        if path:
            count += 1
    conn.commit()
    return count
