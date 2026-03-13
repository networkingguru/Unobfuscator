"""Stage 5: Output Generator — produce highlighted PDFs with full source docs.

Only generates a file when at least one redaction was recovered.
GREEN highlight = recovered text in the destination (redacted) document.
YELLOW highlight = source text in the donor (unredacted) document.

Structure:
  Section 1: Full destination document text with GREEN on recovered passages
  Section 2: Full source document(s) text with YELLOW on passages used for recovery
  Section 3: Metadata page with working links to both documents

Logic reference: PIPELINE.md — Phase 6
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

from core.db import (
    get_merge_result, get_documents_by_ids,
    mark_output_generated, get_pending_output_groups
)

# Highlight colours as RGB tuples for PyMuPDF
GREEN = (0.0, 1.0, 0.0)    # recovered text in destination document
YELLOW = (1.0, 1.0, 0.0)   # source text in donor document


def build_output_path(output_dir: str, source: str, batch: str, doc_id: int) -> str:
    """Build output file path and create parent directories."""
    path = Path(output_dir) / source / batch / f"{doc_id}_merged.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _insert_text_multipage(pdf: fitz.Document, text: str, fontsize: int = 10) -> int:
    """Insert text across multiple pages. Returns number of pages created."""
    margin = 40
    line_height = fontsize * 1.5
    lines = text.split("\n")
    page = None
    y = 0
    max_y = 0
    page_width = 0
    pages_created = 0

    for line in lines:
        if page is None or y + line_height > max_y:
            page = pdf.new_page()
            pages_created += 1
            y = margin
            max_y = page.rect.height - 60
            page_width = page.rect.width - 2 * margin

        chars_per_line = max(int(page_width / (fontsize * 0.52)), 40)
        wrapped = [line[i:i + chars_per_line]
                   for i in range(0, max(len(line), 1), chars_per_line)]
        if not wrapped:
            wrapped = [""]

        for segment in wrapped:
            if y + line_height > max_y:
                page = pdf.new_page()
                pages_created += 1
                y = margin
                max_y = page.rect.height - 60
            page.insert_text((margin, y + fontsize), segment, fontsize=fontsize)
            y += line_height

    return pages_created


def _apply_highlights(page: fitz.Page, texts: list[str], color: tuple) -> None:
    """Highlight occurrences of each text string on the page.

    Splits multi-line recovered text into individual lines and searches for
    each line separately, since search_for works within contiguous text runs.
    Deduplicates search strings and rects to avoid double-annotating.
    Strips block-redaction characters (█■) since they render as dots in PDFs.
    """
    highlighted_rects: set[tuple] = set()
    seen_searches: set[str] = set()
    for text in texts:
        text = text.strip()
        if not text:
            continue
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip block chars that don't survive insert_text rendering
            clean_line = line.replace("█", "").replace("■", "").strip()
            if len(clean_line) < 4:
                continue
            search_text = clean_line[:70]
            if search_text in seen_searches:
                continue
            seen_searches.add(search_text)
            for rect in page.search_for(search_text):
                rect_key = (round(rect.x0, 1), round(rect.y0, 1),
                            round(rect.x1, 1), round(rect.y1, 1))
                if rect_key in highlighted_rects:
                    continue
                highlighted_rects.add(rect_key)
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()


def _write_section_header(pdf: fitz.Document, title: str) -> fitz.Page:
    """Add a page with a bold section header. Returns the page."""
    page = pdf.new_page()
    margin = 40
    fontsize = 14
    page.insert_text((margin, margin + fontsize), title, fontsize=fontsize)
    page.insert_text((margin, margin + fontsize + 20), "=" * 60, fontsize=9)
    return page


def _load_provenance(path: str) -> dict:
    """Load provenance data from JSON. Returns empty dict if file missing."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def generate_output_pdf(
    conn,
    group_id: int,
    output_dir: str,
    redaction_markers: list[str],
    provenance_path: str = None,
) -> Optional[str]:
    """Generate a highlighted output PDF for a merge group.

    Structure:
    1. DESTINATION DOCUMENT — merged text with GREEN highlights on recovered text
    2. SOURCE DOCUMENT(s) — full donor text with YELLOW highlights on passages
       that provided the recovered text
    3. METADATA — links to both documents, recovery stats
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
    # source_docs[0] is the base (destination), rest are donors
    base_doc = source_docs[0] if source_docs else {}
    donor_docs = source_docs[1:] if len(source_docs) > 1 else []

    # Check provenance for non-DOJ sources
    provenance_label = None
    if provenance_path:
        batch = base_doc.get("release_batch", "")
        if batch.startswith("VOL"):
            try:
                dataset_num = str(int(batch[3:]))  # VOL00008 → "8"
                prov_data = _load_provenance(provenance_path)
                if dataset_num in prov_data:
                    provenance_label = prov_data[dataset_num].get("source_label")
            except Exception:
                pass

    output_path = build_output_path(
        output_dir,
        source=base_doc.get("source", "unknown"),
        batch=base_doc.get("release_batch", "unknown"),
        doc_id=source_doc_ids[0] if source_doc_ids else group_id
    )

    pdf = fitz.open()

    # ── SECTION 1: DESTINATION DOCUMENT (merged text, GREEN highlights) ──
    _write_section_header(
        pdf,
        f"DESTINATION DOCUMENT — {base_doc.get('original_filename', '?')}"
    )
    dest_start_page = len(pdf)
    dest_pages = _insert_text_multipage(pdf, merged_text, fontsize=10)

    # Apply GREEN highlights on recovered text in destination pages
    recovered_texts = [seg.get("text", "").strip() for seg in recovered_segments]
    for i in range(dest_start_page, dest_start_page + dest_pages):
        _apply_highlights(pdf[i], recovered_texts, GREEN)

    # ── SECTION 2: SOURCE DOCUMENT(s) (full donor text, YELLOW highlights) ──
    # Group recovered segments by donor doc
    by_donor: dict[str, list[str]] = {}
    for seg in recovered_segments:
        did = seg.get("source_doc_id", "")
        text = seg.get("text", "").strip()
        if text:
            by_donor.setdefault(did, []).append(text)

    for donor in donor_docs:
        donor_id = donor["id"]
        donor_text = donor.get("extracted_text", "")
        donor_name = donor.get("original_filename", donor_id)

        _write_section_header(
            pdf,
            f"SOURCE DOCUMENT — {donor_name}"
        )
        src_start_page = len(pdf)
        src_pages = _insert_text_multipage(pdf, donor_text, fontsize=10)

        # Apply YELLOW highlights on the passages that provided recoveries
        donor_recovered = by_donor.get(donor_id, [])
        for i in range(src_start_page, src_start_page + src_pages):
            _apply_highlights(pdf[i], donor_recovered, YELLOW)

    # ── SECTION 3: METADATA ──
    _write_metadata_page(
        pdf, base_doc, donor_docs,
        recovered_count, soft_recovered_count,
        provenance_label=provenance_label,
    )

    pdf.save(output_path)
    pdf.close()

    mark_output_generated(conn, group_id)
    return output_path


def _write_metadata_page(
    pdf: fitz.Document,
    base_doc: dict,
    donor_docs: list[dict],
    recovered_count: int,
    soft_recovered_count: int,
    provenance_label: str = None,
) -> None:
    """Write the final metadata page with links and stats."""
    page = pdf.new_page()
    margin = 40
    fontsize = 9
    line_height = fontsize * 1.5
    y = margin

    def write(text, fs=fontsize):
        nonlocal page, y
        if y + line_height > page.rect.height - 60:
            page = pdf.new_page()
            y = margin
        page.insert_text((margin, y + fs), text, fontsize=fs)
        y += line_height

    today = date.today().isoformat()
    method_parts = []
    if recovered_count - soft_recovered_count > 0:
        method_parts.append("cross-document merge")
    if soft_recovered_count > 0:
        method_parts.append("soft redaction removal")
    method = " + ".join(method_parts) if method_parts else "cross-document merge"

    write("=" * 70)
    write(f"Unobfuscator v1.0 — {today}")
    write(f"Redactions recovered: {recovered_count}    |    Method: {method}")
    write("=" * 70)
    write("")

    if provenance_label:
        write("")
        write(f"\u26a0 {provenance_label}")
        write("")

    # Destination document
    base_url = base_doc.get("pdf_url") or ""
    write("DESTINATION DOCUMENT (redacted):")
    write(f"  {base_doc.get('original_filename', '?')}")
    write(f"  ID: {base_doc.get('id', '?')}")
    if base_url:
        write(f"  {base_url}")
    write("")

    # Source documents
    write(f"SOURCE DOCUMENT{'S' if len(donor_docs) != 1 else ''} (provided recovered text):")
    for i, d in enumerate(donor_docs, 1):
        url = d.get("pdf_url") or ""
        write(f"  [{i}] {d.get('original_filename', d['id'])}")
        write(f"      ID: {d['id']}")
        if url:
            write(f"      {url}")
    write("")
    write("=" * 70)
    write("Green highlights = text recovered from redactions (destination document)")
    write("Yellow highlights = original unredacted text (source document)")
    write("=" * 70)


def run_output_generator(conn, output_dir: str, redaction_markers: list[str],
                         provenance_path: str = None) -> int:
    """Generate output PDFs for all pending merge results. Returns count generated."""
    count = 0
    for row in get_pending_output_groups(conn):
        path = generate_output_pdf(conn, row["group_id"], output_dir, redaction_markers,
                                   provenance_path=provenance_path)
        if path:
            count += 1
    conn.commit()
    return count
