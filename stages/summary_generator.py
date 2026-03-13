"""Summary report generator — extract entities from recovered segments and build a PDF index.

Logic reference: docs/superpowers/specs/2026-03-13-summary-pdf-design.md
"""

import json
import re
from datetime import date
from pathlib import Path

import fitz

from core.db import get_all_recovery_groups, get_documents_by_ids

# Stopwords for people extraction — common legal/geographic/email phrases
_PEOPLE_STOPWORDS = {
    # Geographic
    "united states", "new york", "palm beach", "west palm beach",
    "south florida", "north florida",
    # Legal/court
    "district court", "southern district", "northern district",
    "eastern district", "western district", "judicial circuit",
    "this report", "the united", "the honorable",
    "certificate of service", "trust agreement", "dispositive provisions",
    "related touhy requests", "deposition attachment",
    # Travel/business
    "air france", "royal air", "american express", "american express travel",
    "flight information", "other information", "hotel information",
    "travel service", "centurion travel service", "privacy statement",
    "entry and exit", "economy class", "estimated time",
    "ticket number", "detail sheet", "airline record locator",
    "best regards", "original message",
    # Email date headers (On Mon, On Tue, etc.)
    "on jan", "on feb", "on mar", "on apr", "on may", "on jun",
    "on jul", "on aug", "on sep", "on oct", "on nov", "on dec",
    "on mon", "on tue", "on wed", "on thu", "on fri", "on sat", "on sun",
    # Time zones / email artifacts
    "pm edt", "am edt", "pm est", "am est", "pm pst", "am pst",
    # Technical/data artifacts
    "incl ticketno", "inquiry regarding", "amlcompliance inquiries",
    "sec owb", "knight capital", "danya friedman on behalf",
    "for palm beach county", "judicial circuit in and",
    "behrend dr ste", "australian ave",
}

_ORG_SUFFIXES = re.compile(
    r'\b(?:LLC|Inc|Corp|Department|Agency|Bureau|Office|Court|FBI|DOJ|OIG|CIA'
    r'|Police|Division|Commission|Authority|Association|Bank|Trust|Capital'
    r'|Securities|Group|Partners|Services|Foundation)\b',
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
_PHONE_RE = re.compile(
    r'(?:\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})'
    r'|\b\d{10}\b'
)
_CASE_RE = re.compile(r'\b\d{2,4}[-–][A-Z]{1,4}[-–]\d{4,}')
_NAME_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b|'
                       r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b')


def extract_entities(text: str) -> list[dict]:
    """Extract categorized entities from a recovered text segment.

    Returns list of {"text": str, "category": str} dicts.
    Extractors run in priority order; matched spans are excluded from later extractors.
    Multi-line segments are split and each line is processed independently.
    """
    # Strip block chars
    text = text.replace("█", " ").replace("■", " ")

    # Split on newlines and process each line to avoid cross-line false matches
    all_entities = []
    for line in text.split("\n"):
        line = re.sub(r'\s+', ' ', line).strip()
        if len(line) < 4:
            continue
        all_entities.extend(_extract_from_line(line))
    return all_entities


def _extract_from_line(text: str) -> list[dict]:
    """Extract entities from a single line of text."""
    entities = []
    consumed: set[tuple[int, int]] = set()

    def _is_consumed(start: int, end: int) -> bool:
        for cs, ce in consumed:
            if start < ce and end > cs:
                return True
        return False

    # 1. Emails
    for m in _EMAIL_RE.finditer(text):
        if not _is_consumed(m.start(), m.end()):
            entities.append({"text": m.group(), "category": "email"})
            consumed.add((m.start(), m.end()))

    # 2. Phones
    for m in _PHONE_RE.finditer(text):
        if not _is_consumed(m.start(), m.end()):
            entities.append({"text": m.group(), "category": "phone"})
            consumed.add((m.start(), m.end()))

    # 3. Case numbers
    for m in _CASE_RE.finditer(text):
        if not _is_consumed(m.start(), m.end()):
            entities.append({"text": m.group(), "category": "case_number"})
            consumed.add((m.start(), m.end()))

    # 4. Organizations (suffix match)
    for m in _ORG_SUFFIXES.finditer(text):
        # Build org name: look backward for preceding words and forward for "of X" phrases
        prefix_start = max(0, m.start() - 60)
        prefix = text[prefix_start:m.end()]
        # Find capitalized words leading into the suffix
        org_match = re.search(
            r'(?:(?:the|of|for|and|in)\s+)*(?:[A-Z][a-zA-Z]*\s+)*' + re.escape(m.group()),
            prefix, re.IGNORECASE
        )
        if not org_match:
            continue
        org_name = org_match.group().strip()
        abs_start = prefix_start + org_match.start()
        abs_end = prefix_start + org_match.end()
        # Extend forward: grab trailing "of X", "for X" phrases
        tail = text[abs_end:]
        tail_match = re.match(r'(?:\s+(?:of|for|and|in|the)\s+[A-Z][a-zA-Z]*)+', tail)
        if tail_match:
            org_name += tail_match.group()
            abs_end += tail_match.end()
        # Strip leading articles
        org_name = re.sub(r'^(?:the|of|for)\s+', '', org_name, flags=re.IGNORECASE).strip()
        if len(org_name) > 3 and not _is_consumed(abs_start, abs_end):
            entities.append({"text": org_name, "category": "organization"})
            consumed.add((abs_start, abs_end))

    # 5. People names
    for m in _NAME_RE.finditer(text):
        name = m.group(1) or m.group(2)
        if not name or _is_consumed(m.start(), m.end()):
            continue
        # Check stopwords — exact match or any stopword is a substring
        name_lower = name.lower()
        if name_lower in _PEOPLE_STOPWORDS:
            continue
        if any(sw in name_lower for sw in _PEOPLE_STOPWORDS if len(sw) > 5):
            continue
        # Skip single-word matches (regex requires 2+, but verify)
        if len(name.split()) < 2:
            continue
        entities.append({"text": name, "category": "people"})
        consumed.add((m.start(), m.end()))

    return entities


def _normalize_key(text: str, category: str) -> str:
    """Normalize entity text for deduplication."""
    if category == "email":
        return text.lower()
    if category == "phone":
        return re.sub(r'\D', '', text)
    if category == "case_number":
        return text  # exact match per spec
    return text.lower()


def aggregate_entities(raw_entities: list[dict]) -> list[dict]:
    """Deduplicate and count entities. Returns sorted list per category.

    Input: list of {"text": str, "category": str, "group_id": int}
    Output: list of {"text": str, "category": str, "count": int, "group_ids": list[int]}
    """
    buckets: dict[str, dict] = {}  # key -> {text, category, count, group_ids, casing_counts}

    for e in raw_entities:
        key = f"{e['category']}:{_normalize_key(e['text'], e['category'])}"
        if key not in buckets:
            buckets[key] = {
                "text": e["text"],
                "category": e["category"],
                "count": 0,
                "group_ids": [],
                "casing_counts": {},
            }
        bucket = buckets[key]
        bucket["count"] += 1
        if e["group_id"] not in bucket["group_ids"]:
            bucket["group_ids"].append(e["group_id"])
        # Track casing to pick the most common form
        bucket["casing_counts"][e["text"]] = bucket["casing_counts"].get(e["text"], 0) + 1

    result = []
    for bucket in buckets.values():
        # Use the most common casing
        best_text = max(bucket["casing_counts"], key=bucket["casing_counts"].get)
        result.append({
            "text": best_text,
            "category": bucket["category"],
            "count": bucket["count"],
            "group_ids": sorted(bucket["group_ids"]),
        })

    # Sort by count descending within each category
    result.sort(key=lambda e: (-e["count"], e["text"]))
    return result


# Category display order and labels
_CATEGORY_ORDER = [
    ("people", "People"),
    ("email", "Email Addresses"),
    ("phone", "Phone Numbers"),
    ("case_number", "Case Numbers"),
    ("organization", "Organizations"),
    ("other", "Other Recoveries"),
]


def _collect_raw_entities(conn) -> list[dict]:
    """Parse all recovered segments and extract entities with group context.

    Segments that yield no categorized entities are added as "other" category
    (truncated to 80 chars).
    """
    groups = get_all_recovery_groups(conn)
    raw = []
    for g in groups:
        segments = json.loads(g["recovered_segments"] or "[]")
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text or len(text) < 4:
                continue
            entities = extract_entities(text)
            if entities:
                for e in entities:
                    e["group_id"] = g["group_id"]
                    raw.append(e)
            else:
                # "Other" category — truncated raw text
                display = text.replace("\n", " ")[:80]
                raw.append({
                    "text": display,
                    "category": "other",
                    "group_id": g["group_id"],
                })
    return raw


def _write_title_page(pdf: fitz.Document, conn, entity_count: int,
                      segment_count: int) -> None:
    """Write page 1: title and high-level stats."""
    page = pdf.new_page()
    margin = 40
    y = margin
    fontsize = 16

    page.insert_text((margin, y + fontsize), "Unobfuscator Summary Report",
                     fontsize=fontsize)
    y += fontsize * 2

    fontsize = 10
    line_height = fontsize * 1.8
    today = date.today().isoformat()

    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    total_groups = conn.execute(
        "SELECT COUNT(*) FROM merge_results WHERE recovered_count > 0"
    ).fetchone()[0]
    recovered_sum = conn.execute(
        "SELECT COALESCE(SUM(recovered_count), 0) FROM merge_results"
    ).fetchone()[0]
    redacted_sum = conn.execute(
        "SELECT COALESCE(SUM(total_redacted), 0) FROM merge_results"
    ).fetchone()[0]
    rate = f"{recovered_sum/redacted_sum:.1%}" if redacted_sum > 0 else "N/A"

    stats = [
        f"Generated: {today}",
        f"Documents indexed: {total_docs:,}",
        f"Match groups with recoveries: {total_groups:,}",
        f"Total recovered segments: {segment_count:,}",
        f"Unique entities extracted: {entity_count:,}",
        f"Recovery rate: {rate} ({recovered_sum:,} / {redacted_sum:,} redactions)",
    ]
    page.insert_text((margin, y + fontsize), "=" * 60, fontsize=9)
    y += line_height
    for line in stats:
        page.insert_text((margin, y + fontsize), line, fontsize=fontsize)
        y += line_height
    page.insert_text((margin, y + fontsize), "=" * 60, fontsize=9)


def _write_top_findings(pdf: fitz.Document, entities: list[dict]) -> None:
    """Write page 2: top 20 entities across all categories."""
    page = pdf.new_page()
    margin = 40
    fontsize = 12
    y = margin

    page.insert_text((margin, y + fontsize), "Top Findings", fontsize=fontsize)
    y += fontsize * 2

    fontsize = 10
    line_height = fontsize * 1.5
    page.insert_text((margin, y + fontsize), "=" * 60, fontsize=9)
    y += line_height

    top = entities[:20]
    for i, e in enumerate(top, 1):
        label = f"{i:>2}. [{e['category'].upper()}] {e['text']} — {e['count']}x"
        if y + line_height > page.rect.height - 60:
            page = pdf.new_page()
            y = margin
        page.insert_text((margin, y + fontsize), label, fontsize=fontsize)
        y += line_height


def _write_category_section(pdf: fitz.Document, label: str,
                            entities: list[dict], conn,
                            output_dir: str) -> None:
    """Write a category section with entity table and links."""
    page = pdf.new_page()
    margin = 40
    fontsize = 12
    y = margin

    page.insert_text((margin, y + fontsize),
                     f"{label} ({len(entities)} unique)", fontsize=fontsize)
    y += fontsize * 2

    fontsize = 9
    line_height = fontsize * 1.6

    for e in entities:
        # Entity line
        group_str = ", ".join(str(g) for g in e["group_ids"][:10])
        if len(e["group_ids"]) > 10:
            group_str += f" (+{len(e['group_ids']) - 10} more)"
        line = f"{e['text']}  —  {e['count']}x  —  groups: {group_str}"

        if y + line_height * 2 > page.rect.height - 60:
            page = pdf.new_page()
            y = margin

        page.insert_text((margin, y + fontsize), line[:100], fontsize=fontsize)
        y += line_height

        # Links for first 3 groups
        for gid in e["group_ids"][:3]:
            source_doc_ids = _get_source_doc_ids(conn, gid)
            if source_doc_ids:
                base_id = source_doc_ids[0]
                docs = get_documents_by_ids(conn, [base_id])
                if docs:
                    d = docs[0]
                    local_path = str(
                        Path(output_dir) / d.get("source", "unknown")
                        / d.get("release_batch", "unknown")
                        / f"{base_id}_merged.pdf"
                    )
                    url = d.get("pdf_url") or ""
                    if y + line_height * 2 > page.rect.height - 60:
                        page = pdf.new_page()
                        y = margin
                    # Local file link
                    local_label = f"    Local: {local_path}"
                    page.insert_text((margin + 10, y + fontsize),
                                     local_label[:90], fontsize=fontsize)
                    local_width = fitz.get_text_length(local_label[:90], fontsize=fontsize)
                    page.insert_link({
                        "kind": fitz.LINK_LAUNCH,
                        "from": fitz.Rect(margin + 10, y,
                                          margin + 10 + local_width, y + fontsize + 2),
                        "file": local_path,
                    })
                    y += line_height
                    # DOJ source URL link
                    if url:
                        url_label = f"    Source: {url}"
                        page.insert_text((margin + 10, y + fontsize),
                                         url_label[:90], fontsize=fontsize)
                        url_width = fitz.get_text_length(url_label[:90], fontsize=fontsize)
                        page.insert_link({
                            "kind": fitz.LINK_URI,
                            "from": fitz.Rect(margin + 10, y,
                                              margin + 10 + url_width, y + fontsize + 2),
                            "uri": url,
                        })
                        y += line_height

        y += line_height * 0.5  # spacing between entities


def _get_source_doc_ids(conn, group_id: int) -> list[str]:
    """Get source_doc_ids for a group from merge_results."""
    row = conn.execute(
        "SELECT source_doc_ids FROM merge_results WHERE group_id = ?",
        (group_id,)
    ).fetchone()
    if row and row["source_doc_ids"]:
        return json.loads(row["source_doc_ids"])
    return []


def generate_summary_pdf(conn, output_dir: str) -> str:
    """Generate the summary report PDF. Returns the output path."""
    raw = _collect_raw_entities(conn)
    aggregated = aggregate_entities(raw)

    # Count total segments
    groups = get_all_recovery_groups(conn)
    segment_count = sum(
        len(json.loads(g["recovered_segments"] or "[]")) for g in groups
    )

    output_path = str(Path(output_dir) / "summary_report.pdf")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open()

    # Page 1: Title and stats
    _write_title_page(pdf, conn, len(aggregated), segment_count)

    # Page 2: Top findings
    _write_top_findings(pdf, aggregated)

    # Category sections
    for cat_key, cat_label in _CATEGORY_ORDER:
        cat_entities = [e for e in aggregated if e["category"] == cat_key]
        if cat_entities:
            _write_category_section(pdf, cat_label, cat_entities, conn, output_dir)

    pdf.save(output_path)
    pdf.close()
    return output_path
