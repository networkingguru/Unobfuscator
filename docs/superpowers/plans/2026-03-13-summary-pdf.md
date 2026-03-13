# Summary PDF Report Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a single summary PDF that indexes all recovered entities (names, emails, phones, case numbers, orgs) across the corpus, ranked by frequency with links to source documents.

**Architecture:** New `stages/summary_generator.py` with regex entity extractors and PyMuPDF PDF builder. New DB helper in `core/db.py`. New CLI command in `unobfuscator.py`. All entity extraction is heuristic regex — no new dependencies.

**Tech Stack:** Python 3.14, SQLite, PyMuPDF (fitz), click (CLI)

---

## Chunk 1: Entity Extraction

### Task 1: DB Helper — get_all_recovery_groups

**Files:**
- Modify: `core/db.py` (add function at end of file)
- Test: `tests/stages/test_summary_generator.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/stages/test_summary_generator.py`:

```python
import pytest
import json
from core.db import (
    init_db, get_connection, upsert_document,
    create_match_group, add_group_member, upsert_merge_result,
    get_all_recovery_groups,
)


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def seed_group(conn, group_docs, merged_text, recovered_count,
               segments, source="doj", batch="VOL00001"):
    """Seed a match group with documents, merge result, and segments."""
    for doc_id, text in group_docs:
        upsert_document(conn, {
            "id": doc_id, "source": source, "release_batch": batch,
            "original_filename": f"{doc_id}.pdf", "page_count": 1,
            "size_bytes": 500, "description": "", "extracted_text": text,
            "pdf_url": f"https://example.com/{doc_id}.pdf",
        })
    g = create_match_group(conn)
    for doc_id, _ in group_docs:
        add_group_member(conn, g, doc_id, 0.9)
    upsert_merge_result(conn, g, merged_text, recovered_count=recovered_count,
                        total_redacted=max(recovered_count, 1),
                        source_doc_ids=[d[0] for d in group_docs],
                        recovered_segments=segments)
    conn.commit()
    return g


def test_get_all_recovery_groups_returns_groups_with_recoveries(conn):
    seed_group(conn,
               group_docs=[("doc-a", "base"), ("doc-b", "donor")],
               merged_text="merged", recovered_count=1,
               segments=[{"text": "SARAH KELLEN", "source_doc_id": "doc-b", "stage": "merge"}])
    # Group with zero recoveries should be excluded
    seed_group(conn,
               group_docs=[("doc-c", "base2"), ("doc-d", "donor2")],
               merged_text="no recovery", recovered_count=0, segments=[])

    rows = get_all_recovery_groups(conn)
    assert len(rows) == 1
    assert rows[0]["recovered_count"] > 0


def test_get_all_recovery_groups_empty_db(conn):
    rows = get_all_recovery_groups(conn)
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py -x -v 2>&1 | tail -20`
Expected: FAIL with ImportError (get_all_recovery_groups not defined)

- [ ] **Step 3: Write minimal implementation**

Add to end of `core/db.py`:

```python
def get_all_recovery_groups(conn) -> list[dict]:
    """Return all merge results that have at least one recovery."""
    rows = conn.execute("""
        SELECT group_id, recovered_count, total_redacted,
               recovered_segments, source_doc_ids
        FROM merge_results
        WHERE recovered_count > 0
    """).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py -x -v 2>&1 | tail -20`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/db.py tests/stages/test_summary_generator.py
git commit -m "feat: add get_all_recovery_groups DB helper for summary report"
```

---

### Task 2: Entity Extractors

**Files:**
- Create: `stages/summary_generator.py`
- Test: `tests/stages/test_summary_generator.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/stages/test_summary_generator.py`:

```python
from stages.summary_generator import extract_entities


def test_extract_people():
    entities = extract_entities("The passenger list included SARAH KELLEN and others.")
    people = [e for e in entities if e["category"] == "people"]
    assert any(e["text"] == "SARAH KELLEN" for e in people)


def test_extract_emails():
    entities = extract_entities("Contact at sarah@example.com for details.")
    emails = [e for e in entities if e["category"] == "email"]
    assert any(e["text"] == "sarah@example.com" for e in emails)


def test_extract_phones():
    entities = extract_entities("Call (877) 877-0987 for info.")
    phones = [e for e in entities if e["category"] == "phone"]
    assert len(phones) == 1


def test_extract_case_numbers():
    entities = extract_entities("Reference case 72-MM-113327 filed today.")
    cases = [e for e in entities if e["category"] == "case_number"]
    assert any("72-MM-113327" in e["text"] for e in cases)


def test_extract_organizations():
    entities = extract_entities("Filed with the Department of Justice today.")
    orgs = [e for e in entities if e["category"] == "organization"]
    assert any("Department of Justice" in e["text"] for e in orgs)


def test_no_double_counting_email_as_name():
    """Email addresses should not also be extracted as people names."""
    entities = extract_entities("From: Sarah Kellen <sarah@example.com>")
    names = [e["text"] for e in entities if e["category"] == "people"]
    assert "Sarah Kellen" in names
    # The email should be in email category, not duplicated as a name
    email_texts = [e["text"] for e in entities if e["category"] == "email"]
    assert "sarah@example.com" in email_texts


def test_people_stopwords_excluded():
    entities = extract_entities("The United States District Court ruled today.")
    people = [e for e in entities if e["category"] == "people"]
    people_texts = [e["text"] for e in people]
    assert "United States" not in people_texts
    assert "District Court" not in people_texts


def test_short_junk_skipped():
    entities = extract_entities(")OOO{XXXXX")
    assert len(entities) == 0


def test_block_chars_stripped():
    entities = extract_entities("████████ some real content here")
    # Should not crash; block chars stripped before extraction
    for e in entities:
        assert "█" not in e["text"]


def test_multi_category_from_single_segment():
    """A segment with both a name and phone should produce entities in both categories."""
    entities = extract_entities("Contact SARAH KELLEN at (877) 877-0987")
    categories = {e["category"] for e in entities}
    assert "people" in categories
    assert "phone" in categories


def test_multiline_segment_extracted_per_line():
    """Newlines should not cause cross-line false matches."""
    entities = extract_entities("SARAH KELLEN\n(877) 877-0987\ntest@example.com")
    categories = {e["category"] for e in entities}
    assert "people" in categories
    assert "phone" in categories
    assert "email" in categories


def test_phone_dot_separator():
    entities = extract_entities("Call 877.877.0987 today.")
    phones = [e for e in entities if e["category"] == "phone"]
    assert len(phones) == 1


def test_phone_bare_digits():
    entities = extract_entities("Call 8778770987 today.")
    phones = [e for e in entities if e["category"] == "phone"]
    assert len(phones) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py -x -v 2>&1 | tail -20`
Expected: FAIL with ImportError (extract_entities not defined)

- [ ] **Step 3: Write implementation**

Create `stages/summary_generator.py`:

```python
"""Summary report generator — extract entities from recovered segments and build a PDF index.

Logic reference: docs/superpowers/specs/2026-03-13-summary-pdf-design.md
"""

import re

# Stopwords for people extraction — common legal/geographic phrases
_PEOPLE_STOPWORDS = {
    "united states", "new york", "district court", "southern district",
    "northern district", "eastern district", "western district",
    "palm beach", "this report", "the united", "the honorable",
    "air france", "royal air", "american express", "flight information",
    "other information", "hotel information", "travel service",
    "privacy statement", "entry and exit",
}

_ORG_SUFFIXES = re.compile(
    r'\b(?:LLC|Inc|Corp|Department|Agency|Bureau|Office|Court|FBI|DOJ|OIG|CIA'
    r'|Police|Division|Commission|Authority|Association)\b',
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
        # Grab up to 5 preceding words to form the org name
        prefix_start = max(0, m.start() - 60)
        prefix = text[prefix_start:m.end()]
        # Find the org name: capitalized words leading into the suffix
        org_match = re.search(
            r'(?:(?:the|of|for|and|in)\s+)*(?:[A-Z][a-zA-Z]*\s+)*' + re.escape(m.group()),
            prefix, re.IGNORECASE
        )
        if org_match:
            org_name = org_match.group().strip()
            # Strip leading articles
            org_name = re.sub(r'^(?:the|of|for)\s+', '', org_name, flags=re.IGNORECASE).strip()
            abs_start = prefix_start + org_match.start()
            abs_end = prefix_start + org_match.end()
            if len(org_name) > 3 and not _is_consumed(abs_start, abs_end):
                entities.append({"text": org_name, "category": "organization"})
                consumed.add((abs_start, abs_end))

    # 5. People names
    for m in _NAME_RE.finditer(text):
        name = m.group(1) or m.group(2)
        if not name or _is_consumed(m.start(), m.end()):
            continue
        # Check stopwords
        if name.lower() in _PEOPLE_STOPWORDS:
            continue
        # Skip single-word matches (regex requires 2+, but verify)
        if len(name.split()) < 2:
            continue
        entities.append({"text": name, "category": "people"})
        consumed.add((m.start(), m.end()))

    return entities
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py -x -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add stages/summary_generator.py tests/stages/test_summary_generator.py
git commit -m "feat: entity extraction for summary report (people, emails, phones, cases, orgs)"
```

---

### Task 3: Entity Aggregation and Deduplication

**Files:**
- Modify: `stages/summary_generator.py` (add function)
- Test: `tests/stages/test_summary_generator.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/stages/test_summary_generator.py`:

```python
from stages.summary_generator import aggregate_entities


def test_aggregate_deduplicates_case_insensitive():
    raw = [
        {"text": "SARAH KELLEN", "category": "people", "group_id": 1},
        {"text": "Sarah Kellen", "category": "people", "group_id": 2},
    ]
    result = aggregate_entities(raw)
    people = [e for e in result if e["category"] == "people"]
    assert len(people) == 1
    assert people[0]["count"] == 2
    assert set(people[0]["group_ids"]) == {1, 2}


def test_aggregate_deduplicates_phones_by_digits():
    raw = [
        {"text": "(877) 877-0987", "category": "phone", "group_id": 1},
        {"text": "877-877-0987", "category": "phone", "group_id": 2},
    ]
    result = aggregate_entities(raw)
    phones = [e for e in result if e["category"] == "phone"]
    assert len(phones) == 1
    assert phones[0]["count"] == 2


def test_aggregate_deduplicates_emails_by_lowercase():
    raw = [
        {"text": "Sarah@Example.com", "category": "email", "group_id": 1},
        {"text": "sarah@example.com", "category": "email", "group_id": 2},
    ]
    result = aggregate_entities(raw)
    emails = [e for e in result if e["category"] == "email"]
    assert len(emails) == 1
    assert emails[0]["count"] == 2


def test_aggregate_sorts_by_frequency():
    raw = [
        {"text": "SARAH KELLEN", "category": "people", "group_id": 1},
        {"text": "SARAH KELLEN", "category": "people", "group_id": 2},
        {"text": "SARAH KELLEN", "category": "people", "group_id": 3},
        {"text": "BILL CLINTON", "category": "people", "group_id": 4},
    ]
    result = aggregate_entities(raw)
    people = [e for e in result if e["category"] == "people"]
    assert people[0]["text"] == "SARAH KELLEN"
    assert people[0]["count"] == 3


def test_aggregate_deduplicates_orgs_by_lowercase():
    raw = [
        {"text": "Department of Justice", "category": "organization", "group_id": 1},
        {"text": "department of justice", "category": "organization", "group_id": 2},
    ]
    result = aggregate_entities(raw)
    orgs = [e for e in result if e["category"] == "organization"]
    assert len(orgs) == 1
    assert orgs[0]["count"] == 2


def test_aggregate_case_number_exact_dedup():
    raw = [
        {"text": "72-MM-113327", "category": "case_number", "group_id": 1},
        {"text": "72-MM-113327", "category": "case_number", "group_id": 2},
    ]
    result = aggregate_entities(raw)
    cases = [e for e in result if e["category"] == "case_number"]
    assert len(cases) == 1
    assert cases[0]["count"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py::test_aggregate_deduplicates_case_insensitive -x -v 2>&1 | tail -10`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

Add to `stages/summary_generator.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py -x -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add stages/summary_generator.py tests/stages/test_summary_generator.py
git commit -m "feat: entity aggregation with case/format-aware deduplication"
```

---

## Chunk 2: PDF Generation and CLI

### Task 4: PDF Builder

**Files:**
- Modify: `stages/summary_generator.py` (add PDF functions)
- Test: `tests/stages/test_summary_generator.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/stages/test_summary_generator.py`:

```python
import fitz
from stages.summary_generator import generate_summary_pdf


def test_generate_summary_pdf_creates_file(conn, tmp_path):
    output_dir = str(tmp_path / "output")
    seed_group(conn,
               group_docs=[("doc-a", "base text [REDACTED] here"), ("doc-b", "donor")],
               merged_text="base text SARAH KELLEN here",
               recovered_count=1,
               segments=[{"text": "SARAH KELLEN", "source_doc_id": "doc-b", "stage": "merge"}])

    path = generate_summary_pdf(conn, output_dir)
    assert path is not None
    from pathlib import Path
    assert Path(path).exists()
    assert path.endswith("summary_report.pdf")


def test_generate_summary_pdf_contains_entity(conn, tmp_path):
    output_dir = str(tmp_path / "output")
    seed_group(conn,
               group_docs=[("doc-a", "base"), ("doc-b", "donor")],
               merged_text="merged",
               recovered_count=1,
               segments=[{"text": "SARAH KELLEN", "source_doc_id": "doc-b", "stage": "merge"}])

    path = generate_summary_pdf(conn, output_dir)
    doc = fitz.open(path)
    full_text = "".join(page.get_text() for page in doc)
    assert "SARAH KELLEN" in full_text
    assert "Unobfuscator Summary Report" in full_text
    doc.close()


def test_generate_summary_pdf_empty_corpus(conn, tmp_path):
    output_dir = str(tmp_path / "output")
    path = generate_summary_pdf(conn, output_dir)
    # Should still generate a PDF with zero stats
    assert path is not None
    doc = fitz.open(path)
    full_text = "".join(page.get_text() for page in doc)
    assert "0 recovered" in full_text or "0 segments" in full_text
    doc.close()


def test_generate_summary_pdf_has_stats(conn, tmp_path):
    output_dir = str(tmp_path / "output")
    seed_group(conn,
               group_docs=[("doc-a", "base"), ("doc-b", "donor")],
               merged_text="merged",
               recovered_count=2,
               segments=[
                   {"text": "SARAH KELLEN", "source_doc_id": "doc-b", "stage": "merge"},
                   {"text": "test@example.com", "source_doc_id": "doc-b", "stage": "merge"},
               ])

    path = generate_summary_pdf(conn, output_dir)
    doc = fitz.open(path)
    full_text = "".join(page.get_text() for page in doc)
    # Should contain stats
    assert "People" in full_text or "Email" in full_text
    doc.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py::test_generate_summary_pdf_creates_file -x -v 2>&1 | tail -10`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

Add to `stages/summary_generator.py`:

```python
import json
from datetime import date
from pathlib import Path
import fitz

from core.db import get_all_recovery_groups, get_documents_by_ids

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/stages/test_summary_generator.py -x -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add stages/summary_generator.py tests/stages/test_summary_generator.py
git commit -m "feat: summary PDF generator with title, top findings, and category sections"
```

---

### Task 5: CLI Command

**Files:**
- Modify: `unobfuscator.py` (add `summary` command)

- [ ] **Step 1: Add the CLI command**

Add import at top of `unobfuscator.py` alongside other stage imports:

```python
from stages.summary_generator import generate_summary_pdf
```

Add new command before `if __name__ == "__main__":`:

```python
@cli.command()
@click.pass_context
def summary(ctx):
    """Generate a summary PDF report of all recovered entities."""
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    output_dir = cfg_get(cfg, "output_dir", default="./output")
    if not os.path.exists(db_path):
        console.print("[red]Database not found. Run 'unobfuscator start' first.[/red]")
        return
    conn = get_connection(db_path)
    console.print("[dim]Generating summary report...[/dim]")
    path = generate_summary_pdf(conn, output_dir)
    console.print(f"[green]Summary report written to {path}[/green]")
```

- [ ] **Step 2: Verify CLI works**

Run: `./venv/bin/python unobfuscator.py summary 2>&1`
Expected: "Summary report written to ./output/summary_report.pdf"

- [ ] **Step 3: Run full test suite**

Run: `./venv/bin/python -m pytest tests/ -x -q 2>&1`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add unobfuscator.py
git commit -m "feat: add ./unob summary CLI command for summary PDF report"
```

---

### Task 6: Verify with Real Data

- [ ] **Step 1: Generate the real summary**

Run: `./venv/bin/python unobfuscator.py summary`

- [ ] **Step 2: Open and review the PDF**

Check:
- Title page has correct stats
- Top 20 findings are reasonable
- Category sections have entities
- Links are clickable
- No obvious false positives in People category

- [ ] **Step 3: Tune regex patterns if needed**

Based on review, adjust `_PEOPLE_STOPWORDS` or extraction patterns. Re-run and verify.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix: tune summary entity extraction based on real data review"
```
