import pytest
import json
from core.db import (
    init_db, get_connection, upsert_document, create_match_group,
    add_group_member, upsert_merge_result
)
from stages.merger import (
    find_redaction_positions, extract_anchors,
    find_text_between_anchors, merge_group, run_merger
)

REDACTION_MARKERS = ["[REDACTED]", "[b(6)]"]

BASE_TEXT = (
    "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
    "Date: 2002-03-10\nSubject: Guest list\n\n"
    "The attendees included [REDACTED] and Prince Andrew. "
    "The flight departed from [REDACTED] at 9am and arrived at the island."
)

DONOR_TEXT_A = (
    "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
    "Date: 2002-03-10\nSubject: Guest list\n\n"
    "The attendees included Bill Clinton and Prince Andrew. "
    "The flight departed from [REDACTED] at 9am and arrived at the island."
)

DONOR_TEXT_B = (
    "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
    "Date: 2002-03-10\nSubject: Guest list\n\n"
    "The attendees included [REDACTED] and Prince Andrew. "
    "The flight departed from Palm Beach at 9am and arrived at the island."
)


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def seed_doc(conn, doc_id, text):
    from core.db import upsert_document
    upsert_document(conn, {
        "id": doc_id, "source": "doj", "release_batch": "VOL00001",
        "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
        "size_bytes": 500, "description": "", "extracted_text": text
    })


# --- Core utility functions ---

def test_find_redaction_positions_returns_all_markers():
    positions = find_redaction_positions(BASE_TEXT, REDACTION_MARKERS)
    assert len(positions) == 2
    for pos, marker in positions:
        assert BASE_TEXT[pos:pos + len(marker)] == marker


def test_find_redaction_positions_returns_empty_for_clean_text():
    positions = find_redaction_positions("This text has no redactions at all.", REDACTION_MARKERS)
    assert positions == []


def test_extract_anchors_returns_text_around_position():
    text = "before context HERE after context"
    pos = text.index("HERE")
    marker_len = len("HERE")
    left, right = extract_anchors(text, pos, marker_len, length=20)
    assert "before" in left
    assert "after" in right


def test_extract_anchors_handles_start_of_text():
    text = "HERE after context"
    marker_len = len("HERE")
    left, right = extract_anchors(text, 0, marker_len, length=10)
    assert left == ""
    assert "after" in right


def test_find_text_between_anchors_recovers_redacted_content():
    left_anchor = "attendees included"
    right_anchor = "and Prince Andrew"
    recovered = find_text_between_anchors(DONOR_TEXT_A, left_anchor, right_anchor)
    assert recovered is not None
    assert "Bill Clinton" in recovered


def test_find_text_between_anchors_returns_none_when_anchors_missing():
    result = find_text_between_anchors(
        "Completely different text here.",
        "attendees included",
        "and Prince Andrew"
    )
    assert result is None


def test_find_between_exact_rejects_ambiguous_match():
    """When anchor pair matches multiple positions in donor, return None."""
    donor = "Hello dear friend World. Some filler. Hello dear enemy World."
    result = find_text_between_anchors(donor, "Hello", "World")
    assert result is None


def test_find_between_exact_accepts_unique_match():
    """When anchor pair matches exactly once, return the text between them."""
    donor = "Hello dear friend World. Some filler text here."
    result = find_text_between_anchors(donor, "Hello", "World")
    assert result == "dear friend"


def test_find_between_exact_rejects_ambiguous_right_anchor_when_left_empty():
    """When left_anchor is empty and right_anchor matches multiple times, return None."""
    donor = "First section MARKER middle section MARKER end"
    result = find_text_between_anchors(donor, "", "MARKER")
    assert result is None


# --- Full merge ---

def test_merge_group_fills_redactions_from_donors(conn):
    seed_doc(conn, 1, BASE_TEXT)
    seed_doc(conn, 2, DONOR_TEXT_A)
    seed_doc(conn, 3, DONOR_TEXT_B)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    add_group_member(conn, g, 3, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=20)
    assert result["recovered_count"] == 2
    assert "Bill Clinton" in result["merged_text"]
    assert "Palm Beach" in result["merged_text"]
    assert "[REDACTED]" not in result["merged_text"]


def test_merge_group_returns_zero_recovered_when_no_donors_help(conn):
    seed_doc(conn, 1, BASE_TEXT)
    seed_doc(conn, 2, "Totally unrelated content that shares nothing.")
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.1)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=20)
    assert result["recovered_count"] == 0
    assert "[REDACTED]" in result["merged_text"]


def test_run_merger_stores_results_and_marks_group_merged(conn):
    seed_doc(conn, 1, BASE_TEXT)
    seed_doc(conn, 2, DONOR_TEXT_A)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    run_merger(conn, REDACTION_MARKERS, anchor_length=20)
    conn.commit()

    row = conn.execute(
        "SELECT merged, recovered_count FROM match_groups "
        "JOIN merge_results USING (group_id) WHERE group_id = ?", (g,)
    ).fetchone()
    assert row is not None
    assert row["merged"] == 1
    assert row["recovered_count"] >= 1
