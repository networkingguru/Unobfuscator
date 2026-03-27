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


def test_merge_group_skips_redactions_with_degenerate_anchors(conn):
    """Consecutive redactions with only quote markers between them should not be recovered."""
    base_text = (
        "From: sender@example.com\n"
        ">>>>>> Employment Counselor\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>>>\n"
        ">>>>>>>> On Jan 27, 2016, Michelle wrote:"
    )
    donor_text = (
        "From: sender@example.com\n"
        ">>>>>> Employment Counselor\n"
        ">>>>>> Regal Domestics, Inc.\n"
        ">>>>>> 123 Main Street\n"
        ">>>>>> New York, NY 10001\n"
        ">>>>>> (555) 123-4567\n"
        ">>>>>> michelle@regal.com\n"
        ">>>>>>>\n"
        ">>>>>>>> On Jan 27, 2016, Michelle wrote:"
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] <= 1


def test_anchor_quality_floor_boundary(conn):
    """Anchors with exactly 7 combined alphanumeric chars are skipped; 8 pass."""
    # Both anchors truncated to short content by adjacent redaction markers
    # Left anchor after truncation: "Abcdefg" (7 alpha), right anchor: "" (truncated by [REDACTED])
    base_7 = "[REDACTED] Abcdefg [REDACTED] [REDACTED]"
    # Left anchor after truncation: "Abcdefgh" (8 alpha), right anchor: "" (truncated by [REDACTED])
    base_8 = "[REDACTED] Abcdefgh [REDACTED] [REDACTED]"
    donor_7 = "[REDACTED] Abcdefg RECOVERED [REDACTED]"
    donor_8 = "[REDACTED] Abcdefgh RECOVERED"

    seed_doc(conn, 1, base_7)
    seed_doc(conn, 2, donor_7)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()
    result_7 = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # base_7 has 3 redactions. The middle one has left="Abcdefg" (7 alpha) and right="" (truncated).
    # Combined = 7 alpha chars < 8, so should be skipped.
    assert result_7["recovered_count"] == 0, "7 alphanumeric chars should be below quality floor"

    seed_doc(conn, 3, base_8)
    seed_doc(conn, 4, donor_8)
    conn.commit()
    g2 = create_match_group(conn)
    add_group_member(conn, g2, 3, 1.0)
    add_group_member(conn, g2, 4, 0.9)
    conn.commit()
    result_8 = merge_group(conn, g2, REDACTION_MARKERS, anchor_length=50)
    # base_8 middle redaction has left="Abcdefgh" (8 alpha) and right="" (truncated).
    # Combined = 8 alpha chars >= 8, so should attempt recovery.
    assert result_8["recovered_count"] >= 1, "8 alphanumeric chars should pass quality floor"


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
