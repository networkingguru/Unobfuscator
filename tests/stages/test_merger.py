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


def test_merge_group_rejects_table_column_degenerate_anchors(conn):
    """Redactions in table columns with only pipe separators should not false-recover."""
    base_text = (
        "| Department | Contact |\n"
        "| East Coast | [REDACTED] |\n"
        "| West Coast | [REDACTED] |\n"
        "| Customer Service | [REDACTED] |\n"
    )
    donor_text = (
        "| Department | Contact |\n"
        "| East Coast | 555-0101 |\n"
        "| West Coast | 555-0102 |\n"
        "| Customer Service | 555-0103 |\n"
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    recovered_texts = [s["text"] for s in result["recovered_segments"]]
    # No single text should appear more than once
    from collections import Counter
    counts = Counter(recovered_texts)
    for text, count in counts.items():
        assert count == 1, f"Text {repr(text)} recovered {count} times — likely false recovery"


def test_merge_group_rejects_consecutive_redactions_real_pattern(conn):
    """Reproduce the group 7082 pattern: 20 consecutive [REDACTED] with >>>>> separators."""
    lines = [">>>>>> [REDACTED]"] * 20
    base_text = (
        "From: Michelle\nEmployment Counselor\nRegal Domestics, Inc.\n"
        + "\n".join(lines)
        + "\n>>>>>>>\n>>>>>>>> On Jan 27, 2016, at 1:05 PM, Michelle\n>> wrote:"
    )
    donor_text = (
        "From: Michelle\nEmployment Counselor\nRegal Domestics, Inc.\n"
        ">>>>>> 123 Main St\n>>>>>> Suite 100\n>>>>>> New York, NY\n"
        ">>>>>> (555) 111-2222\n>>>>>> michelle@regal.com\n"
        ">>>>>> Mon-Fri 9am-5pm\n>>>>>> Fax: (555) 111-3333\n"
        ">>>>>> Emergency: (555) 111-4444\n>>>>>> Regional Office\n"
        ">>>>>> PO Box 12345\n>>>>>> New York, NY 10001\n"
        ">>>>>> Licensed & Bonded\n>>>>>> Est. 1995\n"
        ">>>>>> www.regaldomestics.com\n>>>>>> info@regal.com\n"
        ">>>>>> NAICS: 561311\n>>>>>> EIN: 12-3456789\n"
        ">>>>>> Member: APSA\n>>>>>> Accredited: BBB A+\n"
        ">>>>>>>\n>>>>>>>> On Jan 27, 2016, at 1:05 PM, Michelle\n>> wrote:"
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    recovered_texts = [s["text"] for s in result["recovered_segments"]]
    from collections import Counter
    counts = Counter(recovered_texts)
    for text, count in counts.items():
        assert count <= 1, f"Text {repr(text[:60])} recovered {count} times — false recovery"
    assert result["recovered_count"] <= 2


def test_find_between_accepts_unique_left_with_repeated_right():
    """Left anchor is unique; right anchor repeats. First right match is taken."""
    donor = "Hello dear friend World. And also World."
    result = find_text_between_anchors(donor, "Hello", "World")
    assert result == "dear friend"


def test_merge_group_allows_legitimate_repeated_name_recovery(conn):
    """A name redacted in multiple places with unique anchors should recover each time."""
    base_text = (
        "The meeting was attended by [REDACTED] and the lawyer. "
        "Later that evening, [REDACTED] flew to Palm Beach."
    )
    donor_text = (
        "The meeting was attended by Bill Clinton and the lawyer. "
        "Later that evening, Bill Clinton flew to Palm Beach."
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Reverse merge uses unredacted doc as base — content is there directly
    assert result["merged_text"].count("Bill Clinton") == 2
    assert "[REDACTED]" not in result["merged_text"]


def test_confidence_field_marks_repeated_recoveries_as_questionable(conn):
    """Recoveries of the same text 3+ times get confidence='questionable'."""
    # 4 redactions, each with unique left anchor but donor has same name everywhere
    base_text = (
        "Meeting with [REDACTED] on Monday. "
        "Lunch with [REDACTED] on Tuesday. "
        "Call from [REDACTED] on Wednesday. "
        "Dinner with [REDACTED] on Thursday."
    )
    donor_text = (
        "Meeting with John Smith on Monday. "
        "Lunch with John Smith on Tuesday. "
        "Call from John Smith on Wednesday. "
        "Dinner with John Smith on Thursday."
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # All 4 recover "John Smith" — same text 4x triggers questionable
    for seg in result["recovered_segments"]:
        assert seg["confidence"] == "questionable"


def test_confidence_field_high_for_unique_recoveries(conn):
    """Recoveries of distinct texts get confidence='high'."""
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
    for seg in result["recovered_segments"]:
        assert seg["confidence"] == "high"


def test_is_real_recovery_rejects_redaction_descriptions(conn):
    """Redaction descriptions like 'blacked out' should not be treated as recoveries."""
    base_text = "The document contained [REDACTED] in the margin."
    donor_text = "The document contained [blackened box] in the margin."
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] == 0


def test_reverse_merge_uses_unredacted_doc_as_base(conn):
    """When a donor has 0 redactions, it becomes the base; merged text has no markers."""
    redacted_text = (
        "The attendees included [REDACTED] and Prince Andrew. "
        "The flight departed from [REDACTED] at 9am. "
        "They arrived at [REDACTED] by noon."
    )
    unredacted_text = (
        "The attendees included Bill Clinton and Prince Andrew. "
        "The flight departed from Palm Beach at 9am. "
        "They arrived at Little St. James by noon."
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, unredacted_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.95)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert "[REDACTED]" not in result["merged_text"]
    assert "Bill Clinton" in result["merged_text"]
    assert "Palm Beach" in result["merged_text"]
    assert "Little St. James" in result["merged_text"]
    assert result["total_redacted"] == 3


def test_reverse_merge_recovers_when_normal_merge_fails(conn):
    """Docs with structural differences: using unredacted as base avoids anchor failures."""
    redacted_text = (
        "(U) Interview of [REDACTED]\n"
        "(U) Interview of [REDACTED]\n"
        "(U) Proffer of [REDACTED]\n"
        "EFTA02730496\n--- PAGE BREAK ---\n"
    )
    unredacted_text = (
        "(U) Interview of Miguel Monge\n"
        "(U) Interview of Darrius Dupree\n"
        "(U) Interview of John MARRUGO\n"
        "(U) Proffer of Jason Mojica\n"
        "(U) Proffer of Christian Perez\n"
        "(U) Proffer of Freddy Caraballo\n"
        "EFTA02730942\n--- PAGE BREAK ---\n"
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, unredacted_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.85)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert "[REDACTED]" not in result["merged_text"]
    assert "Jason Mojica" in result["merged_text"]
    assert result["total_redacted"] == 3


def test_reverse_merge_skips_when_length_mismatch(conn):
    """A short unredacted doc should not become base for a much longer redacted doc."""
    long_redacted = (
        "Section 1: The investigation found [REDACTED] at the scene.\n" * 20
    )
    short_unredacted = "Cover page: Case File 2005-0042\n"
    seed_doc(conn, 1, long_redacted)
    seed_doc(conn, 2, short_unredacted)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.3)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["total_redacted"] == 20
    assert "Cover page" not in result["merged_text"][:50]


def test_wider_anchors_recover_when_short_anchors_are_ambiguous(conn):
    """50-char anchors are identical for two redactions; wider context resolves them."""
    shared_pre = "the private counsel representing the individual "  # 48 chars
    shared_suf = " during the confidential deposition proceedings. "  # 50 chars
    base_text = (
        "In January in the downtown office, " + shared_pre + "[REDACTED]" + shared_suf
        + "In February in the uptown office, " + shared_pre + "[REDACTED]" + shared_suf
    )
    # Donor keeps a [REDACTED] elsewhere so reverse-merge doesn't kick in
    donor_text = (
        "In January in the downtown office, " + shared_pre + "Alan Dershowitz" + shared_suf
        + "In February in the uptown office, " + shared_pre + "Leslie Wexner" + shared_suf
        + "In March [REDACTED] attended a gala."
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] == 2
    assert "Alan Dershowitz" in result["merged_text"]
    assert "Leslie Wexner" in result["merged_text"]


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
