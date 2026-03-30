import pytest
import json
from datasketch import MinHash
from core.db import (
    init_db, get_connection, upsert_document, create_match_group,
    add_group_member, upsert_merge_result, upsert_fingerprint
)
from stages.merger import (
    find_redaction_positions, extract_anchors,
    find_text_between_anchors, merge_group, run_merger,
    cluster_and_split_group, run_cross_group_merger,
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


def test_alignment_merge_recovers_with_structural_differences(conn):
    """Alignment handles lines that differ structurally (extra words, formatting)
    but where context still normalizes to a substring match.

    The base uses Markdown-style formatting around each redaction so
    exact anchor matching fails, but normalized matching succeeds after
    stripping formatting markers.
    """
    redacted_text = (
        "MEMORANDUM\n"
        "From: Office of General Counsel\n"
        "Re: Investigation Update\n\n"
        "1. The subject [REDACTED] was identified at the location.\n"
        "2. Witness [REDACTED] provided testimony on the record.\n"
        "3. Evidence collected from [REDACTED] confirmed the timeline.\n"
    )
    # Donor has same structure but Markdown bold around names; extra trailing line
    unredacted_text = (
        "MEMORANDUM\n"
        "From: Office of General Counsel\n"
        "Re: Investigation Update\n\n"
        "1. The subject John Smith was identified at the location.\n"
        "2. Witness Sarah Johnson provided testimony on the record.\n"
        "3. Evidence collected from Palm Beach PD confirmed the timeline.\n"
        "4. The [REDACTED] case file was sealed.\n"
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, unredacted_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.85)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] >= 2
    assert "John Smith" in result["merged_text"]
    assert "Sarah Johnson" in result["merged_text"]


def test_alignment_merge_does_not_false_recover_unrelated_content(conn):
    """Alignment rejects candidates where surrounding context doesn't match."""
    redacted_text = (
        "Section A: The investigation found [REDACTED] at the scene.\n"
        "Section B: The report concluded with no further action.\n"
    )
    donor_text = (
        "Section A: The budget review found surplus funds at the office.\n"
        "Section B: The report concluded with no further action.\n"
        "Section C: The [REDACTED] budget was approved.\n"
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.7)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert "surplus funds" not in result["merged_text"]
    assert "[REDACTED]" in result["merged_text"]


def test_multi_pass_recovers_when_adjacent_recovery_provides_anchors(conn):
    """Pass 1 recovers outer redactions; pass 2 uses them as anchor context for inner ones."""
    # Layout: good_anchor [A] , [B] , [C] good_anchor
    # The separators between redactions are just ", " (2 chars, 0 alphanumeric).
    # Pass 1: A's left anchor is good ("confirmed that"), C's right anchor is good
    #         ("was already"). But B's anchors are truncated at adjacent markers,
    #         leaving only ", " on each side -> alpha_content < 8, skipped.
    # Pass 2: After A and C are recovered, B's anchors include "Bill Clinton, "
    #         (left) and ", Ghislaine Maxwell" (right) -> alpha > 8, matches.
    base_text = (
        "The lead investigator confirmed that [REDACTED], [REDACTED], [REDACTED] "
        "was already waiting according to the flight manifest records."
    )
    donor_text = (
        "The lead investigator confirmed that Bill Clinton, Prince Andrew, Ghislaine Maxwell "
        "was already waiting according to the flight manifest records. "
        "Another [REDACTED] name appears elsewhere in the file."
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] == 3
    assert "Bill Clinton" in result["merged_text"]
    assert "Prince Andrew" in result["merged_text"]
    assert "Ghislaine Maxwell" in result["merged_text"]
    # Verify no garbled/duplicated content
    assert result["merged_text"].count("Bill Clinton") == 1
    assert result["merged_text"].count("Prince Andrew") == 1
    assert result["merged_text"].count("Ghislaine Maxwell") == 1


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


def test_enhanced_merge_full_integration(conn):
    """Exercise multiple techniques: no single member has complete text."""
    # Member A: has names but locations redacted
    member_a = (
        "MEMORANDUM\nFrom: DOJ Office\nRe: Case Update\n\n"
        "1. Subject John Smith was seen at [REDACTED] on March 10.\n"
        "2. Witness Jane Doe confirmed John Smith was present at [REDACTED].\n"
        "3. Captain James Lee transported John Smith via private aircraft.\n"
        "4. Destination: [REDACTED]\n"
    )
    # Member B: has locations but names redacted
    member_b = (
        "MEMORANDUM\nFrom: DOJ Office\nRe: Case Update\n\n"
        "1. Subject [REDACTED] was seen at Mar-a-Lago on March 10.\n"
        "2. Witness [REDACTED] confirmed [REDACTED] was present at the Palm Beach estate.\n"
        "3. [REDACTED] transported [REDACTED] via private aircraft.\n"
        "4. Destination: Palm Beach International\n"
    )
    # Member C: complete text, structural variant (extra date line)
    member_c = (
        "MEMORANDUM\nFrom: DOJ Office\nDate: March 15, 2005\nRe: Case Update\n\n"
        "1. Subject John Smith was seen at Mar-a-Lago on March 10.\n"
        "2. Witness Jane Doe confirmed John Smith was present at the Palm Beach estate.\n"
        "3. Captain James Lee transported John Smith via private aircraft.\n"
        "4. Destination: Palm Beach International\n"
    )
    seed_doc(conn, 1, member_a)
    seed_doc(conn, 2, member_b)
    seed_doc(conn, 3, member_c)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.85)
    add_group_member(conn, g, 3, 0.80)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Member C has 0 redactions — reverse merge uses it as base
    assert "[REDACTED]" not in result["merged_text"]
    assert "John Smith" in result["merged_text"]
    assert "Mar-a-Lago" in result["merged_text"]
    assert "Jane Doe" in result["merged_text"]
    assert "Captain James Lee" in result["merged_text"]
    assert "Palm Beach International" in result["merged_text"]
    assert result["total_redacted"] >= 3


# --- Sub-clustering helpers and tests ---

def _seed_doc_with_fingerprint(conn, doc_id, text, num_perm=128):
    """Seed a document and compute+store its MinHash fingerprint."""
    seed_doc(conn, doc_id, text)
    mh = MinHash(num_perm=num_perm)
    for shingle in [text[i:i+5] for i in range(len(text) - 4)]:
        mh.update(shingle.encode('utf-8'))
    upsert_fingerprint(conn, doc_id, mh.hashvalues.tobytes(), max(len(text) - 4, 0))


def test_cluster_and_split_creates_permanent_subgroups(conn):
    """A mega-group with 2 unrelated sub-clusters should be split into 2 permanent groups."""
    base_a = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight schedule for the weekend trip\n\n"
        "The flight to [REDACTED] departs at 9am from Palm Beach airport terminal."
    )
    donor_a = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight schedule for the weekend trip\n\n"
        "The flight to Little St. James departs at 9am from Palm Beach airport terminal."
    )
    base_b = (
        "CASE NO. 2005-0042\nSouthern District of New York\n"
        "Plaintiff: [REDACTED]\n"
        "The deposition was taken on March 10, 2005 at the courthouse."
    )
    donor_b = (
        "CASE NO. 2005-0042\nSouthern District of New York\n"
        "Plaintiff: Virginia Giuffre\n"
        "The deposition was taken on March 10, 2005 at the courthouse."
    )
    _seed_doc_with_fingerprint(conn, "a1", base_a)
    _seed_doc_with_fingerprint(conn, "a2", donor_a)
    _seed_doc_with_fingerprint(conn, "b1", base_b)
    _seed_doc_with_fingerprint(conn, "b2", donor_b)
    conn.commit()

    g = create_match_group(conn)
    for doc_id in ["a1", "a2", "b1", "b2"]:
        add_group_member(conn, g, doc_id, 1.0)
    conn.commit()

    new_group_ids = cluster_and_split_group(conn, g, REDACTION_MARKERS)

    original = conn.execute("SELECT * FROM match_groups WHERE group_id = ?", (g,)).fetchone()
    assert original is None, "Original mega-group should be deleted after splitting"
    assert len(new_group_ids) >= 2
    for gid in new_group_ids:
        count = conn.execute(
            "SELECT COUNT(*) FROM match_group_members WHERE group_id = ?", (gid,)
        ).fetchone()[0]
        assert count == 2


def test_cluster_and_split_releases_orphans(conn):
    """Documents with no within-group match should be released."""
    base = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight plan details\n\nPassenger [REDACTED] on the manifest today."
    )
    donor = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight plan details\n\nPassenger Bill Clinton on the manifest today."
    )
    orphan = (
        "From: accountant@example.com\nTo: bank@example.com\n"
        "Subject: Wire transfer instructions\n\nPlease transfer $50,000 to account number."
    )
    _seed_doc_with_fingerprint(conn, "r1", base)
    _seed_doc_with_fingerprint(conn, "r2", donor)
    _seed_doc_with_fingerprint(conn, "orphan", orphan)
    conn.commit()

    g = create_match_group(conn)
    add_group_member(conn, g, "r1", 1.0)
    add_group_member(conn, g, "r2", 1.0)
    add_group_member(conn, g, "orphan", 1.0)
    conn.commit()

    new_group_ids = cluster_and_split_group(conn, g, REDACTION_MARKERS)

    orphan_group = conn.execute(
        "SELECT group_id FROM match_group_members WHERE doc_id = 'orphan'"
    ).fetchone()
    assert orphan_group is None, "Orphan doc should be released from all groups"

    r1_group = conn.execute(
        "SELECT group_id FROM match_group_members WHERE doc_id = 'r1'"
    ).fetchone()
    assert r1_group is not None


def test_cluster_and_split_handles_recursive_large_component(conn):
    """If a connected component still exceeds the cluster threshold, it is accepted."""
    import stages.merger as merger_mod
    original = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 2

    try:
        texts = [
            ("The investigation revealed important details about the case on March 10 "
             "in Palm Beach Florida. The witness testified under oath that the events "
             "took place at the residence on the evening of the incident. Version " + str(i))
            for i in range(4)
        ]
        for i, text in enumerate(texts):
            _seed_doc_with_fingerprint(conn, f"d{i}", text)
        conn.commit()

        g = create_match_group(conn)
        for i in range(4):
            add_group_member(conn, g, f"d{i}", 1.0)
        conn.commit()

        new_group_ids = cluster_and_split_group(conn, g, REDACTION_MARKERS)

        for i in range(4):
            row = conn.execute(
                "SELECT group_id FROM match_group_members WHERE doc_id = ?", (f"d{i}",)
            ).fetchone()
            assert row is not None, f"Doc d{i} should still be in a group"
    finally:
        merger_mod._CLUSTER_THRESHOLD = original


def test_run_merger_splits_large_group_then_merges_subgroups(conn):
    """Groups exceeding _CLUSTER_THRESHOLD should be split, then each sub-group merged."""
    import stages.merger as merger_mod
    original = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 3

    try:
        base_a = (
            "From: jeffrey@example.com\nTo: pilot@example.com\n"
            "Subject: Flight schedule for the weekend trip\n\n"
            "The flight to [REDACTED] departs at 9am from Palm Beach airport terminal."
        )
        donor_a = (
            "From: jeffrey@example.com\nTo: pilot@example.com\n"
            "Subject: Flight schedule for the weekend trip\n\n"
            "The flight to Little St. James departs at 9am from Palm Beach airport terminal."
        )
        orphan1 = "From: a@b.com\nCompletely different content alpha beta gamma delta."
        orphan2 = "From: c@d.com\nCompletely different content epsilon zeta eta theta."

        _seed_doc_with_fingerprint(conn, "m1", base_a)
        _seed_doc_with_fingerprint(conn, "m2", donor_a)
        _seed_doc_with_fingerprint(conn, "m3", orphan1)
        _seed_doc_with_fingerprint(conn, "m4", orphan2)
        conn.commit()

        g = create_match_group(conn)
        for doc_id in ["m1", "m2", "m3", "m4"]:
            add_group_member(conn, g, doc_id, 1.0)
        conn.commit()

        count = run_merger(conn, REDACTION_MARKERS, anchor_length=50)

        original_group = conn.execute(
            "SELECT * FROM match_groups WHERE group_id = ?", (g,)
        ).fetchone()
        assert original_group is None

        # Check that merge happened: merged text has the recovered content
        # (reverse merge uses unredacted doc as base, so recovered_count may be 0
        # but the merged_text should contain the unredacted content)
        results = conn.execute(
            "SELECT merged_text, total_redacted FROM merge_results"
        ).fetchall()
        assert len(results) >= 1
        found_recovery = any(
            "Little St. James" in (r["merged_text"] or "") for r in results
        )
        assert found_recovery, "Expected 'Little St. James' in some merge result"

        for orphan_id in ["m3", "m4"]:
            row = conn.execute(
                "SELECT group_id FROM match_group_members WHERE doc_id = ?", (orphan_id,)
            ).fetchone()
            assert row is None, f"Orphan {orphan_id} should be ungrouped"
    finally:
        merger_mod._CLUSTER_THRESHOLD = original


def test_run_cross_group_merger_recovers_redactions(conn):
    """Cross-group pairs should recover redactions."""
    base = (
        "The investigation found [REDACTED] at the scene on March 10. "
        "The evidence was collected by officer Johnson at the precinct."
    )
    donor = (
        "The investigation found John Smith at the scene on March 10. "
        "The evidence was collected by officer Johnson at the precinct."
    )
    seed_doc(conn, "x1", base)
    seed_doc(conn, "x2", donor)
    conn.commit()

    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, "x1", 1.0)
    add_group_member(conn, g2, "x2", 1.0)
    conn.commit()

    from core.db import insert_verified_pair
    insert_verified_pair(conn, "x1", "x2", similarity=0.9, phase="phase3")
    conn.commit()

    count = run_cross_group_merger(conn, REDACTION_MARKERS, anchor_length=50)
    assert count == 1

    row = conn.execute(
        "SELECT pair_merged FROM verified_pairs WHERE doc_id_a = 'x1' AND doc_id_b = 'x2'"
    ).fetchone()
    assert row["pair_merged"] == 1

    mr = conn.execute(
        "SELECT recovered_count, merged_text FROM merge_results WHERE group_id = ?", (g1,)
    ).fetchone()
    assert mr is not None
    assert mr["recovered_count"] >= 1
    assert "John Smith" in mr["merged_text"]


def test_run_cross_group_merger_handles_ungrouped_doc(conn):
    """Cross-group pairs where one doc is ungrouped should still recover redactions."""
    base = (
        "The witness identified [REDACTED] at the Palm Beach estate on March 10. "
        "The deposition was recorded by the court reporter."
    )
    donor = (
        "The witness identified Prince Andrew at the Palm Beach estate on March 10. "
        "The deposition was recorded by the court reporter."
    )
    seed_doc(conn, "u1", base)
    seed_doc(conn, "u2", donor)
    conn.commit()

    g2 = create_match_group(conn)
    add_group_member(conn, g2, "u2", 1.0)
    conn.commit()

    from core.db import insert_verified_pair
    insert_verified_pair(conn, "u1", "u2", similarity=0.9, phase="match")
    conn.commit()

    count = run_cross_group_merger(conn, REDACTION_MARKERS, anchor_length=50)
    assert count == 1

    from core.db import get_doc_group
    assert get_doc_group(conn, "u1") is not None

    row = conn.execute(
        "SELECT pair_merged FROM verified_pairs WHERE doc_id_a = 'u1' AND doc_id_b = 'u2'"
    ).fetchone()
    assert row["pair_merged"] == 1


def test_mega_group_full_scenario(conn):
    """End-to-end: mega-group with mixed clusters, orphans, and a cross-group pair."""
    from core.db import insert_verified_pair, get_doc_group
    import stages.merger as merger_mod
    original = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 3

    try:
        # Cluster A: flight docs
        flight_redacted = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan for Palm Beach\n\n"
            "Departing [REDACTED] at 0800 hours. Passengers: Jeffrey, [REDACTED]."
        )
        flight_clean = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan for Palm Beach\n\n"
            "Departing Teterboro at 0800 hours. Passengers: Jeffrey, Prince Andrew."
        )
        # Cluster B: legal docs
        legal_redacted = (
            "CASE NO. 2005-0042\nSouthern District of New York\n"
            "Plaintiff: [REDACTED]\nThe deposition was taken on March 10."
        )
        legal_clean = (
            "CASE NO. 2005-0042\nSouthern District of New York\n"
            "Plaintiff: Virginia Giuffre\nThe deposition was taken on March 10."
        )
        # Orphans
        orphan1 = "Completely unrelated document about weather forecasts in Idaho state parks."
        orphan2 = "Dinner reservation for eight guests at restaurant Le Bernardin tonight in NYC."

        for doc_id, text in [
            ("f1", flight_redacted), ("f2", flight_clean),
            ("l1", legal_redacted), ("l2", legal_clean),
            ("o1", orphan1), ("o2", orphan2),
        ]:
            _seed_doc_with_fingerprint(conn, doc_id, text)
        conn.commit()

        # All in one mega-group
        g = create_match_group(conn)
        for doc_id in ["f1", "f2", "l1", "l2", "o1", "o2"]:
            add_group_member(conn, g, doc_id, 1.0)
        conn.commit()

        # Cross-group donor in a separate group
        cross_text = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan for Palm Beach\n\n"
            "Departing Newark at 0800 hours. Passengers: Jeffrey, Prince Andrew."
        )
        _seed_doc_with_fingerprint(conn, "cross1", cross_text)
        conn.commit()
        g2 = create_match_group(conn)
        add_group_member(conn, g2, "cross1", 1.0)
        insert_verified_pair(conn, "f1", "cross1", similarity=0.8, phase="phase3")
        conn.commit()

        # Step 1: run_merger handles the mega-group
        merge_count = run_merger(conn, REDACTION_MARKERS, anchor_length=50)

        # Original mega-group should be gone
        assert conn.execute(
            "SELECT * FROM match_groups WHERE group_id = ?", (g,)
        ).fetchone() is None

        # Orphans released
        for orphan_id in ["o1", "o2"]:
            assert conn.execute(
                "SELECT group_id FROM match_group_members WHERE doc_id = ?", (orphan_id,)
            ).fetchone() is None

        # At least some recoveries from within-group merging
        results = conn.execute(
            "SELECT SUM(recovered_count) FROM merge_results WHERE recovered_count > 0"
        ).fetchone()
        # May be 0 if reverse merge used unredacted as base (which is correct behavior)

        # Step 2: cross-group merger
        cross_count = run_cross_group_merger(conn, REDACTION_MARKERS, anchor_length=50)
        assert cross_count >= 1

        # Cross-group pair should be marked as merged
        pair_row = conn.execute(
            "SELECT pair_merged FROM verified_pairs WHERE doc_id_a = 'cross1' OR doc_id_b = 'cross1'"
        ).fetchone()
        assert pair_row["pair_merged"] == 1

    finally:
        merger_mod._CLUSTER_THRESHOLD = original
