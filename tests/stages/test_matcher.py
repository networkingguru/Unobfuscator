import pytest
import numpy as np
from datasketch import MinHash
from core.db import (
    init_db, get_connection, upsert_document, upsert_fingerprint,
    get_doc_group, create_match_group, add_group_member, get_config
)
from unittest.mock import patch
from stages.matcher import (
    extract_email_headers, run_phase0_email_fastpath,
    load_fingerprints, run_phase2_lsh_candidates,
    find_longest_common_substring, run_phase3_verify_and_group,
    _get_total_ram_bytes, _get_rss_bytes, _check_memory, MemoryLimitExceeded,
)
from stages.indexer import build_fingerprint, clean_text


REDACTION_MARKERS = ["[REDACTED]", "[b(6)]"]

EMAIL_TEXT_A = (
    "From: jeffrey@example.com\nTo: assistant@example.com\n"
    "Date: 2002-03-10\nSubject: Meeting notes\n\n"
    "The attendees included [REDACTED] and Prince Andrew. "
    "Location was the island."
)

EMAIL_TEXT_B = (
    "From: jeffrey@example.com\nTo: assistant@example.com\n"
    "Date: 2002-03-10\nSubject: Meeting notes\n\n"
    "The attendees included Bill Clinton and Prince Andrew. "
    "Location was [REDACTED]."
)

UNRELATED_TEXT = (
    "Budget report Q3. Total expenses were 2.4 million. "
    "Department heads reviewed the figures."
)

# Two versions of the same document with >500 chars of shared body text.
# Neither redaction marker appears in both texts simultaneously, so
# _has_complementary_redactions returns False — grouping is confirmed by
# the long common text (secondary signal: >500 chars).
_LONG_SHARED_BODY = (
    "The Palm Beach estate served as the primary venue for gatherings during the period "
    "in question. Multiple witnesses confirmed that the events described in the attached "
    "exhibit took place on or around the dates listed. Transportation was arranged by "
    "staff. Travel logs were maintained and are available upon subpoena. The relevant "
    "parties were present at all material times and their identities are known to counsel. "
    "Corroborating documentation was produced in discovery and has been reviewed by all "
    "parties prior to filing with the court."
)
LONG_OVERLAP_TEXT_A = "Version A preamble. " + _LONG_SHARED_BODY + " End of version A."
LONG_OVERLAP_TEXT_B = "Version B preamble. " + _LONG_SHARED_BODY + " End of version B."


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def seed_doc(conn, doc_id, text, source="doj"):
    doc = {
        "id": doc_id, "source": source, "release_batch": "VOL00001",
        "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
        "size_bytes": 500, "description": "", "extracted_text": text
    }
    upsert_document(conn, doc)


# --- Phase 0 ---

def test_extract_email_headers_finds_from_to_date_subject():
    headers = extract_email_headers(EMAIL_TEXT_A)
    assert any("jeffrey@example.com" in h for h in headers)
    assert any("2002-03-10" in h for h in headers)
    assert any("Meeting notes" in h for h in headers)


def test_extract_email_headers_returns_empty_for_non_email():
    headers = extract_email_headers(UNRELATED_TEXT)
    assert headers == []


def test_phase0_groups_docs_with_matching_headers(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
    seed_doc(conn, 3, UNRELATED_TEXT)
    conn.commit()
    matched = run_phase0_email_fastpath(
        conn, min_header_matches=2
    )
    assert matched == {"1", "2"}
    assert get_doc_group(conn, 1) is not None
    assert get_doc_group(conn, 1) == get_doc_group(conn, 2)
    assert get_doc_group(conn, 3) is None


def test_phase0_does_not_group_single_header_match(conn):
    # Only Subject matches, not From/To/Date — should not group
    text_a = "From: a@a.com\nSubject: Same subject\n\nContent A here with more text."
    text_b = "From: b@b.com\nSubject: Same subject\n\nContent B here with more text."
    seed_doc(conn, 1, text_a)
    seed_doc(conn, 2, text_b)
    conn.commit()
    run_phase0_email_fastpath(conn, min_header_matches=2)
    assert get_doc_group(conn, 1) is None
    assert get_doc_group(conn, 2) is None


# --- Phase 2 (LSH) ---

def make_fingerprint(text):
    m = MinHash(num_perm=128)
    from stages.indexer import shingle
    for s in shingle(clean_text(text, REDACTION_MARKERS)):
        m.update(s.encode("utf-8"))
    return m.hashvalues.tobytes()


def test_load_fingerprints_returns_doc_id_to_minhash_map(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    upsert_fingerprint(conn, 1, make_fingerprint(EMAIL_TEXT_A), 50)
    conn.commit()
    fps = load_fingerprints(conn, num_perm=128)
    assert "1" in fps
    assert isinstance(fps["1"], MinHash)


def test_phase2_finds_similar_docs_as_candidates(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
    seed_doc(conn, 3, UNRELATED_TEXT)
    upsert_fingerprint(conn, 1, make_fingerprint(EMAIL_TEXT_A), 50)
    upsert_fingerprint(conn, 2, make_fingerprint(EMAIL_TEXT_B), 50)
    upsert_fingerprint(conn, 3, make_fingerprint(UNRELATED_TEXT), 10)
    conn.commit()
    candidates = run_phase2_lsh_candidates(conn, threshold=0.3, num_perm=128)
    # docs 1 and 2 share most content — should be candidates
    assert ("1", "2") in candidates or ("2", "1") in candidates
    # doc 3 is unrelated — should not be paired with 1 or 2
    assert ("1", "3") not in candidates and ("3", "1") not in candidates


# --- Phase 3 (Verification) ---

def test_find_longest_common_substring_ignores_redaction_markers():
    a = "The passenger list included [REDACTED] and Prince Andrew."
    b = "The passenger list included Bill Clinton and Prince Andrew."
    common = find_longest_common_substring(a, b, REDACTION_MARKERS)
    assert "Prince Andrew" in common
    assert len(common) >= 20


def test_find_longest_common_substring_returns_empty_for_unrelated():
    a = "The quick brown fox jumped over the fence today."
    b = "Budget figures show a deficit of two million dollars."
    common = find_longest_common_substring(a, b, REDACTION_MARKERS)
    assert len(common) < 10


def test_phase3_groups_confirmed_candidate_pair(conn):
    # Uses >500 chars of shared text (secondary confirmation signal).
    seed_doc(conn, 1, LONG_OVERLAP_TEXT_A)
    seed_doc(conn, 2, LONG_OVERLAP_TEXT_B)
    conn.commit()
    candidates = [(1, 2)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=20
    )
    conn.commit()
    assert get_doc_group(conn, 1) is not None
    assert get_doc_group(conn, 1) == get_doc_group(conn, 2)


def test_phase3_rejects_unrelated_pair(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 3, UNRELATED_TEXT)
    conn.commit()
    candidates = [(1, 3)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=50
    )
    conn.commit()
    assert get_doc_group(conn, 1) is None
    assert get_doc_group(conn, 3) is None


def test_phase3_merges_two_existing_groups(conn):
    # doc 1 and 2 are already in separate groups; phase3 should merge them.
    # Uses >500 chars of shared text (secondary confirmation signal).
    seed_doc(conn, 1, LONG_OVERLAP_TEXT_A)
    seed_doc(conn, 2, LONG_OVERLAP_TEXT_B)
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, 1, 1.0)
    add_group_member(conn, g2, 2, 1.0)
    conn.commit()
    candidates = [(1, 2)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=20
    )
    conn.commit()
    assert get_doc_group(conn, 1) == get_doc_group(conn, 2)


def test_phase3_groups_complementary_redactions_with_short_overlap(conn):
    # One doc has a redaction where the other has text — complementary signal.
    # Common text is well under 500 chars, but complementary redactions confirm the match.
    shared = "The meeting took place at the Palm Beach estate on March 10th 2002."
    text_a = shared + " Guests included [REDACTED] and several staff members."
    text_b = shared + " Guests included Prince Andrew and several staff members."
    seed_doc(conn, 1, text_a)
    seed_doc(conn, 2, text_b)
    conn.commit()
    candidates = [(1, 2)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=20
    )
    conn.commit()
    # Complementary redactions present → should be grouped despite short common text
    assert get_doc_group(conn, 1) is not None
    assert get_doc_group(conn, 1) == get_doc_group(conn, 2)


def test_phase3_rejects_medium_overlap_without_complementary_redactions(conn):
    # Shared segment is between min_overlap_chars (200) and 500 chars.
    # Neither document has any redaction markers — no complementary signal.
    # The new logic requires complementary redactions OR >500 chars; this has neither.
    shared = (
        "This document concerns the financial arrangements made between the parties "
        "in the spring of 2001. The terms were agreed upon after several rounds of "
        "negotiation and were considered final by all participants at that time."
    )
    # shared is ~230 chars — above min_overlap_chars=200 but below 500
    text_a = "Addendum A. " + shared + " No further amendments were anticipated."
    text_b = "Addendum B. " + shared + " Subsequent reviews confirmed the arrangements."
    seed_doc(conn, 1, text_a)
    seed_doc(conn, 2, text_b)
    conn.commit()
    candidates = [(1, 2)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=200
    )
    conn.commit()
    # No complementary redactions AND common text <= 500 chars → should be rejected
    assert get_doc_group(conn, 1) is None
    assert get_doc_group(conn, 2) is None


# --- Memory measurement helpers ---


def test_get_total_ram_bytes_returns_positive_int():
    total = _get_total_ram_bytes()
    assert isinstance(total, int)
    assert total > 0
    assert total >= 512 * 1024 * 1024
    assert total <= 1024 * 1024 * 1024 * 1024


def test_get_rss_bytes_returns_positive_int():
    rss = _get_rss_bytes()
    assert isinstance(rss, int)
    assert rss > 0
    assert rss >= 10 * 1024 * 1024


def test_rss_is_less_than_total_ram():
    rss = _get_rss_bytes()
    total = _get_total_ram_bytes()
    assert rss < total


# --- Memory check and exception ---


def test_check_memory_passes_when_under_limit():
    """No exception when RSS is well under the limit."""
    with patch("stages.matcher._get_rss_bytes", return_value=1_000_000_000), \
         patch("stages.matcher._get_total_ram_bytes", return_value=16_000_000_000):
        _check_memory(70)  # no exception


def test_check_memory_raises_when_over_limit():
    """MemoryLimitExceeded when RSS exceeds limit_pct of total RAM."""
    with patch("stages.matcher._get_rss_bytes", return_value=12_000_000_000), \
         patch("stages.matcher._get_total_ram_bytes", return_value=16_000_000_000):
        with pytest.raises(MemoryLimitExceeded) as exc_info:
            _check_memory(70)
        assert exc_info.value.rss_mb == 12_000_000_000 // (1024 * 1024)
        assert exc_info.value.limit_mb == (16_000_000_000 * 70 // 100) // (1024 * 1024)
        assert exc_info.value.limit_pct == 70
        assert exc_info.value.total_mb == 16_000_000_000 // (1024 * 1024)


def test_check_memory_skips_when_rss_unavailable():
    """If _get_rss_bytes returns 0, guard is disabled — no exception."""
    with patch("stages.matcher._get_rss_bytes", return_value=0), \
         patch("stages.matcher._get_total_ram_bytes", return_value=16_000_000_000):
        _check_memory(70)  # no exception


def test_load_fingerprints_raises_when_memory_exceeded(conn):
    """load_fingerprints should raise MemoryLimitExceeded when over limit."""
    for i in range(5):
        text = f"unique text number {i} " * 50
        seed_doc(conn, f"doc_{i}", text)
        sig = build_fingerprint(clean_text(text, []))
        upsert_fingerprint(conn, f"doc_{i}", sig, 100)
    conn.commit()

    with patch("stages.matcher._check_memory",
               side_effect=MemoryLimitExceeded(8547, 8192, 70, 16384)):
        with pytest.raises(MemoryLimitExceeded):
            load_fingerprints(conn, memory_limit_pct=70)


def test_phase2_catches_memory_exceeded_returns_empty(conn):
    """run_phase2_lsh_candidates should return [] and write DB flag on memory exceeded."""
    for i in range(5):
        text = f"unique text number {i} " * 50
        seed_doc(conn, f"doc_{i}", text)
        sig = build_fingerprint(clean_text(text, []))
        upsert_fingerprint(conn, f"doc_{i}", sig, 100)
    conn.commit()

    with patch("stages.matcher._check_memory",
               side_effect=MemoryLimitExceeded(8547, 8192, 70, 16384)):
        result = run_phase2_lsh_candidates(conn, memory_limit_pct=70)

    assert result == []
    warning = get_config(conn, "lsh_memory_warning", default="")
    assert "memory limit" in warning.lower()


def test_phase2_clears_warning_on_success(conn):
    """A successful LSH run should clear any previous memory warning."""
    from core.db import set_config
    set_config(conn, "lsh_memory_warning", "previous warning")
    conn.commit()

    for i in range(2):
        doc = {
            "id": f"clear_test_{i}", "source": "test",
            "release_batch": "TEST", "original_filename": f"t{i}.pdf",
            "page_count": 1, "size_bytes": 100,
            "description": "test", "extracted_text": f"text {i} " * 50,
        }
        upsert_document(conn, doc)
        sig = build_fingerprint(clean_text(doc["extracted_text"], []))
        upsert_fingerprint(conn, doc["id"], sig, 100)
    conn.commit()

    run_phase2_lsh_candidates(conn, memory_limit_pct=70)

    warning = get_config(conn, "lsh_memory_warning", default="")
    assert warning == ""


def test_memory_limit_exceeded_has_correct_attributes():
    exc = MemoryLimitExceeded(rss_mb=8547, limit_mb=8192, limit_pct=50, total_mb=16384)
    assert exc.rss_mb == 8547
    assert exc.limit_mb == 8192
    assert exc.limit_pct == 50
    assert exc.total_mb == 16384
