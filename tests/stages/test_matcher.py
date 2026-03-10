import pytest
import numpy as np
from datasketch import MinHash
from core.db import (
    init_db, get_connection, upsert_document, upsert_fingerprint,
    get_doc_group, create_match_group, add_group_member
)
from stages.matcher import (
    extract_email_headers, run_phase0_email_fastpath,
    load_fingerprints, run_phase2_lsh_candidates,
    find_longest_common_substring, run_phase3_verify_and_group
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
    assert matched == {1, 2}
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
    assert 1 in fps
    assert isinstance(fps[1], MinHash)


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
    assert (1, 2) in candidates or (2, 1) in candidates
    # doc 3 is unrelated — should not be paired with 1 or 2
    assert (1, 3) not in candidates and (3, 1) not in candidates


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
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
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
    # doc 1 and 2 are already in separate groups; phase3 should merge them
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
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
