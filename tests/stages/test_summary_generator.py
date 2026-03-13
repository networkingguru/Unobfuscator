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


# ── Task 2: Entity Extractors ──

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


# ── Task 3: Entity Aggregation ──

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
