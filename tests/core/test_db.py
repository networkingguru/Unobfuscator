import pytest
import json
from pathlib import Path
from core.db import (
    init_db, get_connection, upsert_document, get_unprocessed_documents,
    mark_text_processed, upsert_fingerprint, get_all_fingerprints,
    create_match_group, add_group_member, get_doc_group, merge_groups,
    upsert_merge_result, get_config, set_config
)


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def conn(db_path):
    c = get_connection(db_path)
    yield c
    c.close()


SAMPLE_DOC = {
    "id": 1, "source": "doj", "release_batch": "VOL00001",
    "original_filename": "test.pdf", "page_count": 5,
    "size_bytes": 1000, "description": "A test document",
    "extracted_text": "This is test content about someone important.",
    "pdf_url": "https://data.jmail.world/v1/files/doj/VOL00001/test.pdf",
}


def test_init_db_creates_all_tables(db_path):
    conn = get_connection(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for expected in ["documents", "document_fingerprints", "match_groups",
                     "match_group_members", "merge_results", "release_batches",
                     "jobs", "config"]:
        assert expected in tables


def test_upsert_document_stores_record(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    rows = get_unprocessed_documents(conn)
    assert len(rows) == 1
    assert rows[0]["id"] == "1"
    assert rows[0]["extracted_text"] == SAMPLE_DOC["extracted_text"]


def test_mark_text_processed_removes_from_unprocessed(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    mark_text_processed(conn, 1)
    conn.commit()
    assert get_unprocessed_documents(conn) == []


def test_upsert_fingerprint_stores_and_retrieves(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    sig = b"\x01\x02\x03"
    upsert_fingerprint(conn, 1, sig, 42)
    conn.commit()
    fps = get_all_fingerprints(conn)
    assert len(fps) == 1
    assert fps[0]["doc_id"] == "1"
    assert fps[0]["minhash_sig"] == sig


def test_match_group_member_uniqueness(conn):
    upsert_document(conn, SAMPLE_DOC)
    upsert_document(conn, {**SAMPLE_DOC, "id": 2, "original_filename": "b.pdf"})
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    conn.commit()
    add_group_member(conn, g1, 1, 1.0)
    conn.commit()
    # Adding doc 1 to g2 should be ignored (UNIQUE on doc_id)
    add_group_member(conn, g2, 1, 0.9)
    conn.commit()
    assert get_doc_group(conn, 1) == g1


def test_merge_groups_reassigns_members(conn):
    for i in [1, 2]:
        upsert_document(conn, {**SAMPLE_DOC, "id": i, "original_filename": f"{i}.pdf"})
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    conn.commit()
    add_group_member(conn, g1, 1, 1.0)
    add_group_member(conn, g2, 2, 1.0)
    conn.commit()
    merge_groups(conn, g1, g2)
    conn.commit()
    assert get_doc_group(conn, 2) == g1


def test_upsert_merge_result_tracks_previous_count(conn):
    g = create_match_group(conn)
    conn.commit()
    upsert_merge_result(conn, g, "merged text", 5, 10, [1, 2])
    conn.commit()
    upsert_merge_result(conn, g, "more merged text", 8, 10, [1, 2, 3])
    conn.commit()
    row = conn.execute(
        "SELECT recovered_count, previous_recovered_count FROM merge_results WHERE group_id = ?", (g,)
    ).fetchone()
    assert row["recovered_count"] == 8
    assert row["previous_recovered_count"] == 5


def test_config_get_set_roundtrip(conn):
    set_config(conn, "test_key", {"nested": 42})
    conn.commit()
    assert get_config(conn, "test_key") == {"nested": 42}


def test_config_get_returns_default_when_missing(conn):
    assert get_config(conn, "nonexistent", default="fallback") == "fallback"


def test_upsert_document_stores_pdf_url(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    row = conn.execute("SELECT pdf_url FROM documents WHERE id = 1").fetchone()
    assert row is not None
    assert row["pdf_url"] == SAMPLE_DOC["pdf_url"]


def test_upsert_document_without_pdf_url_stores_null(conn):
    doc = {**SAMPLE_DOC, "id": 99, "pdf_url": None}
    upsert_document(conn, doc)
    conn.commit()
    row = conn.execute("SELECT pdf_url FROM documents WHERE id = 99").fetchone()
    assert row is not None
    assert row["pdf_url"] is None
