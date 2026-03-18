import pytest
import json
from pathlib import Path
from core.db import (
    init_db, get_connection, upsert_document, get_unprocessed_documents,
    mark_text_processed, upsert_fingerprint, get_all_fingerprints,
    create_match_group, add_group_member, get_doc_group, merge_groups,
    upsert_merge_result, get_config, set_config
)
from core.db import (
    get_docs_needing_text_recovery, get_docs_needing_backfill,
    update_extracted_text, mark_ocr_processed
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


def test_new_columns_exist_after_init(conn):
    row = conn.execute("SELECT text_source, ocr_processed, page_tags FROM documents LIMIT 0").fetchone()


def test_update_extracted_text_stores_text_and_source(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    update_extracted_text(conn, SAMPLE_DOC["id"], "Recovered text", "ocr")
    conn.commit()
    row = conn.execute("SELECT extracted_text, text_source FROM documents WHERE id = ?", (SAMPLE_DOC["id"],)).fetchone()
    assert row["extracted_text"] == "Recovered text"
    assert row["text_source"] == "ocr"


def test_update_extracted_text_stores_page_tags(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    tags = '{"0": "text", "1": "photo"}'
    update_extracted_text(conn, SAMPLE_DOC["id"], "text", "ocr", page_tags=tags)
    conn.commit()
    row = conn.execute("SELECT page_tags FROM documents WHERE id = ?", (SAMPLE_DOC["id"],)).fetchone()
    assert row["page_tags"] == tags


def test_mark_ocr_processed(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    mark_ocr_processed(conn, SAMPLE_DOC["id"])
    conn.commit()
    row = conn.execute("SELECT ocr_processed FROM documents WHERE id = ?", (SAMPLE_DOC["id"],)).fetchone()
    assert row["ocr_processed"] == 1


def test_get_docs_needing_text_recovery(conn):
    doc_no_text = {**SAMPLE_DOC, "id": "no_text", "extracted_text": ""}
    doc_has_text = {**SAMPLE_DOC, "id": "has_text", "extracted_text": "hello world"}
    upsert_document(conn, doc_no_text)
    upsert_document(conn, doc_has_text)
    conn.commit()
    docs = get_docs_needing_text_recovery(conn)
    ids = [d["id"] for d in docs]
    assert "no_text" in ids
    assert "has_text" not in ids


def test_get_docs_needing_text_recovery_excludes_ocr_processed(conn):
    doc = {**SAMPLE_DOC, "id": "processed", "extracted_text": ""}
    upsert_document(conn, doc)
    mark_ocr_processed(conn, "processed")
    conn.commit()
    docs = get_docs_needing_text_recovery(conn)
    ids = [d["id"] for d in docs]
    assert "processed" not in ids


def test_get_docs_needing_backfill(conn):
    doc = {**SAMPLE_DOC, "id": "backfill_me", "extracted_text": "", "release_batch": "VOL00001"}
    upsert_document(conn, doc)
    mark_text_processed(conn, "backfill_me")
    conn.commit()
    docs = get_docs_needing_backfill(conn, known_batches={"VOL00001"})
    ids = [d["id"] for d in docs]
    assert "backfill_me" in ids


def test_get_docs_needing_backfill_excludes_unknown_batch(conn):
    doc = {**SAMPLE_DOC, "id": "unknown_batch", "extracted_text": "", "release_batch": "MYSTERY"}
    upsert_document(conn, doc)
    mark_text_processed(conn, "unknown_batch")
    conn.commit()
    docs = get_docs_needing_backfill(conn, known_batches={"VOL00001"})
    ids = [d["id"] for d in docs]
    assert "unknown_batch" not in ids


def test_migration_sets_text_source_for_existing_docs(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    from core.db import _migrate_text_recovery_columns
    _migrate_text_recovery_columns(conn)
    conn.commit()
    row = conn.execute("SELECT text_source FROM documents WHERE id = ?", (SAMPLE_DOC["id"],)).fetchone()
    assert row["text_source"] == "jmail"
