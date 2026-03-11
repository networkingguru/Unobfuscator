import pytest
import yaml
from unittest.mock import patch, call
from core.db import init_db, get_connection, insert_release_batch, upsert_document
from unobfuscator import _run_one_cycle


@pytest.fixture
def cfg(tmp_path):
    config = {
        "output_dir": str(tmp_path / "output"),
        "db_path": str(tmp_path / "test.db"),
        "matching": {
            "min_overlap_chars": 20,
            "similarity_threshold": 0.3,
            "email_header_min_matches": 2
        },
        "polling": {"interval_minutes": 60},
        "redaction_markers": ["[REDACTED]"],
        "workers": {"text": 1, "pdf": 1}
    }
    return config


@pytest.fixture
def conn(cfg):
    init_db(cfg["db_path"])
    c = get_connection(cfg["db_path"])
    yield c
    c.close()


@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_run_one_cycle_completes_without_error(mock_text_batch, mock_meta, conn, cfg):
    """Smoke test: one full cycle with a seeded batch does not raise."""
    insert_release_batch(conn, "VOL00001")
    conn.commit()

    mock_meta.return_value = [{
        "id": 1, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "a.pdf", "page_count": 1,
        "size_bytes": 100, "description": ""
    }]
    mock_text_batch.return_value = {
        1: (
            "From: a@a.com\nTo: b@b.com\nDate: 2002-01-01\nSubject: Test\n\n"
            "The attendee was [REDACTED] at the location."
        )
    }

    _run_one_cycle(conn, cfg)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1


@patch("stages.pdf_processor.httpx.get")
@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_run_one_cycle_processes_multiple_pdfs(mock_text_batch, mock_meta, mock_http, conn, cfg):
    """Multiple pending PDFs are all processed in a single daemon cycle."""
    # Seed two documents that already have pdf_url set and are pending PDF processing.
    for doc_id in (10, 11):
        upsert_document(conn, {
            "id": doc_id, "source": "doj", "release_batch": "VOL00002",
            "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
            "size_bytes": 500, "description": "", "extracted_text": "Some text."
        })
        conn.execute(
            "UPDATE documents SET pdf_url = ?, text_processed = 1 WHERE id = ?",
            (f"http://example.com/doc{doc_id}.pdf", doc_id)
        )
    conn.commit()

    # Return a minimal clean PDF so process_pdf_for_document succeeds without errors.
    import fitz
    doc = fitz.open()
    doc.new_page()
    clean_pdf = doc.tobytes()
    mock_http.return_value.status_code = 200
    mock_http.return_value.content = clean_pdf

    # No batches to index, but two PDFs are pending.
    mock_meta.return_value = []
    mock_text_batch.return_value = {}

    # Use pdf worker limit of 5 so both documents are picked up.
    cfg_with_pdf_workers = {**cfg, "workers": {"text": 1, "pdf": 5}}
    _run_one_cycle(conn, cfg_with_pdf_workers)
    conn.commit()

    processed = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE pdf_processed = 1"
    ).fetchone()[0]
    assert processed == 2, f"Expected 2 PDFs processed, got {processed}"
