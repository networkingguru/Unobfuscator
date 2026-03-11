import pytest
from unittest.mock import patch, MagicMock, call
from core.db import init_db, get_connection, get_all_fingerprints, get_unprocessed_documents
from stages.indexer import (
    clean_text, shingle, build_fingerprint,
    index_document, run_indexer_batch
)

REDACTION_MARKERS = ["[REDACTED]", "[b(6)]", "XXXXXXXXX"]


# --- Text cleaning and shingling ---

def test_clean_text_strips_redaction_markers():
    text = "Hello [REDACTED] world [b(6)] test XXXXXXXXX end"
    result = clean_text(text, REDACTION_MARKERS)
    assert "[REDACTED]" not in result
    assert "[b(6)]" not in result
    assert "XXXXXXXXX" not in result
    assert "hello" in result  # lowercased


def test_clean_text_normalizes_whitespace():
    text = "one   two\n\nthree\t\tfour"
    result = clean_text(text, REDACTION_MARKERS)
    assert "  " not in result
    assert "\n" not in result
    assert "\t" not in result


def test_shingle_produces_overlapping_windows():
    words = "a b c d e f g h i j".split()
    shingles = shingle(" ".join(words), window=4)
    assert "a b c d" in shingles
    assert "b c d e" in shingles
    assert len(shingles) == len(words) - 4 + 1


def test_shingle_returns_empty_for_short_text():
    shingles = shingle("only three words", window=8)
    assert shingles == []


def test_build_fingerprint_returns_bytes_of_expected_length():
    text = " ".join(["word"] * 100)  # enough words to shingle
    sig = build_fingerprint(text, num_perm=128)
    assert isinstance(sig, bytes)
    assert len(sig) > 0


def test_build_fingerprint_is_consistent():
    text = "the quick brown fox jumped over the lazy dog today"
    sig1 = build_fingerprint(text, num_perm=128)
    sig2 = build_fingerprint(text, num_perm=128)
    assert sig1 == sig2


# --- DB integration ---

@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def test_index_document_stores_doc_and_fingerprint(conn):
    doc = {
        "id": 1, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "a.pdf", "page_count": 2,
        "size_bytes": 500, "description": "Test",
        "extracted_text": "The flight departed from Palm Beach with several passengers on board.",
        "pdf_url": "https://data.jmail.world/v1/files/doj/VOL00001/a.pdf",
    }
    index_document(conn, doc, redaction_markers=REDACTION_MARKERS, num_perm=128)
    conn.commit()
    assert get_unprocessed_documents(conn) == []  # marked as processed
    fps = get_all_fingerprints(conn)
    assert len(fps) == 1
    assert fps[0]["doc_id"] == 1


def test_index_document_handles_empty_text_gracefully(conn):
    doc = {
        "id": 2, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "empty.pdf", "page_count": 1,
        "size_bytes": 100, "description": "Empty doc",
        "extracted_text": "",
        "pdf_url": None,
    }
    index_document(conn, doc, redaction_markers=REDACTION_MARKERS, num_perm=128)
    conn.commit()
    # Should not crash; no fingerprint for empty docs
    fps = get_all_fingerprints(conn)
    assert len(fps) == 0


@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_run_indexer_batch_processes_all_docs(mock_text_batch, mock_meta, conn):
    mock_meta.return_value = [
        {"id": 10, "source": "doj", "release_batch": "VOL00001",
         "original_filename": "x.pdf", "page_count": 1,
         "size_bytes": 200, "description": "Doc 10"},
    ]
    mock_text_batch.return_value = {
        10: "This is a document with enough words to generate shingles for testing."
    }
    run_indexer_batch(conn, batch_id="VOL00001",
                      redaction_markers=REDACTION_MARKERS, num_perm=128)
    conn.commit()
    fps = get_all_fingerprints(conn)
    assert len(fps) == 1
    assert fps[0]["doc_id"] == 10


@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_run_indexer_batch_uses_batch_text_fetch(mock_text_batch, mock_meta, conn):
    """fetch_documents_text_batch must be called once (not per-document) with all IDs."""
    mock_meta.return_value = [
        {"id": 20, "source": "fbi", "release_batch": "VOL00002",
         "original_filename": "a.pdf", "page_count": 1,
         "size_bytes": 100, "description": "Doc 20"},
        {"id": 21, "source": "fbi", "release_batch": "VOL00002",
         "original_filename": "b.pdf", "page_count": 1,
         "size_bytes": 100, "description": "Doc 21"},
    ]
    mock_text_batch.return_value = {20: "some text", 21: "other text"}
    run_indexer_batch(conn, batch_id="VOL00002",
                      redaction_markers=REDACTION_MARKERS, num_perm=128)
    # Must be called exactly once with both IDs
    mock_text_batch.assert_called_once_with([20, 21])


@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_run_indexer_batch_stores_pdf_url(mock_text_batch, mock_meta, conn):
    """pdf_url must be populated in the documents table after indexing."""
    mock_meta.return_value = [
        {"id": 30, "source": "doj", "release_batch": "VOL00003",
         "original_filename": "report.pdf", "page_count": 3,
         "size_bytes": 800, "description": "Doc 30"},
    ]
    mock_text_batch.return_value = {30: ""}
    run_indexer_batch(conn, batch_id="VOL00003",
                      redaction_markers=REDACTION_MARKERS, num_perm=128)
    conn.commit()
    row = conn.execute("SELECT pdf_url FROM documents WHERE id = 30").fetchone()
    assert row is not None
    expected_url = "https://data.jmail.world/v1/files/doj/VOL00003/report.pdf"
    assert row["pdf_url"] == expected_url
