import pytest
import io
import fitz  # PyMuPDF
from unittest.mock import patch, MagicMock
from core.db import init_db, get_connection, upsert_document, create_match_group, add_group_member
from stages.pdf_processor import (
    extract_soft_redactions, build_pdf_url, process_pdf_for_document
)


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def seed_doc(conn, doc_id, source="doj", batch="VOL00001", filename="test.pdf"):
    from core.db import upsert_document
    upsert_document(conn, {
        "id": doc_id, "source": source, "release_batch": batch,
        "original_filename": filename, "page_count": 1,
        "size_bytes": 1000, "description": "Test doc",
        "extracted_text": "Some text with [REDACTED] in it."
    })


def make_pdf_with_soft_redaction(hidden_text: str) -> bytes:
    """Create an in-memory PDF where text is present but covered by a black rectangle.

    This simulates a poorly-done Acrobat redaction: the text stream still contains
    the words, but a filled black rectangle annotation sits on top.
    """
    doc = fitz.open()
    page = doc.new_page()
    # Insert the "hidden" text into the page stream
    rect = fitz.Rect(50, 50, 300, 70)
    page.insert_text((50, 65), hidden_text, fontsize=12)
    # Cover it with a filled black rectangle (simulating soft redaction)
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(fill=(0, 0, 0), color=(0, 0, 0))
    shape.commit()
    return doc.tobytes()


def make_clean_pdf() -> bytes:
    """Create a clean PDF with visible text and no redaction overlays."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 65), "This is fully visible public text.", fontsize=12)
    return doc.tobytes()


# --- build_pdf_url ---

def test_build_pdf_url_constructs_exact_expected_url():
    url = build_pdf_url(doc_id=12345, source="doj", batch="VOL00008",
                        original_filename="myfile.pdf")
    assert url == "https://data.jmail.world/v1/files/doj/VOL00008/myfile.pdf"


# --- extract_soft_redactions ---

def test_extract_soft_redactions_finds_text_under_black_rectangle():
    pdf_bytes = make_pdf_with_soft_redaction("Secret name here")
    recovered = extract_soft_redactions(pdf_bytes)
    assert len(recovered) > 0
    assert any("Secret" in r["text"] for r in recovered)


def test_extract_soft_redactions_returns_empty_for_clean_pdf():
    pdf_bytes = make_clean_pdf()
    recovered = extract_soft_redactions(pdf_bytes)
    # A clean PDF with no black overlays should yield no soft redactions
    assert recovered == []


def test_extract_soft_redactions_returns_empty_for_invalid_pdf():
    recovered = extract_soft_redactions(b"not a pdf")
    assert recovered == []


# --- process_pdf_for_document ---

@patch("stages.pdf_processor.httpx.get")
def test_process_pdf_inserts_merge_job_when_soft_redaction_found(mock_get, conn):
    seed_doc(conn, 1)
    conn.execute("UPDATE documents SET pdf_url = 'http://example.com/doc.pdf' WHERE id = 1")
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    conn.commit()

    pdf_bytes = make_pdf_with_soft_redaction("Hidden content found")
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = pdf_bytes

    process_pdf_for_document(conn, doc_id=1)
    conn.commit()

    # Should insert a merge job for the group
    job = conn.execute(
        "SELECT * FROM jobs WHERE stage = 'merge' AND status = 'pending'"
    ).fetchone()
    assert job is not None
    import json
    assert json.loads(job["payload"])["group_id"] == g


@patch("stages.pdf_processor.httpx.get")
def test_process_pdf_skips_when_no_soft_redactions(mock_get, conn):
    seed_doc(conn, 2)
    conn.execute("UPDATE documents SET pdf_url = 'http://example.com/doc2.pdf' WHERE id = 2")
    conn.commit()

    mock_get.return_value.status_code = 200
    mock_get.return_value.content = make_clean_pdf()

    process_pdf_for_document(conn, doc_id=2)
    conn.commit()

    job = conn.execute(
        "SELECT * FROM jobs WHERE stage = 'merge' AND status = 'pending'"
    ).fetchone()
    assert job is None


@patch("stages.pdf_processor.httpx.get")
def test_process_pdf_marks_pdf_processed_on_completion(mock_get, conn):
    seed_doc(conn, 3)
    conn.execute("UPDATE documents SET pdf_url = 'http://example.com/doc3.pdf' WHERE id = 3")
    conn.commit()

    mock_get.return_value.status_code = 200
    mock_get.return_value.content = make_clean_pdf()

    process_pdf_for_document(conn, doc_id=3)
    conn.commit()

    row = conn.execute("SELECT pdf_processed FROM documents WHERE id = 3").fetchone()
    assert row["pdf_processed"] == 1


@patch("stages.pdf_processor.httpx.get")
def test_process_pdf_skips_document_with_no_pdf_url(mock_get, conn):
    seed_doc(conn, 4)
    conn.commit()  # pdf_url is NULL

    process_pdf_for_document(conn, doc_id=4)

    mock_get.assert_not_called()
