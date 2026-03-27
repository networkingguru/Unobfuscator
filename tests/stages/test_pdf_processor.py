import pytest
import io
import threading
import fitz  # PyMuPDF
from unittest.mock import patch, MagicMock
from core.db import init_db, get_connection, upsert_document, create_match_group, add_group_member
from stages.pdf_processor import extract_soft_redactions, process_pdf_for_document


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

@patch("stages.pdf_processor._download_pdf")
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


@patch("stages.pdf_processor._download_pdf")
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


@patch("stages.pdf_processor._download_pdf")
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


@patch("stages.pdf_processor._download_pdf")
def test_process_pdf_skips_document_with_no_pdf_url(mock_get, conn):
    seed_doc(conn, 4)
    conn.commit()  # pdf_url is NULL

    process_pdf_for_document(conn, doc_id=4)

    mock_get.assert_not_called()


# --- fingerprint regeneration after soft-redaction recovery ---

@patch("stages.pdf_processor._download_pdf")
def test_process_pdf_regenerates_fingerprint_after_soft_redaction(mock_get, conn):
    """When soft redactions are found, the document's fingerprint should be updated."""
    seed_doc(conn, 5)
    conn.execute("UPDATE documents SET pdf_url = 'http://example.com/doc5.pdf' WHERE id = 5")
    conn.commit()

    # Record the original fingerprint state (should be empty)
    old_fp = conn.execute(
        "SELECT * FROM document_fingerprints WHERE doc_id = 5"
    ).fetchone()
    assert old_fp is None  # No fingerprint yet

    pdf_bytes = make_pdf_with_soft_redaction("This is a long enough hidden text that should produce shingles for fingerprinting purposes in the document")
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = pdf_bytes

    process_pdf_for_document(conn, doc_id=5, redaction_markers=["[REDACTED]"])
    conn.commit()

    # A fingerprint should now exist
    new_fp = conn.execute(
        "SELECT * FROM document_fingerprints WHERE doc_id = 5"
    ).fetchone()
    assert new_fp is not None
    assert new_fp["shingle_count"] > 0


@patch("stages.pdf_processor._download_pdf")
def test_process_pdf_no_fingerprint_when_no_soft_redactions(mock_get, conn):
    """When no soft redactions are found, no fingerprint should be created."""
    seed_doc(conn, 6)
    conn.execute("UPDATE documents SET pdf_url = 'http://example.com/doc6.pdf' WHERE id = 6")
    conn.commit()

    mock_get.return_value.status_code = 200
    mock_get.return_value.content = make_clean_pdf()

    process_pdf_for_document(conn, doc_id=6, redaction_markers=["[REDACTED]"])
    conn.commit()

    fp = conn.execute(
        "SELECT * FROM document_fingerprints WHERE doc_id = 6"
    ).fetchone()
    assert fp is None


@patch("stages.pdf_processor._download_pdf")
def test_process_pdf_updates_existing_fingerprint(mock_get, conn):
    """If a fingerprint already exists, it should be replaced with one reflecting the new text."""
    seed_doc(conn, 7)
    conn.execute("UPDATE documents SET pdf_url = 'http://example.com/doc7.pdf' WHERE id = 7")
    # Pre-create a fingerprint
    from stages.indexer import clean_text, build_fingerprint
    old_sig = build_fingerprint(clean_text("Some text with in it.", ["[REDACTED]"]))
    from core.db import upsert_fingerprint
    upsert_fingerprint(conn, 7, old_sig, 5)
    conn.commit()

    pdf_bytes = make_pdf_with_soft_redaction("Significant additional recovered content that changes the fingerprint substantially for matching")
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = pdf_bytes

    process_pdf_for_document(conn, doc_id=7, redaction_markers=["[REDACTED]"])
    conn.commit()

    new_fp = conn.execute(
        "SELECT minhash_sig FROM document_fingerprints WHERE doc_id = 7"
    ).fetchone()
    assert new_fp is not None
    assert new_fp["minhash_sig"] != old_sig  # Fingerprint should have changed


# --- parallel processing (_run_pdf_parallel) ---

@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "parallel_test.db")
    init_db(path)
    return path


def _setup_pdf_cache(tmp_path, batch, filenames, pdf_bytes):
    """Write PDFs to a local cache dir so process workers find them without network."""
    cache_dir = tmp_path / "pdf_cache" / batch
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fn in filenames:
        (cache_dir / fn).write_bytes(pdf_bytes)
    return tmp_path / "pdf_cache"


def test_run_pdf_parallel_processes_all_docs(db_path, tmp_path):
    """All docs should be marked pdf_processed=1 after parallel run."""
    from unobfuscator import _run_pdf_parallel

    batch = "VOL00001"
    conn = get_connection(db_path)
    filenames = []
    for i in range(20):
        fn = f"par{i}.pdf"
        filenames.append(fn)
        seed_doc(conn, f"par-{i}", filename=fn, batch=batch)
        conn.execute(
            "UPDATE documents SET pdf_url = ? WHERE id = ?",
            (f"http://example.com/{i}.pdf", f"par-{i}"),
        )
    conn.commit()

    cache_dir = _setup_pdf_cache(tmp_path, batch, filenames, make_clean_pdf())
    pdf_docs = [{"id": f"par-{i}", "pdf_url": f"http://example.com/{i}.pdf",
                 "release_batch": batch, "original_filename": f"par{i}.pdf"}
                for i in range(20)]

    _run_pdf_parallel(pdf_docs, [], db_path, num_workers=4,
                      pdf_cache_dir=cache_dir)

    processed = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE pdf_processed = 1"
    ).fetchone()[0]
    conn.close()
    assert processed == 20


def test_run_pdf_parallel_uses_separate_processes(db_path, tmp_path):
    """Workers run in separate processes (not threads) for PyMuPDF safety."""
    from unobfuscator import _run_pdf_parallel

    batch = "VOL00001"
    conn = get_connection(db_path)
    filenames = []
    for i in range(4):
        fn = f"proc{i}.pdf"
        filenames.append(fn)
        seed_doc(conn, f"proc-{i}", filename=fn, batch=batch)
        conn.execute(
            "UPDATE documents SET pdf_url = ? WHERE id = ?",
            (f"http://example.com/proc{i}.pdf", f"proc-{i}"),
        )
    conn.commit()
    conn.close()

    cache_dir = _setup_pdf_cache(tmp_path, batch, filenames, make_clean_pdf())
    pdf_docs = [{"id": f"proc-{i}", "pdf_url": f"http://example.com/proc{i}.pdf",
                 "release_batch": batch, "original_filename": f"proc{i}.pdf"}
                for i in range(4)]

    # Verify ProcessPoolExecutor is used (not ThreadPoolExecutor)
    from concurrent.futures import ProcessPoolExecutor
    with patch("unobfuscator.ProcessPoolExecutor", wraps=ProcessPoolExecutor) as mock_ppe:
        _run_pdf_parallel(pdf_docs, [], db_path, num_workers=2,
                          pdf_cache_dir=cache_dir)
    mock_ppe.assert_called_once_with(max_workers=2)


def test_run_pdf_parallel_handles_worker_errors(db_path, tmp_path):
    """A failing worker should not crash the pool or block other workers."""
    from unobfuscator import _run_pdf_parallel

    batch = "VOL00001"
    conn = get_connection(db_path)
    filenames = []
    for i in range(5):
        fn = f"err{i}.pdf"
        filenames.append(fn)
        seed_doc(conn, f"err-{i}", filename=fn, batch=batch)
        conn.execute(
            "UPDATE documents SET pdf_url = ? WHERE id = ?",
            (f"http://example.com/err{i}.pdf", f"err-{i}"),
        )
    conn.commit()

    # Write valid PDFs for all except err2 (write corrupt bytes)
    cache_dir = tmp_path / "pdf_cache" / batch
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        fn = f"err{i}.pdf"
        if i == 2:
            # Don't write a cache file — and the URL won't work either,
            # so the worker returns ok=False
            pass
        else:
            (cache_dir / fn).write_bytes(make_clean_pdf())

    pdf_docs = [{"id": f"err-{i}", "pdf_url": f"http://invalid.test/err{i}.pdf",
                 "release_batch": batch, "original_filename": f"err{i}.pdf"}
                for i in range(5)]

    _run_pdf_parallel(pdf_docs, [], db_path, num_workers=3,
                      pdf_cache_dir=tmp_path / "pdf_cache")

    processed = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE pdf_processed = 1"
    ).fetchone()[0]
    conn.close()
    assert processed == 5  # all marked processed (including failed one)


def test_run_pdf_parallel_concurrent_writes_no_corruption(db_path, tmp_path):
    """Soft redaction text recovered in workers is correctly written to DB by main process."""
    from unobfuscator import _run_pdf_parallel

    batch = "VOL00001"
    conn = get_connection(db_path)
    filenames = []
    for i in range(10):
        fn = f"wr{i}.pdf"
        filenames.append(fn)
        seed_doc(conn, f"wr-{i}", filename=fn, batch=batch)
        conn.execute(
            "UPDATE documents SET pdf_url = ? WHERE id = ?",
            (f"http://example.com/wr{i}.pdf", f"wr-{i}"),
        )
    conn.commit()

    cache_dir = _setup_pdf_cache(
        tmp_path, batch, filenames,
        make_pdf_with_soft_redaction(
            "Hidden text that should be recovered from the redaction overlay"
        ),
    )
    pdf_docs = [{"id": f"wr-{i}", "pdf_url": f"http://example.com/wr{i}.pdf",
                 "release_batch": batch, "original_filename": f"wr{i}.pdf"}
                for i in range(10)]

    _run_pdf_parallel(pdf_docs, ["[REDACTED]"], db_path, num_workers=4,
                      pdf_cache_dir=cache_dir)

    # All should have soft redaction text appended
    for i in range(10):
        row = conn.execute(
            "SELECT extracted_text FROM documents WHERE id = ?", (f"wr-{i}",)
        ).fetchone()
        assert "[SOFT_REDACTION_RECOVERED]" in row["extracted_text"]

    processed = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE pdf_processed = 1"
    ).fetchone()[0]
    conn.close()
    assert processed == 10
