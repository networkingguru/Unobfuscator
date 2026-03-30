import json
import pytest
import yaml
import fitz
from unittest.mock import patch, call
from click.testing import CliRunner
from core.db import (
    init_db, get_connection, insert_release_batch, upsert_document,
    create_match_group, add_group_member, get_config,
)
from core.queue import enqueue, get_queue_stats
from unobfuscator import _run_one_cycle, cli


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


@patch("stages.pdf_processor._download_pdf")
@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_run_one_cycle_processes_multiple_pdfs(mock_text_batch, mock_meta, mock_dl, conn, cfg):
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
    doc = fitz.open()
    doc.new_page()
    clean_pdf = doc.tobytes()
    mock_dl.return_value.content = clean_pdf

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


@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_index_jobs_marked_done_after_cycle(mock_text_batch, mock_meta, conn, cfg):
    """After a cycle, any pending 'index' queue jobs are marked done."""
    # Enqueue two index jobs before running the cycle.
    enqueue(conn, stage="index", payload={"doc_id": 1})
    enqueue(conn, stage="index", payload={"doc_id": 2})
    conn.commit()

    # No documents to actually index — empty API returns.
    mock_meta.return_value = []
    mock_text_batch.return_value = {}

    stats_before = get_queue_stats(conn)
    assert stats_before["pending"] == 2

    _run_one_cycle(conn, cfg)
    conn.commit()

    stats_after = get_queue_stats(conn)
    assert stats_after["pending"] == 0, (
        f"Expected 0 pending jobs after cycle, got {stats_after['pending']}"
    )
    assert stats_after["running"] == 0, (
        f"Expected 0 running jobs after cycle, got {stats_after['running']}"
    )
    assert stats_after["done"] == 2, (
        f"Expected 2 done jobs after cycle, got {stats_after['done']}"
    )


@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_documents_text_batch")
def test_merge_job_resets_group_and_reruns_merger(mock_text_batch, mock_meta, conn, cfg):
    """A pending 'merge' queue job causes the named group to be re-merged."""
    # Seed two documents with complementary text so the merger can recover a redaction.
    base_text = "The witness was [REDACTED] at the event."
    donor_text = "The witness was Maxwell at the event. The date was [REDACTED]."

    for doc_id, text in ((20, base_text), (21, donor_text)):
        upsert_document(conn, {
            "id": doc_id, "source": "doj", "release_batch": "VOL00010",
            "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
            "size_bytes": 100, "description": "", "extracted_text": text,
        })
        conn.execute(
            "UPDATE documents SET text_processed = 1 WHERE id = ?", (doc_id,)
        )

    # Create a match group containing both documents that has already been merged.
    group_id = create_match_group(conn)
    add_group_member(conn, group_id, 20, similarity=0.9)
    add_group_member(conn, group_id, 21, similarity=0.9)
    conn.execute("UPDATE match_groups SET merged = 1 WHERE group_id = ?", (group_id,))
    conn.commit()

    # Enqueue a merge job targeting this group (simulating a pdf_processor re-merge request).
    enqueue(conn, stage="merge", payload={"group_id": group_id})
    conn.commit()

    mock_meta.return_value = []
    mock_text_batch.return_value = {}

    _run_one_cycle(conn, cfg)
    conn.commit()

    # The merge job should now be done (not pending).
    stats = get_queue_stats(conn)
    assert stats["pending"] == 0, (
        f"Expected 0 pending jobs after cycle, got {stats['pending']}"
    )

    # The group should have been re-merged and have a merge_result entry.
    merge_row = conn.execute(
        "SELECT recovered_count FROM merge_results WHERE group_id = ?", (group_id,)
    ).fetchone()
    assert merge_row is not None, "Expected a merge_result row for the re-merged group"
    assert merge_row["recovered_count"] >= 1, (
        f"Expected at least 1 recovered redaction, got {merge_row['recovered_count']}"
    )


def test_config_set_persists_value(tmp_path):
    """config set stores a key/value in the DB and can be retrieved via get_config."""
    db_path = str(tmp_path / "cfg_test.db")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump({"db_path": db_path})
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_file), "config", "set", "output_dir", "/tmp/out"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI exited with {result.exit_code}:\n{result.output}"
    assert "Config set:" in result.output

    conn = get_connection(db_path)
    stored = get_config(conn, "output_dir")
    conn.close()
    assert stored == "/tmp/out", f"Expected '/tmp/out', got {stored!r}"


def test_run_one_cycle_calls_text_recovery(tmp_path):
    """Phase 5.5 must be called between PDF processing and output generation."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = get_connection(db_path)
    cfg = {
        "redaction_markers": ["[REDACTED]"],
        "matching": {"min_overlap_chars": 200, "similarity_threshold": 0.70,
                     "email_header_min_matches": 2},
        "output_dir": str(tmp_path / "output"),
        "ocr": {"min_words_per_page": 50},
    }
    with patch("unobfuscator.run_indexer_batch"), \
         patch("unobfuscator.run_phase0_email_fastpath", return_value=[]), \
         patch("unobfuscator.run_phase2_lsh_candidates", return_value=[]), \
         patch("unobfuscator.run_phase3_verify_and_group"), \
         patch("unobfuscator.run_merger", return_value=0), \
         patch("unobfuscator.run_cross_group_merger", return_value=0), \
         patch("unobfuscator.dequeue", return_value=None), \
         patch("unobfuscator.run_text_recovery") as mock_tr, \
         patch("unobfuscator.run_output_generator", return_value=0), \
         patch("unobfuscator.get_pending_pdf_documents", return_value=[]):
        from unobfuscator import _run_one_cycle
        _run_one_cycle(conn, cfg)
        mock_tr.assert_called_once()


def test_config_set_overwrites_existing_value(tmp_path):
    """config set on an existing key replaces the previous value."""
    db_path = str(tmp_path / "cfg_test.db")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"db_path": db_path}))

    runner = CliRunner()
    runner.invoke(
        cli,
        ["--config", str(config_file), "config", "set", "some_key", "first_value"],
        catch_exceptions=False,
    )
    runner.invoke(
        cli,
        ["--config", str(config_file), "config", "set", "some_key", "second_value"],
        catch_exceptions=False,
    )

    conn = get_connection(db_path)
    stored = get_config(conn, "some_key")
    conn.close()
    assert stored == "second_value", f"Expected 'second_value', got {stored!r}"


def test_backfill_fingerprints_rebuilds_stale_fingerprints(tmp_path):
    """backfill-fingerprints should rebuild fingerprints for docs with soft-redaction text."""
    db_path = str(tmp_path / "bf_test.db")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump({"db_path": db_path, "redaction_markers": ["[REDACTED]"]})
    )

    init_db(db_path)
    conn = get_connection(db_path)

    # Seed a document that has soft-redaction text but no fingerprint (simulating the bug)
    long_text = (
        "The witness testified about a meeting with [REDACTED] on January fifth "
        "two thousand and two at the residence located on the island. "
        "\n\n[SOFT_REDACTION_RECOVERED]\n"
        "Additional recovered content from the soft redaction overlay in the PDF."
    )
    upsert_document(conn, {
        "id": "doc_sr_1", "source": "doj", "release_batch": "VOL00001",
        "original_filename": "sr.pdf", "page_count": 1,
        "size_bytes": 1000, "description": "", "extracted_text": long_text,
    })
    conn.commit()

    # Verify no fingerprint exists yet
    fp = conn.execute(
        "SELECT * FROM document_fingerprints WHERE doc_id = 'doc_sr_1'"
    ).fetchone()
    assert fp is None
    conn.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_file), "backfill-fingerprints"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI exited with {result.exit_code}:\n{result.output}"
    assert "fingerprints rebuilt" in result.output

    # Verify fingerprint was created
    conn = get_connection(db_path)
    fp = conn.execute(
        "SELECT * FROM document_fingerprints WHERE doc_id = 'doc_sr_1'"
    ).fetchone()
    conn.close()
    assert fp is not None
    assert fp["shingle_count"] > 0


def test_backfill_fingerprints_skips_when_no_soft_redactions(tmp_path):
    """backfill-fingerprints should report nothing to do when no docs have the marker."""
    db_path = str(tmp_path / "bf_empty.db")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"db_path": db_path}))

    init_db(db_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_file), "backfill-fingerprints"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "No documents" in result.output
