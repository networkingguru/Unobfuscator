import pytest
import yaml
from unittest.mock import patch
from core.db import init_db, get_connection, insert_release_batch
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
@patch("stages.indexer.fetch_document_text")
def test_run_one_cycle_completes_without_error(mock_text, mock_meta, conn, cfg):
    """Smoke test: one full cycle with a seeded batch does not raise."""
    insert_release_batch(conn, "VOL00001")
    conn.commit()

    mock_meta.return_value = [{
        "id": 1, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "a.pdf", "page_count": 1,
        "size_bytes": 100, "description": ""
    }]
    mock_text.return_value = (
        "From: a@a.com\nTo: b@b.com\nDate: 2002-01-01\nSubject: Test\n\n"
        "The attendee was [REDACTED] at the location."
    )

    _run_one_cycle(conn, cfg)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1
