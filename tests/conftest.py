import pytest
from pathlib import Path
from core.db import init_db, get_connection, upsert_document

SAMPLE_DOCS = [
    {
        "id": 1, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "doc1.pdf", "page_count": 3,
        "size_bytes": 5000, "description": "Flight log document",
        "extracted_text": (
            "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
            "Date: 2002-01-15\nSubject: Flight arrangements\n\n"
            "The passenger list included [REDACTED] and several others. "
            "The flight departed from Palm Beach at 9am."
        )
    },
    {
        "id": 2, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "doc2.pdf", "page_count": 3,
        "size_bytes": 5100, "description": "Flight log document duplicate",
        "extracted_text": (
            "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
            "Date: 2002-01-15\nSubject: Flight arrangements\n\n"
            "The passenger list included Donald Trump and several others. "
            "The flight departed from [REDACTED] at 9am."
        )
    },
    {
        "id": 3, "source": "house_oversight", "release_batch": "DataSet11",
        "original_filename": "unrelated.pdf", "page_count": 1,
        "size_bytes": 800, "description": "Unrelated administrative document",
        "extracted_text": "This document has completely different content about budgets."
    }
]


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


@pytest.fixture
def populated_conn(conn):
    """Connection with sample documents pre-loaded."""
    for doc in SAMPLE_DOCS:
        upsert_document(conn, doc)
    conn.commit()
    return conn
