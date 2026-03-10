import pytest
from unittest.mock import patch, MagicMock
from core.api import (
    fetch_release_batches, fetch_documents_metadata, fetch_document_text,
    fetch_documents_text_batch, search_documents_by_keyword, fetch_person_document_ids
)


def make_mock_relation(rows):
    """Helper: mock a DuckDB relation that returns given rows as dicts."""
    mock = MagicMock()
    mock.fetchdf.return_value = rows
    return mock


@patch("core.api.duckdb.connect")
def test_fetch_release_batches_returns_batch_ids(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        {"batch_id": ["VOL00008", "VOL00009"]}
    )
    batches = fetch_release_batches()
    assert "VOL00008" in batches
    assert "VOL00009" in batches


@patch("core.api.duckdb.connect")
def test_fetch_documents_metadata_returns_list_of_dicts(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{
        "id": 1, "source": "doj", "release_batch": "VOL00008",
        "original_filename": "test.pdf", "page_count": 3,
        "size": 1000, "document_description": "A document",
        "has_thumbnail": False
    }])
    docs = fetch_documents_metadata(batch_id="VOL00008")
    assert len(docs) == 1
    assert docs[0]["id"] == 1
    assert docs[0]["source"] == "doj"


@patch("core.api.duckdb.connect")
def test_fetch_document_text_returns_extracted_text(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{
        "id": 1, "extracted_text": "This is the full document text."
    }])
    result = fetch_document_text(doc_id=1)
    assert result == "This is the full document text."


@patch("core.api.duckdb.connect")
def test_fetch_document_text_returns_none_for_missing(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        columns=["id", "extracted_text"]
    )
    result = fetch_document_text(doc_id=9999)
    assert result is None


@patch("core.api.duckdb.connect")
def test_fetch_documents_text_batch_returns_id_to_text_map(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([
        {"id": 1, "extracted_text": "Text of doc 1"},
        {"id": 2, "extracted_text": "Text of doc 2"},
    ])
    result = fetch_documents_text_batch([1, 2])
    assert result == {1: "Text of doc 1", 2: "Text of doc 2"}


@patch("core.api.duckdb.connect")
def test_search_documents_by_keyword_returns_matching_docs(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{
        "id": 5, "source": "doj", "release_batch": "VOL00008",
        "original_filename": "flight.pdf", "page_count": 2,
        "size": 900, "document_description": "Flight log"
    }])
    docs = search_documents_by_keyword("flight")
    assert len(docs) == 1
    assert docs[0]["id"] == 5


@patch("core.api.duckdb.connect")
def test_fetch_person_document_ids_returns_list(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        {"document_id": [10, 20, 30]}
    )
    result = fetch_person_document_ids("Trump")
    assert result == [10, 20, 30]
