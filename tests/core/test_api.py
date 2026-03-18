import pytest
from unittest.mock import patch, MagicMock
from core.api import (
    fetch_release_batches, fetch_documents_metadata, fetch_document_text,
    fetch_documents_text_batch, search_documents_by_keyword, fetch_person_document_ids
)
from core.api import resolve_shard, SHARD_MAP, DEFAULT_SHARD


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
        {"release_batch": ["VOL00008", "VOL00009"]}
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
        "id": "EFTA00001.pdf", "source": "doj", "release_batch": "VOL00008",
        "original_filename": "EFTA00001.pdf", "page_count": 3,
        "size": 1000, "document_description": "A document",
        "has_thumbnail": False, "source_url": "https://example.com/doc.pdf"
    }])
    docs = fetch_documents_metadata(batch_id="VOL00008")
    assert len(docs) == 1
    assert docs[0]["id"] == "EFTA00001.pdf"
    assert docs[0]["source"] == "doj"


@patch("core.api.duckdb.connect")
def test_fetch_documents_metadata_uses_parameterized_query(mock_connect):
    """Verify batch_id is passed as a parameter, not interpolated into the query."""
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        columns=["id", "source", "release_batch", "original_filename",
                 "page_count", "size", "document_description", "has_thumbnail", "source_url"]
    )
    fetch_documents_metadata(batch_id="VOL00008")
    call_args = mock_conn.execute.call_args
    # Second argument must be a list/tuple containing the batch_id
    assert call_args is not None
    args = call_args[0]
    assert len(args) == 2, "execute() must be called with (query, params)"
    params = args[1]
    assert isinstance(params, (list, tuple)), "params must be a list or tuple"
    assert "VOL00008" in params
    # batch_id must NOT be interpolated into the query string
    assert "VOL00008" not in args[0]


@patch("core.api.duckdb.connect")
def test_fetch_documents_metadata_no_filter_passes_empty_params(mock_connect):
    """When no batch_id given, execute() is still called with an empty params list."""
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        columns=["id", "source", "release_batch", "original_filename",
                 "page_count", "size", "document_description", "has_thumbnail", "source_url"]
    )
    fetch_documents_metadata()
    call_args = mock_conn.execute.call_args
    args = call_args[0]
    assert len(args) == 2
    assert args[1] == []


@patch("core.api.duckdb.connect")
def test_fetch_document_text_uses_parameterized_query(mock_connect):
    """Verify doc_id is passed as a parameter, not interpolated into the query."""
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{
        "id": "EFTA00042.pdf", "extracted_text": "Some text"
    }])
    fetch_document_text(doc_id="EFTA00042.pdf", batch_id="VOL00001")
    call_args = mock_conn.execute.call_args
    args = call_args[0]
    assert len(args) == 2, "execute() must be called with (query, params)"
    params = args[1]
    assert isinstance(params, (list, tuple))
    assert "EFTA00042.pdf" in params
    # doc_id must NOT be interpolated into the query string
    assert "EFTA00042.pdf" not in args[0]


@patch("core.api.duckdb.connect")
def test_fetch_document_text_returns_extracted_text(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{
        "id": "EFTA00001.pdf", "extracted_text": "This is the full document text."
    }])
    result = fetch_document_text(doc_id="EFTA00001.pdf", batch_id="VOL00001")
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
    result = fetch_document_text(doc_id="MISSING.pdf", batch_id="VOL00001")
    assert result is None


@patch("core.api.duckdb.connect")
def test_fetch_documents_text_batch_returns_id_to_text_map(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([
        {"id": "EFTA00001.pdf", "extracted_text": "Text of doc 1"},
        {"id": "EFTA00002.pdf", "extracted_text": "Text of doc 2"},
    ])
    result = fetch_documents_text_batch(["EFTA00001.pdf", "EFTA00002.pdf"], "VOL00001")
    assert result == {"EFTA00001.pdf": "Text of doc 1", "EFTA00002.pdf": "Text of doc 2"}


def test_fetch_documents_text_batch_empty_list_returns_empty_dict():
    """Empty input must return {} immediately without touching DuckDB."""
    result = fetch_documents_text_batch([], "VOL00001")
    assert result == {}


def test_fetch_documents_text_batch_rejects_non_string_ids():
    """Non-string IDs (e.g. integers) must raise TypeError before any query is executed."""
    with pytest.raises((ValueError, TypeError)):
        fetch_documents_text_batch(["EFTA00001.pdf", 2], "VOL00001")  # type: ignore[list-item]


@patch("core.api.duckdb.connect")
def test_search_documents_by_keyword_returns_matching_docs(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame([{
        "id": "FLIGHT001.pdf", "source": "doj", "release_batch": "VOL00008",
        "original_filename": "FLIGHT001.pdf", "page_count": 2,
        "size": 900, "document_description": "Flight log"
    }])
    docs = search_documents_by_keyword("flight")
    assert len(docs) == 1
    assert docs[0]["id"] == "FLIGHT001.pdf"


@patch("core.api.duckdb.connect")
def test_search_documents_by_keyword_uses_parameterized_query(mock_connect):
    """Verify keyword is passed as a %-wrapped parameter, not interpolated into the query."""
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        columns=["id", "source", "release_batch", "original_filename",
                 "page_count", "size", "document_description"]
    )
    search_documents_by_keyword("flight")
    call_args = mock_conn.execute.call_args
    args = call_args[0]
    assert len(args) == 2, "execute() must be called with (query, params)"
    params = args[1]
    assert isinstance(params, (list, tuple))
    # The param should be the %-wrapped pattern
    assert "%flight%" in params
    # The raw keyword must NOT appear directly in the query string
    assert "flight" not in args[0]


@patch("core.api.duckdb.connect")
def test_fetch_person_document_ids_returns_list(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        {"doc_id": ["EFTA00010.pdf", "EFTA00020.pdf", "EFTA00030.pdf"]}
    )
    result = fetch_person_document_ids("Trump")
    assert result == ["EFTA00010.pdf", "EFTA00020.pdf", "EFTA00030.pdf"]


def test_resolve_shard_known_batch():
    assert resolve_shard("VOL00008-2") == "VOL00008"
    assert resolve_shard("VOL00008-OFFICIAL-DOJ-LATEST") == "VOL00008"
    assert resolve_shard("VOL00009") == "VOL00009"
    assert resolve_shard("DataSet11") == "DataSet11"

def test_resolve_shard_unknown_batch_returns_other():
    assert resolve_shard("VOL00001") == "other"
    assert resolve_shard("DOJ-COURT") == "other"
    assert resolve_shard("batch-3") == "other"

@patch("core.api.duckdb.connect")
def test_fetch_documents_text_batch_uses_resolved_shard(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(columns=["id", "extracted_text"])
    fetch_documents_text_batch(["doc1.pdf"], "VOL00001")
    query = mock_conn.execute.call_args_list[-1][0][0]
    assert "other.parquet" in query
    assert "VOL00001.parquet" not in query

@patch("core.api.duckdb.connect")
def test_fetch_documents_text_batch_vol00008_2_uses_vol00008_shard(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(columns=["id", "extracted_text"])
    fetch_documents_text_batch(["doc1.pdf"], "VOL00008-2")
    query = mock_conn.execute.call_args_list[-1][0][0]
    assert "VOL00008.parquet" in query
    assert "VOL00008-2.parquet" not in query

@patch("core.api.duckdb.connect")
def test_fetch_document_text_uses_resolved_shard(mock_connect):
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(columns=["id", "extracted_text"])
    fetch_document_text(doc_id="doc1.pdf", batch_id="DOJ-COURT")
    query = mock_conn.execute.call_args_list[-1][0][0]
    assert "other.parquet" in query


@patch("core.api.duckdb.connect")
def test_fetch_person_document_ids_uses_parameterized_query(mock_connect):
    """Verify name is passed as a %-wrapped parameter, not interpolated into the query."""
    import pandas as pd
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = lambda s: mock_conn
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame(
        {"doc_id": ["EFTA00010.pdf"]}
    )
    fetch_person_document_ids("Trump")
    call_args = mock_conn.execute.call_args
    args = call_args[0]
    assert len(args) == 2, "execute() must be called with (query, params)"
    params = args[1]
    assert isinstance(params, (list, tuple))
    assert "%Trump%" in params
    # The raw name must NOT appear directly in the query string
    assert "Trump" not in args[0]
