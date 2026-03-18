import math
import duckdb
from typing import Optional


def _nan_to_none(record: dict) -> dict:
    """Replace float NaN values (from pandas null strings) with None for sqlite3."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in record.items()}

# Jmail API Parquet URLs (read-only via DuckDB — no local download needed)
JMAIL_DOCS_META_URL = "https://data.jmail.world/v1/documents.parquet"
JMAIL_DOCS_FULL_GLOB = "https://data.jmail.world/v1/documents-full/*.parquet"
JMAIL_EMAILS_URL = "https://data.jmail.world/v1/emails.parquet"
JMAIL_PEOPLE_URL = "https://data.jmail.world/v1/people.parquet"

SHARD_MAP = {
    "VOL00008": "VOL00008",
    "VOL00008-2": "VOL00008",
    "VOL00008-OFFICIAL-DOJ-LATEST": "VOL00008",
    "VOL00009": "VOL00009",
    "VOL00010": "VOL00010",
    "DataSet11": "DataSet11",
}
DEFAULT_SHARD = "other"


def resolve_shard(batch_id: str) -> str:
    """Map a release batch to its Jmail documents-full shard filename."""
    return SHARD_MAP.get(batch_id, DEFAULT_SHARD)


def fetch_release_batches() -> list[str]:
    """Return list of all known release batch IDs from Jmail documents metadata."""
    with duckdb.connect() as conn:
        conn.execute("SET force_download=true")
        df = conn.execute(
            f"SELECT DISTINCT release_batch FROM read_parquet('{JMAIL_DOCS_META_URL}') "
            "WHERE release_batch IS NOT NULL"
        ).fetchdf()
    return df["release_batch"].tolist()


def fetch_documents_metadata(batch_id: Optional[str] = None) -> list[dict]:
    """Return document metadata records, optionally filtered by batch."""
    query = f"""
        SELECT id, source, release_batch, original_filename,
               page_count, size, document_description, has_thumbnail, source_url
        FROM read_parquet('{JMAIL_DOCS_META_URL}')
    """
    params: list = []
    if batch_id:
        query += " WHERE release_batch = $1"
        params = [batch_id]

    with duckdb.connect() as conn:
        conn.execute("SET force_download=true")
        df = conn.execute(query, params).fetchdf()

    return [_nan_to_none(r) for r in df.rename(columns={
        "size": "size_bytes",
        "document_description": "description"
    }).to_dict(orient="records")]


def fetch_document_text(doc_id: str, batch_id: str) -> Optional[str]:
    """Return extracted text for a single document ID within a batch."""
    shard = resolve_shard(batch_id)
    url = f"https://data.jmail.world/v1/documents-full/{shard}.parquet"
    with duckdb.connect() as conn:
        conn.execute("SET force_download=true")
        df = conn.execute(f"""
            SELECT id, extracted_text
            FROM read_parquet('{url}')
            WHERE id = $1
        """, [doc_id]).fetchdf()
    if df.empty:
        return None
    return df.iloc[0]["extracted_text"]


def fetch_documents_text_batch(doc_ids: list[str], batch_id: str) -> dict[str, str]:
    """Return {doc_id: extracted_text} for a list of document IDs within a batch.

    NOTE: This function builds the IN-list clause by quoting string IDs rather than
    using DuckDB's $1 parameterized syntax. DuckDB does not support a single parameter
    placeholder for a variable-length IN list. To prevent SQL injection we validate
    that every element is a plain Python str first — any non-string value raises
    TypeError before a query is issued. Real document IDs are filenames containing
    no SQL metacharacters.
    """
    for i in doc_ids:
        if not isinstance(i, str):
            raise TypeError(
                f"doc_id must be str, got {type(i).__name__}: {i!r}"
            )
    if not doc_ids:
        return {}
    shard = resolve_shard(batch_id)
    url = f"https://data.jmail.world/v1/documents-full/{shard}.parquet"
    ids_str = ", ".join(f"'{i}'" for i in doc_ids)
    try:
        with duckdb.connect() as conn:
            conn.execute("SET force_download=true")
            df = conn.execute(f"""
                SELECT id, extracted_text
                FROM read_parquet('{url}')
                WHERE id IN ({ids_str})
            """).fetchdf()
    except duckdb.IOException as e:
        if "HTTP 404" in str(e):
            return {}
        raise
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in zip(df["id"], df["extracted_text"])}


def search_documents_by_keyword(keyword: str) -> list[dict]:
    """Return document metadata where description contains keyword."""
    with duckdb.connect() as conn:
        conn.execute("SET force_download=true")
        df = conn.execute(f"""
            SELECT id, source, release_batch, original_filename,
                   page_count, size, document_description
            FROM read_parquet('{JMAIL_DOCS_META_URL}')
            WHERE LOWER(document_description) LIKE LOWER($1)
        """, [f"%{keyword}%"]).fetchdf()
    return [_nan_to_none(r) for r in df.rename(columns={
        "size": "size_bytes",
        "document_description": "description"
    }).to_dict(orient="records")]


def fetch_person_document_ids(name: str) -> list[str]:
    """Return document IDs from emails where the person appears as a participant."""
    with duckdb.connect() as conn:
        conn.execute("SET force_download=true")
        df = conn.execute(f"""
            SELECT DISTINCT doc_id
            FROM read_parquet('{JMAIL_EMAILS_URL}')
            WHERE LOWER(all_participants) LIKE LOWER($1)
              AND doc_id IS NOT NULL
        """, [f"%{name}%"]).fetchdf()
    return df["doc_id"].tolist() if not df.empty else []
