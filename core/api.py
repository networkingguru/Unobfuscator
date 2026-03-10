import duckdb
from typing import Optional

# Jmail API Parquet URLs (read-only via DuckDB — no local download needed)
JMAIL_DOCS_META_URL = "https://data.jmail.world/v1/documents.parquet"
JMAIL_DOCS_TEXT_URL = "https://data.jmail.world/v1/documents_text/shard_*.parquet"
JMAIL_BATCHES_URL = "https://data.jmail.world/v1/release_batches.parquet"
JMAIL_PEOPLE_URL = "https://data.jmail.world/v1/people.parquet"


def fetch_release_batches() -> list[str]:
    """Return list of all known release batch IDs from Jmail."""
    with duckdb.connect() as conn:
        df = conn.execute(
            f"SELECT DISTINCT batch_id FROM read_parquet('{JMAIL_BATCHES_URL}')"
        ).fetchdf()
    return df["batch_id"].tolist()


def fetch_documents_metadata(batch_id: Optional[str] = None) -> list[dict]:
    """Return document metadata records, optionally filtered by batch."""
    query = f"""
        SELECT id, source, release_batch, original_filename,
               page_count, size, document_description, has_thumbnail
        FROM read_parquet('{JMAIL_DOCS_META_URL}')
    """
    if batch_id:
        query += f" WHERE release_batch = '{batch_id}'"

    with duckdb.connect() as conn:
        df = conn.execute(query).fetchdf()

    return df.rename(columns={
        "size": "size_bytes",
        "document_description": "description"
    }).to_dict(orient="records")


def fetch_document_text(doc_id: int) -> Optional[str]:
    """Return extracted text for a single document ID."""
    with duckdb.connect() as conn:
        df = conn.execute(f"""
            SELECT id, extracted_text
            FROM read_parquet('{JMAIL_DOCS_TEXT_URL}')
            WHERE id = {doc_id}
        """).fetchdf()
    if df.empty:
        return None
    return df.iloc[0]["extracted_text"]


def fetch_documents_text_batch(doc_ids: list[int]) -> dict[int, str]:
    """Return {doc_id: extracted_text} for a list of document IDs."""
    ids_str = ", ".join(str(i) for i in doc_ids)
    with duckdb.connect() as conn:
        df = conn.execute(f"""
            SELECT id, extracted_text
            FROM read_parquet('{JMAIL_DOCS_TEXT_URL}')
            WHERE id IN ({ids_str})
        """).fetchdf()
    return dict(zip(df["id"], df["extracted_text"]))


def search_documents_by_keyword(keyword: str) -> list[dict]:
    """Return document metadata where description or text contains keyword."""
    with duckdb.connect() as conn:
        df = conn.execute(f"""
            SELECT id, source, release_batch, original_filename,
                   page_count, size, document_description
            FROM read_parquet('{JMAIL_DOCS_META_URL}')
            WHERE LOWER(document_description) LIKE LOWER('%{keyword}%')
        """).fetchdf()
    return df.rename(columns={
        "size": "size_bytes",
        "document_description": "description"
    }).to_dict(orient="records")


def fetch_person_document_ids(name: str) -> list[int]:
    """Return document IDs associated with a named person via Jmail people dataset."""
    with duckdb.connect() as conn:
        df = conn.execute(f"""
            SELECT DISTINCT document_id
            FROM read_parquet('{JMAIL_PEOPLE_URL}')
            WHERE LOWER(name) LIKE LOWER('%{name}%')
        """).fetchdf()
    return df["document_id"].tolist() if not df.empty else []
