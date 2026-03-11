import sqlite3
import json
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    release_batch TEXT,
    original_filename TEXT,
    page_count INTEGER,
    size_bytes INTEGER,
    description TEXT,
    extracted_text TEXT,
    pdf_url TEXT,
    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    text_processed BOOLEAN DEFAULT 0,
    pdf_processed BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS document_fingerprints (
    doc_id INTEGER PRIMARY KEY REFERENCES documents(id),
    minhash_sig BLOB NOT NULL,
    shingle_count INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS match_groups (
    group_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    merged BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS match_group_members (
    group_id INTEGER REFERENCES match_groups(group_id),
    doc_id INTEGER UNIQUE REFERENCES documents(id),
    similarity REAL,
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_id, doc_id)
);

CREATE TABLE IF NOT EXISTS merge_results (
    group_id INTEGER PRIMARY KEY REFERENCES match_groups(group_id),
    merged_text TEXT,
    recovered_count INTEGER DEFAULT 0,
    previous_recovered_count INTEGER DEFAULT 0,
    total_redacted INTEGER DEFAULT 0,
    source_doc_ids TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    output_generated BOOLEAN DEFAULT 0,
    recovered_segments TEXT,
    soft_recovered_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS release_batches (
    batch_id TEXT PRIMARY KEY,
    first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    fully_indexed BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    payload TEXT,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_document(conn, doc: dict) -> None:
    # INSERT OR IGNORE preserves text_processed/pdf_processed flags on re-index.
    # Then UPDATE refreshes metadata fields without touching progress flags.
    # pdf_url is optional — callers that don't supply it get NULL.
    params = {**doc, "pdf_url": doc.get("pdf_url")}
    conn.execute("""
        INSERT OR IGNORE INTO documents
        (id, source, release_batch, original_filename, page_count,
         size_bytes, description, extracted_text, pdf_url, indexed_at)
        VALUES (:id, :source, :release_batch, :original_filename, :page_count,
                :size_bytes, :description, :extracted_text, :pdf_url, CURRENT_TIMESTAMP)
    """, params)
    conn.execute("""
        UPDATE documents SET
            source = :source, release_batch = :release_batch,
            original_filename = :original_filename, page_count = :page_count,
            size_bytes = :size_bytes, description = :description,
            extracted_text = :extracted_text, pdf_url = :pdf_url
        WHERE id = :id
          AND text_processed = 0 AND pdf_processed = 0
    """, params)


def get_unprocessed_documents(conn, limit: int = 1000) -> list:
    return conn.execute("""
        SELECT id, extracted_text, release_batch, source
        FROM documents WHERE text_processed = 0 LIMIT ?
    """, (limit,)).fetchall()


def mark_text_processed(conn, doc_id: int) -> None:
    conn.execute("UPDATE documents SET text_processed = 1 WHERE id = ?", (doc_id,))


def upsert_fingerprint(conn, doc_id: int, sig: bytes, shingle_count: int) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO document_fingerprints (doc_id, minhash_sig, shingle_count)
        VALUES (?, ?, ?)
    """, (doc_id, sig, shingle_count))


def get_all_fingerprints(conn) -> list:
    return conn.execute(
        "SELECT doc_id, minhash_sig FROM document_fingerprints"
    ).fetchall()


def create_match_group(conn) -> int:
    cursor = conn.execute("INSERT INTO match_groups (merged) VALUES (0)")
    return cursor.lastrowid


def add_group_member(conn, group_id: int, doc_id: int, similarity: float) -> None:
    conn.execute("""
        INSERT OR IGNORE INTO match_group_members (group_id, doc_id, similarity)
        VALUES (?, ?, ?)
    """, (group_id, doc_id, similarity))


def get_doc_group(conn, doc_id: int) -> Optional[int]:
    row = conn.execute(
        "SELECT group_id FROM match_group_members WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return row["group_id"] if row else None


def merge_groups(conn, group_id_keep: int, group_id_remove: int) -> None:
    conn.execute("""
        UPDATE OR IGNORE match_group_members SET group_id = ?
        WHERE group_id = ?
    """, (group_id_keep, group_id_remove))
    conn.execute("DELETE FROM match_groups WHERE group_id = ?", (group_id_remove,))


def upsert_merge_result(conn, group_id: int, merged_text: str,
                        recovered_count: int, total_redacted: int,
                        source_doc_ids: list,
                        recovered_segments: list = None,
                        soft_recovered_count: int = 0) -> None:
    existing = conn.execute(
        "SELECT recovered_count FROM merge_results WHERE group_id = ?", (group_id,)
    ).fetchone()
    prev_count = existing["recovered_count"] if existing else 0
    conn.execute("""
        INSERT OR REPLACE INTO merge_results
        (group_id, merged_text, recovered_count, previous_recovered_count,
         total_redacted, source_doc_ids, recovered_segments,
         soft_recovered_count, updated_at, output_generated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0)
    """, (group_id, merged_text, recovered_count, prev_count,
          total_redacted, json.dumps(source_doc_ids),
          json.dumps(recovered_segments or []), soft_recovered_count))


def get_config(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_config(conn, key: str, value) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )


def get_document_for_pdf(conn, doc_id: int) -> Optional[dict]:
    """Return id, source, release_batch, original_filename, pdf_url for a document."""
    row = conn.execute(
        "SELECT id, source, release_batch, original_filename, pdf_url "
        "FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    return dict(row) if row else None


def append_soft_redaction_text(conn, doc_id: int, recovered_text: str) -> None:
    """Append recovered soft-redaction text to a document's extracted_text."""
    conn.execute(
        "UPDATE documents SET extracted_text = extracted_text || ? WHERE id = ?",
        (f"\n\n[SOFT_REDACTION_RECOVERED]\n{recovered_text}", doc_id)
    )


def mark_pdf_processed(conn, doc_id: int) -> None:
    """Mark a document's PDF as processed."""
    conn.execute("UPDATE documents SET pdf_processed = 1 WHERE id = ?", (doc_id,))


def get_merge_result(conn, group_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM merge_results WHERE group_id = ?", (group_id,)
    ).fetchone()
    return dict(row) if row else None


def get_documents_by_ids(conn, doc_ids: list[int]) -> list[dict]:
    if not doc_ids:
        return []
    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"SELECT id, source, release_batch, original_filename, extracted_text "
        f"FROM documents WHERE id IN ({placeholders})", doc_ids
    ).fetchall()
    return [dict(r) for r in rows]


def mark_output_generated(conn, group_id: int) -> None:
    conn.execute(
        "UPDATE merge_results SET output_generated = 1, updated_at = CURRENT_TIMESTAMP "
        "WHERE group_id = ?", (group_id,)
    )


def get_pending_output_groups(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT group_id, merged_text, recovered_count, total_redacted,
               source_doc_ids, recovered_segments, soft_recovered_count
        FROM merge_results
        WHERE output_generated = 0 AND recovered_count > 0
    """).fetchall()
    return [dict(r) for r in rows]


def get_pending_pdf_documents(conn, limit: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id FROM documents WHERE pdf_processed = 0 AND pdf_url IS NOT NULL LIMIT ?",
        (limit,)
    ).fetchall()
    return [dict(row) for row in rows]


def get_known_batch_ids(conn) -> set:
    return {r["batch_id"] for r in conn.execute(
        "SELECT batch_id FROM release_batches"
    ).fetchall()}


def insert_release_batch(conn, batch_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO release_batches (batch_id, fully_indexed) VALUES (?, 0)",
        (batch_id,)
    )


def reset_group_merged(conn, group_id: int) -> None:
    """Reset merged flag so run_merger will reprocess this group."""
    conn.execute(
        "UPDATE match_groups SET merged = 0 WHERE group_id = ?", (group_id,)
    )
