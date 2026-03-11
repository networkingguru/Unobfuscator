# Unobfuscator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that cross-references 1.41M Epstein archive documents from the Jmail API to recover redacted text via content matching and soft PDF redaction removal, outputting highlighted PDFs with source footnotes.

**Architecture:** A SQLite-backed job queue drives five sequential stages (Index → Match → Merge → PDF Process → Output). DuckDB queries the remote Jmail Parquet files read-only; SQLite stores all local state. A background daemon runs all stages continuously; manual search inserts high-priority jobs that feed into the same shared dataset.

**Tech Stack:** Python 3.11+, click (CLI), duckdb (Jmail API queries), datasketch (MinHash/LSH), pymupdf/fitz (PDF processing), sqlite3 (built-in, state), pyyaml (config), httpx (HTTP), rich (terminal display), pytest + pytest-asyncio (tests)

---

## File Map

| File | Responsibility |
|------|---------------|
| `unobfuscator.py` | CLI entry point — all `click` commands |
| `PIPELINE.md` | Plain-English pseudocode — source of truth for matching logic |
| `config.yaml` | Default user configuration |
| `core/db.py` | All SQLite schema creation and CRUD — no SQL anywhere else |
| `core/queue.py` | Job queue operations (enqueue, dequeue, mark done/failed) |
| `core/api.py` | Jmail API wrapper — DuckDB queries, caching, retries |
| `core/config.py` | Config loading (merges config.yaml with DB overrides) |
| `stages/indexer.py` | Stage 1: fetch docs from Jmail, store in DB, build MinHash fingerprints |
| `stages/matcher.py` | Stage 2: email fast-path + LSH candidate finding + verification + grouping |
| `stages/merger.py` | Stage 3: anchor-phrase merging of match groups |
| `stages/pdf_processor.py` | Stage 4: download PDFs, detect/strip soft redactions |
| `stages/output_generator.py` | Stage 5: write highlighted output PDFs with footnote pages |
| `tests/conftest.py` | Shared pytest fixtures (tmp DB, sample docs, mock API) |
| `tests/core/test_db.py` | DB schema and CRUD tests |
| `tests/core/test_queue.py` | Queue priority and status tests |
| `tests/core/test_api.py` | API wrapper tests (mocked DuckDB) |
| `tests/core/test_config.py` | Config loading and override tests |
| `tests/stages/test_indexer.py` | Indexer fetch, store, fingerprint tests |
| `tests/stages/test_matcher.py` | All four matcher phases |
| `tests/stages/test_merger.py` | Anchor phrase and merge tests |
| `tests/stages/test_pdf_processor.py` | Soft redaction detection tests |
| `tests/stages/test_output_generator.py` | PDF output tests |
| `requirements.txt` | All dependencies pinned |
| `.gitignore` | Ignore data/, __pycache__, .env, etc. |

---

## Chunk 1: Project Setup + Core Infrastructure

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `config.yaml`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/stages/__init__.py`
- Create: `core/__init__.py`
- Create: `stages/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
click==8.1.8
duckdb==1.2.1
datasketch==1.6.5
numpy==2.2.3
pymupdf==1.25.3
pyyaml==6.0.2
httpx==0.28.1
rich==13.9.4
pytest==8.3.5
pytest-asyncio==0.25.3
```

- [ ] **Step 2: Create .gitignore**

```
data/
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
dist/
build/
.pytest_cache/
```

- [ ] **Step 3: Create config.yaml**

```yaml
output_dir: ./output
cache_dir: ./data/cache
db_path: ./data/unobfuscator.db

workers:
  text: 4
  pdf: 2

matching:
  min_overlap_chars: 200
  similarity_threshold: 0.70
  email_header_min_matches: 2

polling:
  interval_minutes: 60

redaction_markers:
  - "[REDACTED]"
  - "[REDACTED TEXT]"
  - "[REDACTED PER]"
  - "[b(6)]"
  - "[b(7)(c)]"
  - "[b(7)(e)]"
  - "XXXXXXXXX"
  - "■■■■■■■"
  - "(redacted)"
  - "*** REDACTED ***"
  - "<REDACTED>"
```

- [ ] **Step 4: Create all empty `__init__.py` files**

```bash
touch core/__init__.py stages/__init__.py tests/__init__.py tests/core/__init__.py tests/stages/__init__.py
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore config.yaml core/__init__.py stages/__init__.py tests/__init__.py tests/core/__init__.py tests/stages/__init__.py
git commit -m "chore: project scaffolding and dependencies"
```

---

### Task 2: core/db.py — Schema and CRUD

**Files:**
- Create: `core/db.py`
- Create: `tests/core/test_db.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_db.py`:

```python
import pytest
import json
from pathlib import Path
from core.db import (
    init_db, get_connection, upsert_document, get_unprocessed_documents,
    mark_text_processed, upsert_fingerprint, get_all_fingerprints,
    create_match_group, add_group_member, get_doc_group, merge_groups,
    upsert_merge_result, get_config, set_config
)


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


SAMPLE_DOC = {
    "id": 1, "source": "doj", "release_batch": "VOL00001",
    "original_filename": "test.pdf", "page_count": 5,
    "size_bytes": 1000, "description": "A test document",
    "extracted_text": "This is test content about someone important."
}


def test_init_db_creates_all_tables(db_path):
    conn = get_connection(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    for expected in ["documents", "document_fingerprints", "match_groups",
                     "match_group_members", "merge_results", "release_batches",
                     "jobs", "config"]:
        assert expected in tables


def test_upsert_document_stores_record(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    rows = get_unprocessed_documents(conn)
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["extracted_text"] == SAMPLE_DOC["extracted_text"]


def test_mark_text_processed_removes_from_unprocessed(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    mark_text_processed(conn, 1)
    conn.commit()
    assert get_unprocessed_documents(conn) == []


def test_upsert_fingerprint_stores_and_retrieves(conn):
    upsert_document(conn, SAMPLE_DOC)
    conn.commit()
    sig = b"\x01\x02\x03"
    upsert_fingerprint(conn, 1, sig, 42)
    conn.commit()
    fps = get_all_fingerprints(conn)
    assert len(fps) == 1
    assert fps[0]["doc_id"] == 1
    assert fps[0]["minhash_sig"] == sig


def test_match_group_member_uniqueness(conn):
    upsert_document(conn, SAMPLE_DOC)
    upsert_document(conn, {**SAMPLE_DOC, "id": 2, "original_filename": "b.pdf"})
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    conn.commit()
    add_group_member(conn, g1, 1, 1.0)
    conn.commit()
    # Adding doc 1 to g2 should be ignored (UNIQUE on doc_id)
    add_group_member(conn, g2, 1, 0.9)
    conn.commit()
    assert get_doc_group(conn, 1) == g1


def test_merge_groups_reassigns_members(conn):
    for i in [1, 2]:
        upsert_document(conn, {**SAMPLE_DOC, "id": i, "original_filename": f"{i}.pdf"})
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    conn.commit()
    add_group_member(conn, g1, 1, 1.0)
    add_group_member(conn, g2, 2, 1.0)
    conn.commit()
    merge_groups(conn, g1, g2)
    conn.commit()
    assert get_doc_group(conn, 2) == g1


def test_upsert_merge_result_tracks_previous_count(conn):
    g = create_match_group(conn)
    conn.commit()
    upsert_merge_result(conn, g, "merged text", 5, 10, [1, 2])
    conn.commit()
    upsert_merge_result(conn, g, "more merged text", 8, 10, [1, 2, 3])
    conn.commit()
    row = conn.execute(
        "SELECT recovered_count, previous_recovered_count FROM merge_results WHERE group_id = ?", (g,)
    ).fetchone()
    assert row["recovered_count"] == 8
    assert row["previous_recovered_count"] == 5


def test_config_get_set_roundtrip(conn):
    set_config(conn, "test_key", {"nested": 42})
    conn.commit()
    assert get_config(conn, "test_key") == {"nested": 42}


def test_config_get_returns_default_when_missing(conn):
    assert get_config(conn, "nonexistent", default="fallback") == "fallback"
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
pytest tests/core/test_db.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `core.db` does not exist yet.

- [ ] **Step 3: Implement core/db.py**

```python
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
    output_generated BOOLEAN DEFAULT 0
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
    conn.execute("""
        INSERT OR IGNORE INTO documents
        (id, source, release_batch, original_filename, page_count,
         size_bytes, description, extracted_text, indexed_at)
        VALUES (:id, :source, :release_batch, :original_filename, :page_count,
                :size_bytes, :description, :extracted_text, CURRENT_TIMESTAMP)
    """, doc)
    conn.execute("""
        UPDATE documents SET
            source = :source, release_batch = :release_batch,
            original_filename = :original_filename, page_count = :page_count,
            size_bytes = :size_bytes, description = :description,
            extracted_text = :extracted_text
        WHERE id = :id
          AND text_processed = 0 AND pdf_processed = 0
    """, doc)


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
                        source_doc_ids: list) -> None:
    existing = conn.execute(
        "SELECT recovered_count FROM merge_results WHERE group_id = ?", (group_id,)
    ).fetchone()
    prev_count = existing["recovered_count"] if existing else 0
    conn.execute("""
        INSERT OR REPLACE INTO merge_results
        (group_id, merged_text, recovered_count, previous_recovered_count,
         total_redacted, source_doc_ids, updated_at, output_generated)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 0)
    """, (group_id, merged_text, recovered_count, prev_count,
          total_redacted, json.dumps(source_doc_ids)))


def get_config(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_config(conn, key: str, value) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/core/test_db.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/db.py tests/core/test_db.py
git commit -m "feat: core/db.py — SQLite schema and CRUD"
```

---

### Task 3: core/queue.py — Job Queue

**Files:**
- Create: `core/queue.py`
- Create: `tests/core/test_queue.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_queue.py`:

```python
import pytest
import json
from core.db import init_db, get_connection
from core.queue import enqueue, dequeue, mark_done, mark_failed, get_queue_stats


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def test_enqueue_creates_pending_job(conn):
    enqueue(conn, stage="index", payload={"doc_id": 1})
    conn.commit()
    job = dequeue(conn)
    assert job["stage"] == "index"
    assert json.loads(job["payload"]) == {"doc_id": 1}
    assert job["status"] == "running"


def test_dequeue_returns_highest_priority_first(conn):
    enqueue(conn, stage="index", payload={"doc_id": 1}, priority=0)
    enqueue(conn, stage="index", payload={"doc_id": 2}, priority=100)
    conn.commit()
    job = dequeue(conn)
    assert json.loads(job["payload"])["doc_id"] == 2


def test_dequeue_returns_none_when_empty(conn):
    assert dequeue(conn) is None


def test_dequeue_filters_by_stage(conn):
    enqueue(conn, stage="index", payload={"doc_id": 1})
    enqueue(conn, stage="match", payload={"doc_id": 2})
    conn.commit()
    job = dequeue(conn, stage="match")
    assert job["stage"] == "match"


def test_mark_done_updates_status(conn):
    enqueue(conn, stage="index", payload={"doc_id": 1})
    conn.commit()
    job = dequeue(conn)
    mark_done(conn, job["job_id"])
    conn.commit()
    done = conn.execute(
        "SELECT status FROM jobs WHERE job_id = ?", (job["job_id"],)
    ).fetchone()
    assert done["status"] == "done"


def test_mark_failed_stores_error_message(conn):
    enqueue(conn, stage="index", payload={"doc_id": 1})
    conn.commit()
    job = dequeue(conn)
    mark_failed(conn, job["job_id"], "connection timeout")
    conn.commit()
    row = conn.execute(
        "SELECT status, error FROM jobs WHERE job_id = ?", (job["job_id"],)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "connection timeout"


def test_get_queue_stats_counts_by_status(conn):
    enqueue(conn, stage="index", payload={})
    enqueue(conn, stage="index", payload={})
    enqueue(conn, stage="match", payload={})
    conn.commit()
    stats = get_queue_stats(conn)
    assert stats["pending"] == 3
    assert stats["running"] == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/core/test_queue.py -v
```

Expected: `ImportError` — `core.queue` does not exist yet.

- [ ] **Step 3: Implement core/queue.py**

```python
import json
from typing import Optional


def enqueue(conn, stage: str, payload: dict, priority: int = 0) -> int:
    cursor = conn.execute("""
        INSERT INTO jobs (stage, payload, priority, status)
        VALUES (?, ?, ?, 'pending')
    """, (stage, json.dumps(payload), priority))
    return cursor.lastrowid


def dequeue(conn, stage: Optional[str] = None) -> Optional[dict]:
    """Atomically claim the highest-priority pending job using UPDATE...RETURNING.

    SQLite 3.35+ (Python 3.11 standard) supports RETURNING, which makes the
    SELECT + UPDATE atomic — safe for multiple parallel workers.
    """
    if stage:
        row = conn.execute("""
            UPDATE jobs SET status = 'running', updated_at = CURRENT_TIMESTAMP
            WHERE job_id = (
                SELECT job_id FROM jobs
                WHERE status = 'pending' AND stage = ?
                ORDER BY priority DESC, job_id ASC LIMIT 1
            )
            RETURNING *
        """, (stage,)).fetchone()
    else:
        row = conn.execute("""
            UPDATE jobs SET status = 'running', updated_at = CURRENT_TIMESTAMP
            WHERE job_id = (
                SELECT job_id FROM jobs
                WHERE status = 'pending'
                ORDER BY priority DESC, job_id ASC LIMIT 1
            )
            RETURNING *
        """).fetchone()

    return dict(row) if row else None


def mark_done(conn, job_id: int) -> None:
    conn.execute("""
        UPDATE jobs SET status = 'done', updated_at = CURRENT_TIMESTAMP
        WHERE job_id = ?
    """, (job_id,))


def mark_failed(conn, job_id: int, error: str) -> None:
    conn.execute("""
        UPDATE jobs SET status = 'failed', error = ?,
        updated_at = CURRENT_TIMESTAMP WHERE job_id = ?
    """, (error, job_id))


def get_queue_stats(conn) -> dict:
    rows = conn.execute("""
        SELECT status, COUNT(*) as count FROM jobs GROUP BY status
    """).fetchall()
    return {row["status"]: row["count"] for row in rows}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/core/test_queue.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/queue.py tests/core/test_queue.py
git commit -m "feat: core/queue.py — priority job queue"
```

---

### Task 4: core/config.py — Config Loading

**Files:**
- Create: `core/config.py`
- Create: `tests/core/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_config.py`:

```python
import pytest
import yaml
from pathlib import Path
from core.config import load_config, get


@pytest.fixture
def config_file(tmp_path):
    cfg = {
        "output_dir": "./output",
        "db_path": "./data/test.db",
        "matching": {"min_overlap_chars": 200, "similarity_threshold": 0.70},
        "redaction_markers": ["[REDACTED]", "XXXXXXXXX"],
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


def test_load_config_reads_yaml(config_file):
    cfg = load_config(config_file)
    assert cfg["output_dir"] == "./output"
    assert cfg["matching"]["min_overlap_chars"] == 200


def test_get_nested_key(config_file):
    cfg = load_config(config_file)
    assert get(cfg, "matching.min_overlap_chars") == 200


def test_get_returns_default_for_missing_key(config_file):
    cfg = load_config(config_file)
    assert get(cfg, "nonexistent.key", default=99) == 99


def test_load_config_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/core/test_config.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement core/config.py**

```python
import yaml
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)


def get(cfg: dict, key_path: str, default: Any = None) -> Any:
    """Access nested config values with dot notation: 'matching.min_overlap_chars'"""
    keys = key_path.split(".")
    value = cfg
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            return default
        value = value[k]
    return value
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/core/test_config.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/core/test_config.py
git commit -m "feat: core/config.py — config loading with dot-notation access"
```

---

### Task 5: core/api.py — Jmail API Wrapper

**Files:**
- Create: `core/api.py`
- Create: `tests/core/test_api.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_api.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/core/test_api.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement core/api.py**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/core/test_api.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/api.py tests/core/test_api.py
git commit -m "feat: core/api.py — Jmail DuckDB wrapper"
```

---

### Task 6: PIPELINE.md + unobfuscator.py CLI Skeleton

**Files:**
- Create: `PIPELINE.md`
- Create: `unobfuscator.py`

- [ ] **Step 1: Create PIPELINE.md**

```markdown
# Unobfuscator Pipeline — Plain English Logic

This file is the source of truth for what the tool does.
The Python code in `stages/` implements these steps exactly.

---

## Phase 0 — Email Chain Fast-Path

Run this BEFORE fingerprinting. Handles most of the email corpus instantly.

    FOR EACH document:
      Extract any email headers present (From:, To:, Date:, Subject: lines)

    BUILD an index: header_value → [list of doc_ids that contain it]

    FOR EACH header_value that appears in 2 or more documents:
      All those documents are confirmed matches.
      Group them immediately.
      Mark them as "phase0_matched" so they skip Phases 1–3.

---

## Phase 1 — Fingerprinting (for documents NOT matched in Phase 0)

    FOR EACH unmatched document:
      1. Clean the text:
         - Remove all redaction markers (from config.yaml redaction_markers list)
         - Normalize whitespace (collapse multiple spaces/newlines to single space)
         - Lowercase everything

      2. Shingling — slice into overlapping 8-word windows:
         e.g. "the meeting was held at mar a lago" produces:
           ["the meeting was held at mar a",
            "meeting was held at mar a lago"]

      3. MinHash — generate a 128-number signature:
         Use datasketch MinHashLSH with num_perm=128
         Each number represents the minimum hash of all shingles
         under one of 128 different hash functions.
         Documents with similar content will have similar signatures.

      4. Store the signature in document_fingerprints table.

---

## Phase 2 — Candidate Finding via LSH Banding

The similarity_threshold in config.yaml (default: 0.70) is the target
Jaccard similarity. The LSH parameters (bands/rows) are set to guarantee
that documents with ≥70% overlap will share at least one bucket with
>99% probability.

    USING datasketch MinHashLSH:
      Insert all fingerprints into the LSH index.

      FOR EACH document:
        Query the LSH index for its nearest neighbors.
        These are "candidates" — documents likely to overlap ≥70%.

    Candidates proceed to Phase 3 for verification.
    Non-candidates are skipped — assumed not the same document.

---

## Phase 3 — Verification and Grouping

    FOR EACH candidate pair (doc_A, doc_B):

      1. Strip redaction markers from both texts.
         Find the longest common substring between them.
         IF that substring is shorter than min_overlap_chars (from config):
           → REJECT this pair. Not the same document.

      2. Check for complementary redactions:
         Find all redaction marker positions in doc_A.
         Find all redaction marker positions in doc_B.
         IF doc_A has text where doc_B is redacted, OR vice versa:
           → CONFIRMED MATCH. Proceed to grouping.
         ELSE IF common text is long (>500 chars) but no complementary redactions:
           → Still group them (more versions = better coverage later).

      3. Assign to a match group:
         - IF doc_A already in a group → add doc_B to that group
         - IF doc_B already in a group → add doc_A to that group
         - IF both in different groups → merge the two groups (keep lower group_id)
         - IF neither in a group → create new group, add both

---

## Phase 4 — Merging (Stage 3)

    FOR EACH match group with 2 or more members:

      1. Pick the "base" document: the member with the fewest redaction markers.

      2. FOR EACH redaction marker in the base document:
         a. Extract the 50 characters immediately BEFORE the marker ("left anchor")
         b. Extract the 50 characters immediately AFTER the marker ("right anchor")

         c. FOR EACH other member of the group:
            Search for the left anchor in their text.
            IF found:
              Also search for the right anchor after that position.
              IF right anchor also found:
                Extract the text between those two anchor positions.
                → This is the recovered text for this redaction.
                → Record the source doc_id.
                → STOP searching other members for this redaction.

      3. Build merged_text:
         Start with base document text.
         Replace each redaction marker where recovery was successful
         with the recovered text.

      4. Store in merge_results:
         - merged_text
         - recovered_count (how many redaction gaps were filled)
         - total_redacted (how many redaction gaps existed in base)
         - source_doc_ids (which docs provided recovered text)

---

## Phase 5 — PDF Soft Redaction Removal (Stage 4)

    FOR EACH document that has a PDF available:

      1. Open the PDF with PyMuPDF (fitz).

      2. FOR EACH page:
         Get all annotation objects on the page.
         FOR EACH annotation of type "Redact":
           Extract the text hidden under the annotation rectangle
           using page.get_textbox(annotation.rect).
           IF text found:
             → Soft redaction detected. Text was never truly removed.
             → Record the position and recovered text.

      3. IF any soft redactions found:
         Insert a new "merge" job for this document's match group
         so Stage 3 re-runs with the additional recovered text.

---

## Phase 6 — Output Generation (Stage 5)

    FOR EACH merge_result where recovered_count > 0
    AND output_generated = False (or recovered_count increased):

      1. Determine PDF source:
         IF original PDF available:
           Open it with PyMuPDF as the base.
         ELSE:
           Create a new blank PDF with fpdf2 or PyMuPDF.

      2. For each recovered segment:
         IF using original PDF:
           Find the text position on the page.
           Draw a yellow rectangle highlight behind it (cross-doc merge)
           OR green rectangle (soft redaction removal).
         ELSE (reconstructed PDF):
           Insert the merged_text as plain text.
           Apply yellow highlight to recovered segments.

      3. Add a final footnote page:
         List all source documents (ID, source, batch, filename, URL).
         Show recovered_count, soft redactions removed, date generated.

      4. Save to: output/{source}/{batch}/{doc_id}_merged.pdf
         Mark output_generated = True in merge_results.
         IF overwriting: update updated_at.
```

- [ ] **Step 2: Create unobfuscator.py CLI skeleton**

```python
#!/usr/bin/env python3
"""Unobfuscator — CLI entry point."""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

console = Console()


@click.group()
@click.option("--config", default="config.yaml", show_default=True,
              help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """Unobfuscator: cross-reference and unredact Epstein archive documents."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.pass_context
def start(ctx):
    """Start the background daemon (all 5 stages)."""
    console.print("[green]Starting Unobfuscator daemon...[/green]")
    # Implementation added in Chunk 6


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the background daemon gracefully."""
    console.print("[yellow]Stopping daemon...[/yellow]")
    # Implementation added in Chunk 6


@cli.command()
@click.option("--doc", type=int, default=None, help="Show details for a specific document ID")
@click.pass_context
def status(ctx, doc):
    """Show processing progress across all stages."""
    console.print("[bold]Unobfuscator — Status[/bold]")
    console.print("(Daemon not yet implemented — see Chunk 6)")


@cli.command()
@click.argument("query", required=False)
@click.option("--person", default=None, help="Search by person name via Jmail people dataset")
@click.option("--date", nargs=2, default=None, metavar="FROM TO",
              help="Filter by date range (YYYY-MM-DD YYYY-MM-DD)")
@click.option("--batch", default=None, help="Target a specific release batch")
@click.option("--doc", "doc_id", type=int, default=None, help="Process a specific document ID")
@click.option("--wait", is_flag=True, help="Block until results are ready")
@click.option("--output", default=None, help="Override output directory for this run")
@click.pass_context
def search(ctx, query, person, date, batch, doc_id, wait, output):
    """Run a targeted manual search with priority +100."""
    console.print("[green]Search queued (implementation in Chunk 2+)[/green]")


@cli.group()
def config():
    """View and update configuration."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx):
    """Print current configuration."""
    from core.config import load_config
    import yaml
    cfg = load_config(ctx.obj["config_path"])
    console.print(yaml.dump(cfg))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a configuration value (dot notation supported)."""
    console.print(f"[yellow]Config set {key}={value} (DB integration added in Chunk 2)[/yellow]")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 3: Verify CLI skeleton runs**

```bash
python unobfuscator.py --help
python unobfuscator.py status
python unobfuscator.py config show
```

Expected: Help text prints, status prints placeholder message, config show prints config.yaml contents.

- [ ] **Step 4: Commit**

```bash
git add PIPELINE.md unobfuscator.py
git commit -m "feat: PIPELINE.md pseudocode and CLI skeleton"
```

---

### Task 7: tests/conftest.py — Shared Fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create shared test fixtures**

```python
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
```

- [ ] **Step 2: Verify conftest is importable**

```bash
pytest tests/ --collect-only
```

Expected: All existing tests collected with no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: shared fixtures in conftest.py"
```

---

*End of Chunk 1*

---

## Chunk 2: Stage 1 — Indexer

Stage 1 fetches documents from the Jmail API and stores them in the local DB. It runs in two passes per document: first store metadata, then generate a MinHash fingerprint from the extracted text.

### Task 8: stages/indexer.py

**Files:**
- Create: `stages/indexer.py`
- Create: `tests/stages/test_indexer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/stages/test_indexer.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from core.db import init_db, get_connection, get_all_fingerprints, get_unprocessed_documents
from stages.indexer import (
    clean_text, shingle, build_fingerprint,
    index_document, run_indexer_batch
)

REDACTION_MARKERS = ["[REDACTED]", "[b(6)]", "XXXXXXXXX"]


# --- Text cleaning and shingling ---

def test_clean_text_strips_redaction_markers():
    text = "Hello [REDACTED] world [b(6)] test XXXXXXXXX end"
    result = clean_text(text, REDACTION_MARKERS)
    assert "[REDACTED]" not in result
    assert "[b(6)]" not in result
    assert "XXXXXXXXX" not in result
    assert "hello" in result  # lowercased


def test_clean_text_normalizes_whitespace():
    text = "one   two\n\nthree\t\tfour"
    result = clean_text(text, REDACTION_MARKERS)
    assert "  " not in result
    assert "\n" not in result
    assert "\t" not in result


def test_shingle_produces_overlapping_windows():
    words = "a b c d e f g h i j".split()
    shingles = shingle(" ".join(words), window=4)
    assert "a b c d" in shingles
    assert "b c d e" in shingles
    assert len(shingles) == len(words) - 4 + 1


def test_shingle_returns_empty_for_short_text():
    shingles = shingle("only three words", window=8)
    assert shingles == []


def test_build_fingerprint_returns_bytes_of_expected_length():
    text = " ".join(["word"] * 100)  # enough words to shingle
    sig = build_fingerprint(text, num_perm=128)
    assert isinstance(sig, bytes)
    assert len(sig) > 0


def test_build_fingerprint_is_consistent():
    text = "the quick brown fox jumped over the lazy dog today"
    sig1 = build_fingerprint(text, num_perm=128)
    sig2 = build_fingerprint(text, num_perm=128)
    assert sig1 == sig2


# --- DB integration ---

@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def test_index_document_stores_doc_and_fingerprint(conn):
    doc = {
        "id": 1, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "a.pdf", "page_count": 2,
        "size_bytes": 500, "description": "Test",
        "extracted_text": "The flight departed from Palm Beach with several passengers on board."
    }
    index_document(conn, doc, redaction_markers=REDACTION_MARKERS, num_perm=128)
    conn.commit()
    assert get_unprocessed_documents(conn) == []  # marked as processed
    fps = get_all_fingerprints(conn)
    assert len(fps) == 1
    assert fps[0]["doc_id"] == 1


def test_index_document_handles_empty_text_gracefully(conn):
    doc = {
        "id": 2, "source": "doj", "release_batch": "VOL00001",
        "original_filename": "empty.pdf", "page_count": 1,
        "size_bytes": 100, "description": "Empty doc",
        "extracted_text": ""
    }
    index_document(conn, doc, redaction_markers=REDACTION_MARKERS, num_perm=128)
    conn.commit()
    # Should not crash; no fingerprint for empty docs
    fps = get_all_fingerprints(conn)
    assert len(fps) == 0


@patch("stages.indexer.fetch_documents_metadata")
@patch("stages.indexer.fetch_document_text")
def test_run_indexer_batch_processes_all_docs(mock_text, mock_meta, conn):
    mock_meta.return_value = [
        {"id": 10, "source": "doj", "release_batch": "VOL00001",
         "original_filename": "x.pdf", "page_count": 1,
         "size_bytes": 200, "description": "Doc 10"},
    ]
    mock_text.return_value = "This is a document with enough words to generate shingles for testing."
    run_indexer_batch(conn, batch_id="VOL00001",
                      redaction_markers=REDACTION_MARKERS, num_perm=128)
    conn.commit()
    fps = get_all_fingerprints(conn)
    assert len(fps) == 1
    assert fps[0]["doc_id"] == 10
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/stages/test_indexer.py -v
```

Expected: `ImportError` — `stages.indexer` does not exist yet.

- [ ] **Step 3: Implement stages/indexer.py**

```python
"""Stage 1: Indexer — fetch documents from Jmail and build MinHash fingerprints.

Logic reference: PIPELINE.md — Phase 1 (Fingerprinting)
"""

import re
import numpy as np
from typing import Optional
from datasketch import MinHash
from core.api import fetch_documents_metadata, fetch_document_text
from core.db import (
    upsert_document, upsert_fingerprint, mark_text_processed
)


def clean_text(text: str, redaction_markers: list[str]) -> str:
    """Remove redaction markers, normalize whitespace, lowercase."""
    for marker in redaction_markers:
        text = text.replace(marker, " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shingle(text: str, window: int = 8) -> list[str]:
    """Slice text into overlapping n-word windows (shingles)."""
    words = text.split()
    if len(words) < window:
        return []
    return [" ".join(words[i:i + window]) for i in range(len(words) - window + 1)]


def build_fingerprint(text: str, num_perm: int = 128) -> bytes:
    """Generate a MinHash signature for the given text.

    Returns raw bytes of the uint64 hashvalues array.
    Deserialize in matcher.py with: np.frombuffer(sig, dtype=np.uint64)
    """
    m = MinHash(num_perm=num_perm)
    for s in shingle(text):
        m.update(s.encode("utf-8"))
    return m.hashvalues.tobytes()


def index_document(conn, doc: dict, redaction_markers: list[str],
                   num_perm: int = 128) -> None:
    """Store a document in the DB and generate its MinHash fingerprint."""
    upsert_document(conn, doc)

    text = doc.get("extracted_text") or ""
    if not text.strip():
        mark_text_processed(conn, doc["id"])
        return

    cleaned = clean_text(text, redaction_markers)
    shingles = shingle(cleaned)
    if not shingles:
        mark_text_processed(conn, doc["id"])
        return

    sig = build_fingerprint(cleaned, num_perm=num_perm)
    upsert_fingerprint(conn, doc["id"], sig, len(shingles))
    mark_text_processed(conn, doc["id"])


def run_indexer_batch(conn, batch_id: Optional[str],
                      redaction_markers: list[str],
                      num_perm: int = 128) -> int:
    """Fetch all documents for a batch from Jmail and index them. Returns count."""
    docs = fetch_documents_metadata(batch_id=batch_id)
    count = 0
    for meta in docs:
        text = fetch_document_text(meta["id"])
        doc = {**meta, "extracted_text": text or ""}
        index_document(conn, doc, redaction_markers, num_perm)
        conn.commit()
        count += 1
    return count
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/stages/test_indexer.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add stages/indexer.py tests/stages/test_indexer.py
git commit -m "feat: Stage 1 indexer — fetch, store, fingerprint documents"
```

---

*End of Chunk 2*

---

## Chunk 3: Stage 2 — Matcher

Stage 2 runs all four phases from PIPELINE.md: Phase 0 (email fast-path), Phase 1 (already done by indexer — fingerprints exist), Phase 2 (LSH banding via datasketch), and Phase 3 (verification and grouping).

### Task 9: stages/matcher.py — Phase 0 (Email Header Fast-Path)

**Files:**
- Create: `stages/matcher.py`
- Create: `tests/stages/test_matcher.py`

- [ ] **Step 1: Write failing tests for Phase 0**

Create `tests/stages/test_matcher.py`:

```python
import pytest
import numpy as np
from datasketch import MinHash
from core.db import (
    init_db, get_connection, upsert_document, upsert_fingerprint,
    get_doc_group, create_match_group, add_group_member
)
from stages.matcher import (
    extract_email_headers, run_phase0_email_fastpath,
    load_fingerprints, run_phase2_lsh_candidates,
    find_longest_common_substring, run_phase3_verify_and_group
)
from stages.indexer import build_fingerprint, clean_text


REDACTION_MARKERS = ["[REDACTED]", "[b(6)]"]

EMAIL_TEXT_A = (
    "From: jeffrey@example.com\nTo: assistant@example.com\n"
    "Date: 2002-03-10\nSubject: Meeting notes\n\n"
    "The attendees included [REDACTED] and Prince Andrew. "
    "Location was the island."
)

EMAIL_TEXT_B = (
    "From: jeffrey@example.com\nTo: assistant@example.com\n"
    "Date: 2002-03-10\nSubject: Meeting notes\n\n"
    "The attendees included Bill Clinton and Prince Andrew. "
    "Location was [REDACTED]."
)

UNRELATED_TEXT = (
    "Budget report Q3. Total expenses were 2.4 million. "
    "Department heads reviewed the figures."
)


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def seed_doc(conn, doc_id, text, source="doj"):
    doc = {
        "id": doc_id, "source": source, "release_batch": "VOL00001",
        "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
        "size_bytes": 500, "description": "", "extracted_text": text
    }
    upsert_document(conn, doc)


# --- Phase 0 ---

def test_extract_email_headers_finds_from_to_date_subject():
    headers = extract_email_headers(EMAIL_TEXT_A)
    assert any("jeffrey@example.com" in h for h in headers)
    assert any("2002-03-10" in h for h in headers)
    assert any("Meeting notes" in h for h in headers)


def test_extract_email_headers_returns_empty_for_non_email():
    headers = extract_email_headers(UNRELATED_TEXT)
    assert headers == []


def test_phase0_groups_docs_with_matching_headers(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
    seed_doc(conn, 3, UNRELATED_TEXT)
    conn.commit()
    matched = run_phase0_email_fastpath(
        conn, min_header_matches=2
    )
    assert matched == {1, 2}
    assert get_doc_group(conn, 1) is not None
    assert get_doc_group(conn, 1) == get_doc_group(conn, 2)
    assert get_doc_group(conn, 3) is None


def test_phase0_does_not_group_single_header_match(conn):
    # Only Subject matches, not From/To/Date — should not group
    text_a = "From: a@a.com\nSubject: Same subject\n\nContent A here with more text."
    text_b = "From: b@b.com\nSubject: Same subject\n\nContent B here with more text."
    seed_doc(conn, 1, text_a)
    seed_doc(conn, 2, text_b)
    conn.commit()
    run_phase0_email_fastpath(conn, min_header_matches=2)
    assert get_doc_group(conn, 1) is None
    assert get_doc_group(conn, 2) is None


# --- Phase 2 (LSH) ---

def make_fingerprint(text):
    m = MinHash(num_perm=128)
    from stages.indexer import shingle
    for s in shingle(clean_text(text, REDACTION_MARKERS)):
        m.update(s.encode("utf-8"))
    return m.hashvalues.tobytes()


def test_load_fingerprints_returns_doc_id_to_minhash_map(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    upsert_fingerprint(conn, 1, make_fingerprint(EMAIL_TEXT_A), 50)
    conn.commit()
    fps = load_fingerprints(conn, num_perm=128)
    assert 1 in fps
    assert isinstance(fps[1], MinHash)


def test_phase2_finds_similar_docs_as_candidates(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
    seed_doc(conn, 3, UNRELATED_TEXT)
    upsert_fingerprint(conn, 1, make_fingerprint(EMAIL_TEXT_A), 50)
    upsert_fingerprint(conn, 2, make_fingerprint(EMAIL_TEXT_B), 50)
    upsert_fingerprint(conn, 3, make_fingerprint(UNRELATED_TEXT), 10)
    conn.commit()
    candidates = run_phase2_lsh_candidates(conn, threshold=0.3, num_perm=128)
    # docs 1 and 2 share most content — should be candidates
    assert (1, 2) in candidates or (2, 1) in candidates
    # doc 3 is unrelated — should not be paired with 1 or 2
    assert (1, 3) not in candidates and (3, 1) not in candidates


# --- Phase 3 (Verification) ---

def test_find_longest_common_substring_ignores_redaction_markers():
    a = "The passenger list included [REDACTED] and Prince Andrew."
    b = "The passenger list included Bill Clinton and Prince Andrew."
    common = find_longest_common_substring(a, b, REDACTION_MARKERS)
    assert "Prince Andrew" in common
    assert len(common) >= 20


def test_find_longest_common_substring_returns_empty_for_unrelated():
    a = "The quick brown fox jumped over the fence today."
    b = "Budget figures show a deficit of two million dollars."
    common = find_longest_common_substring(a, b, REDACTION_MARKERS)
    assert len(common) < 10


def test_phase3_groups_confirmed_candidate_pair(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
    conn.commit()
    candidates = [(1, 2)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=20
    )
    conn.commit()
    assert get_doc_group(conn, 1) is not None
    assert get_doc_group(conn, 1) == get_doc_group(conn, 2)


def test_phase3_rejects_unrelated_pair(conn):
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 3, UNRELATED_TEXT)
    conn.commit()
    candidates = [(1, 3)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=50
    )
    conn.commit()
    assert get_doc_group(conn, 1) is None
    assert get_doc_group(conn, 3) is None


def test_phase3_merges_two_existing_groups(conn):
    # doc 1 and 2 are already in separate groups; phase3 should merge them
    seed_doc(conn, 1, EMAIL_TEXT_A)
    seed_doc(conn, 2, EMAIL_TEXT_B)
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, 1, 1.0)
    add_group_member(conn, g2, 2, 1.0)
    conn.commit()
    candidates = [(1, 2)]
    run_phase3_verify_and_group(
        conn, candidates,
        redaction_markers=REDACTION_MARKERS,
        min_overlap_chars=20
    )
    conn.commit()
    assert get_doc_group(conn, 1) == get_doc_group(conn, 2)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/stages/test_matcher.py -v
```

Expected: `ImportError` — `stages.matcher` does not exist yet.

- [ ] **Step 3: Implement stages/matcher.py**

```python
"""Stage 2: Matcher — find document groups that share overlapping content.

Logic reference: PIPELINE.md — Phases 0, 2, and 3
(Phase 1 fingerprinting is done by the Indexer in Stage 1.)
"""

import re
import numpy as np
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from core.db import (
    get_connection, create_match_group, add_group_member,
    get_doc_group, merge_groups
)

# Email header patterns to extract for Phase 0 fast-path
_HEADER_PATTERNS = [
    re.compile(r"^From:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^To:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Date:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Subject:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
]


def extract_email_headers(text: str) -> list[str]:
    """Return a list of normalized header values found in the text."""
    headers = []
    for pattern in _HEADER_PATTERNS:
        for match in pattern.finditer(text):
            headers.append(match.group(1).strip().lower())
    return headers


def run_phase0_email_fastpath(conn, min_header_matches: int = 2) -> set[int]:
    """Group documents that share min_header_matches or more identical headers.

    Returns the set of doc_ids that were matched and grouped.
    """
    rows = conn.execute(
        "SELECT id, extracted_text FROM documents WHERE extracted_text IS NOT NULL"
    ).fetchall()

    # Build index: header_value → [doc_ids]
    header_index: dict[str, list[int]] = defaultdict(list)
    doc_headers: dict[int, list[str]] = {}
    for row in rows:
        headers = extract_email_headers(row["extracted_text"] or "")
        doc_headers[row["id"]] = headers
        for h in headers:
            header_index[h].append(row["id"])

    # Find pairs sharing >= min_header_matches headers
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for header_val, doc_ids in header_index.items():
        if len(doc_ids) < 2:
            continue
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                pair = (min(doc_ids[i], doc_ids[j]), max(doc_ids[i], doc_ids[j]))
                pair_counts[pair] += 1

    matched: set[int] = set()
    for (doc_a, doc_b), count in pair_counts.items():
        if count >= min_header_matches:
            _assign_to_group(conn, doc_a, doc_b, similarity=1.0)
            matched.add(doc_a)
            matched.add(doc_b)

    return matched


def load_fingerprints(conn, num_perm: int = 128) -> dict[int, MinHash]:
    """Load all stored fingerprints from DB and reconstruct MinHash objects."""
    rows = conn.execute(
        "SELECT doc_id, minhash_sig FROM document_fingerprints"
    ).fetchall()
    result = {}
    for row in rows:
        hashvalues = np.frombuffer(row["minhash_sig"], dtype=np.uint64)
        m = MinHash(num_perm=num_perm)
        m.hashvalues = hashvalues.copy()
        result[row["doc_id"]] = m
    return result


def run_phase2_lsh_candidates(
    conn, threshold: float = 0.70, num_perm: int = 128
) -> list[tuple[int, int]]:
    """Use LSH banding to find candidate pairs likely to be the same document.

    Excludes documents already grouped by Phase 0 — they need no further matching.
    Returns list of (doc_id_a, doc_id_b) candidate pairs for Phase 3 verification.
    """
    fingerprints = load_fingerprints(conn, num_perm=num_perm)
    if len(fingerprints) < 2:
        return []

    # Exclude docs already assigned to a group by Phase 0
    already_grouped = {
        row["doc_id"]
        for row in conn.execute(
            "SELECT doc_id FROM match_group_members"
        ).fetchall()
    }
    fingerprints = {
        doc_id: mh for doc_id, mh in fingerprints.items()
        if doc_id not in already_grouped
    }
    if len(fingerprints) < 2:
        return []

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for doc_id, minhash in fingerprints.items():
        lsh.insert(str(doc_id), minhash)

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for doc_id, minhash in fingerprints.items():
        neighbors = lsh.query(minhash)
        for neighbor_id_str in neighbors:
            neighbor_id = int(neighbor_id_str)
            if neighbor_id == doc_id:
                continue
            pair = (min(doc_id, neighbor_id), max(doc_id, neighbor_id))
            if pair not in seen:
                seen.add(pair)
                candidates.append(pair)

    return candidates


def find_longest_common_substring(
    text_a: str, text_b: str, redaction_markers: list[str]
) -> str:
    """Find the longest common substring between two texts, ignoring redaction markers."""
    def strip_markers(t):
        for m in redaction_markers:
            t = t.replace(m, " ")
        return re.sub(r"\s+", " ", t).strip()

    a = strip_markers(text_a)
    b = strip_markers(text_b)

    # Use a sliding window approach: try decreasing lengths until a match is found
    min_len = 10
    best = ""
    # Limit to 500-char windows for performance
    for length in range(min(len(a), len(b), 500), min_len - 1, -10):
        for start in range(0, len(a) - length + 1, 10):
            substr = a[start:start + length]
            if substr in b and len(substr) > len(best):
                best = substr
                break
        if len(best) >= length - 10:
            break
    return best


def _has_complementary_redactions(
    text_a: str, text_b: str, redaction_markers: list[str]
) -> bool:
    """Return True if one doc has text where the other has a redaction marker."""
    for marker in redaction_markers:
        if marker in text_a and marker not in text_b:
            return True
        if marker in text_b and marker not in text_a:
            return True
    return False


def run_phase3_verify_and_group(
    conn, candidates: list[tuple[int, int]],
    redaction_markers: list[str],
    min_overlap_chars: int = 200
) -> None:
    """Verify candidate pairs and group confirmed matches.

    Logic reference: PIPELINE.md — Phase 3

    Primary confirmation: complementary redactions (one has text where other is redacted).
    Secondary: long common text (>500 chars) even without complementary redactions.
    Rejection: common text shorter than min_overlap_chars.
    """
    texts = {
        row["id"]: row["extracted_text"]
        for row in conn.execute(
            "SELECT id, extracted_text FROM documents WHERE extracted_text IS NOT NULL"
        ).fetchall()
    }

    for doc_a, doc_b in candidates:
        text_a = texts.get(doc_a, "")
        text_b = texts.get(doc_b, "")

        common = find_longest_common_substring(text_a, text_b, redaction_markers)
        if len(common) < min_overlap_chars:
            continue  # Not the same document — reject

        # Primary signal: complementary redactions
        if _has_complementary_redactions(text_a, text_b, redaction_markers):
            _assign_to_group(conn, doc_a, doc_b,
                             similarity=len(common) / max(len(text_a), len(text_b), 1))
            continue

        # Secondary signal: long overlap even without complementary redactions
        if len(common) >= 500:
            _assign_to_group(conn, doc_a, doc_b,
                             similarity=len(common) / max(len(text_a), len(text_b), 1))


def _assign_to_group(conn, doc_a: int, doc_b: int, similarity: float) -> None:
    """Assign two documents to a shared match group, merging if needed."""
    group_a = get_doc_group(conn, doc_a)
    group_b = get_doc_group(conn, doc_b)

    if group_a is not None and group_b is not None:
        if group_a != group_b:
            merge_groups(conn, group_a, group_b)
    elif group_a is not None:
        add_group_member(conn, group_a, doc_b, similarity)
    elif group_b is not None:
        add_group_member(conn, group_b, doc_a, similarity)
    else:
        new_group = create_match_group(conn)
        add_group_member(conn, new_group, doc_a, 1.0)
        add_group_member(conn, new_group, doc_b, similarity)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/stages/test_matcher.py -v
```

Expected: All 11 tests PASS. (10 original + 1 merge-groups test)

- [ ] **Step 5: Commit**

```bash
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "feat: Stage 2 matcher — email fast-path, LSH candidates, verification and grouping"
```

---

*End of Chunk 3*

---

## Chunk 4: Stage 3 — Merger

Stage 3 takes each match group and produces a merged reconstruction. It uses anchor-phrase matching to locate where each redacted segment in the base document can be filled from another group member.

### Task 10: stages/merger.py

**Files:**
- Create: `stages/merger.py`
- Create: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing tests**

Create `tests/stages/test_merger.py`:

```python
import pytest
import json
from core.db import (
    init_db, get_connection, upsert_document, create_match_group,
    add_group_member, upsert_merge_result
)
from stages.merger import (
    find_redaction_positions, extract_anchors,
    find_text_between_anchors, merge_group, run_merger
)

REDACTION_MARKERS = ["[REDACTED]", "[b(6)]"]

BASE_TEXT = (
    "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
    "Date: 2002-03-10\nSubject: Guest list\n\n"
    "The attendees included [REDACTED] and Prince Andrew. "
    "The flight departed from [REDACTED] at 9am and arrived at the island."
)

DONOR_TEXT_A = (
    "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
    "Date: 2002-03-10\nSubject: Guest list\n\n"
    "The attendees included Bill Clinton and Prince Andrew. "
    "The flight departed from [REDACTED] at 9am and arrived at the island."
)

DONOR_TEXT_B = (
    "From: jeffrey@example.com\nTo: ghislaine@example.com\n"
    "Date: 2002-03-10\nSubject: Guest list\n\n"
    "The attendees included [REDACTED] and Prince Andrew. "
    "The flight departed from Palm Beach at 9am and arrived at the island."
)


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def seed_doc(conn, doc_id, text):
    from core.db import upsert_document
    upsert_document(conn, {
        "id": doc_id, "source": "doj", "release_batch": "VOL00001",
        "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
        "size_bytes": 500, "description": "", "extracted_text": text
    })


# --- Core utility functions ---

def test_find_redaction_positions_returns_all_markers():
    positions = find_redaction_positions(BASE_TEXT, REDACTION_MARKERS)
    assert len(positions) == 2
    for pos, marker in positions:
        assert BASE_TEXT[pos:pos + len(marker)] == marker


def test_find_redaction_positions_returns_empty_for_clean_text():
    positions = find_redaction_positions("This text has no redactions at all.", REDACTION_MARKERS)
    assert positions == []


def test_extract_anchors_returns_text_around_position():
    text = "before context HERE after context"
    pos = text.index("HERE")
    left, right = extract_anchors(text, pos, length=7)
    assert "before" in left
    assert "after" in right


def test_extract_anchors_handles_start_of_text():
    text = "HERE after context"
    left, right = extract_anchors(text, 0, length=10)
    assert left == ""
    assert "after" in right


def test_find_text_between_anchors_recovers_redacted_content():
    left_anchor = "attendees included"
    right_anchor = "and Prince Andrew"
    recovered = find_text_between_anchors(DONOR_TEXT_A, left_anchor, right_anchor)
    assert recovered is not None
    assert "Bill Clinton" in recovered


def test_find_text_between_anchors_returns_none_when_anchors_missing():
    result = find_text_between_anchors(
        "Completely different text here.",
        "attendees included",
        "and Prince Andrew"
    )
    assert result is None


# --- Full merge ---

def test_merge_group_fills_redactions_from_donors(conn):
    seed_doc(conn, 1, BASE_TEXT)
    seed_doc(conn, 2, DONOR_TEXT_A)
    seed_doc(conn, 3, DONOR_TEXT_B)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    add_group_member(conn, g, 3, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=20)
    assert result["recovered_count"] == 2
    assert "Bill Clinton" in result["merged_text"]
    assert "Palm Beach" in result["merged_text"]
    assert "[REDACTED]" not in result["merged_text"]


def test_merge_group_returns_zero_recovered_when_no_donors_help(conn):
    seed_doc(conn, 1, BASE_TEXT)
    seed_doc(conn, 2, "Totally unrelated content that shares nothing.")
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.1)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=20)
    assert result["recovered_count"] == 0
    assert "[REDACTED]" in result["merged_text"]


def test_run_merger_stores_results_and_marks_group_merged(conn):
    seed_doc(conn, 1, BASE_TEXT)
    seed_doc(conn, 2, DONOR_TEXT_A)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    run_merger(conn, REDACTION_MARKERS, anchor_length=20)
    conn.commit()

    row = conn.execute(
        "SELECT merged, recovered_count FROM match_groups "
        "JOIN merge_results USING (group_id) WHERE group_id = ?", (g,)
    ).fetchone()
    assert row is not None
    assert row["merged"] == 1
    assert row["recovered_count"] >= 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/stages/test_merger.py -v
```

Expected: `ImportError` — `stages.merger` does not exist yet.

- [ ] **Step 3: Implement stages/merger.py**

```python
"""Stage 3: Merger — fill redaction gaps using anchor-phrase matching across match groups.

Logic reference: PIPELINE.md — Phase 4 (Merging)
"""

from typing import Optional
from core.db import upsert_merge_result


def find_redaction_positions(text: str, redaction_markers: list[str]) -> list[tuple[int, str]]:
    """Return list of (position, marker) for every redaction marker in text."""
    positions = []
    for marker in redaction_markers:
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos == -1:
                break
            positions.append((pos, marker))
            start = pos + len(marker)
    return sorted(positions, key=lambda x: x[0])


def extract_anchors(text: str, pos: int, length: int = 50) -> tuple[str, str]:
    """Extract left and right anchor phrases around a position in text."""
    left = text[max(0, pos - length):pos].strip()
    right = text[pos:pos + length].strip()
    return left, right


def find_text_between_anchors(
    text: str, left_anchor: str, right_anchor: str
) -> Optional[str]:
    """Search text for left_anchor, then find right_anchor after it.

    Returns the text between them if both anchors are found, else None.
    """
    if not left_anchor:
        # No left anchor — look for right anchor from start
        right_pos = text.find(right_anchor)
        if right_pos == -1:
            return None
        return text[:right_pos].strip()

    left_pos = text.find(left_anchor)
    if left_pos == -1:
        return None

    search_from = left_pos + len(left_anchor)
    right_pos = text.find(right_anchor, search_from)
    if right_pos == -1:
        return None

    return text[search_from:right_pos].strip()


def merge_group(
    conn, group_id: int, redaction_markers: list[str], anchor_length: int = 50
) -> dict:
    """Merge all documents in a match group to produce the best reconstruction.

    Returns dict with merged_text, recovered_count, total_redacted, source_doc_ids.
    Logic reference: PIPELINE.md — Phase 4
    """
    rows = conn.execute("""
        SELECT d.id, d.extracted_text
        FROM match_group_members m
        JOIN documents d ON d.id = m.doc_id
        WHERE m.group_id = ?
    """, (group_id,)).fetchall()

    # Sort in Python: base = member with fewest redaction markers (no SQL interpolation)
    def redaction_count(row):
        t = row["extracted_text"] or ""
        return sum(t.count(marker) for marker in redaction_markers)

    members = sorted(rows, key=redaction_count)

    if not members:
        return {"merged_text": "", "recovered_count": 0, "total_redacted": 0, "source_doc_ids": []}

    # Pick base: fewest redaction markers (first after ORDER BY above)
    base_id = members[0]["id"]
    base_text = members[0]["extracted_text"] or ""
    donors = [(row["id"], row["extracted_text"] or "") for row in members[1:]]

    positions = find_redaction_positions(base_text, redaction_markers)
    total_redacted = len(positions)
    recovered_count = 0
    source_doc_ids = []
    merged = base_text

    # Process in reverse order so string positions remain valid after substitution
    for pos, marker in reversed(positions):
        left_anchor, _ = extract_anchors(base_text, pos, anchor_length)
        # Right anchor: text after the marker
        right_start = pos + len(marker)
        right_anchor = base_text[right_start:right_start + anchor_length].strip()

        for donor_id, donor_text in donors:
            recovered = find_text_between_anchors(donor_text, left_anchor, right_anchor)
            if recovered and recovered not in redaction_markers and len(recovered) > 0:
                merged = merged[:pos] + recovered + merged[pos + len(marker):]
                recovered_count += 1
                if donor_id not in source_doc_ids:
                    source_doc_ids.append(donor_id)
                break  # Found recovery for this gap — move to next

    return {
        "merged_text": merged,
        "recovered_count": recovered_count,
        "total_redacted": total_redacted,
        "source_doc_ids": [base_id] + source_doc_ids
    }


def run_merger(conn, redaction_markers: list[str], anchor_length: int = 50) -> int:
    """Run merger on all unmerged match groups with 2+ members. Returns count processed."""
    groups = conn.execute("""
        SELECT group_id FROM match_groups
        WHERE merged = 0
        AND (SELECT COUNT(*) FROM match_group_members WHERE group_id = match_groups.group_id) >= 2
    """).fetchall()

    count = 0
    for row in groups:
        group_id = row["group_id"]
        result = merge_group(conn, group_id, redaction_markers, anchor_length)
        upsert_merge_result(
            conn, group_id,
            result["merged_text"],
            result["recovered_count"],
            result["total_redacted"],
            result["source_doc_ids"]
        )
        conn.execute(
            "UPDATE match_groups SET merged = 1 WHERE group_id = ?", (group_id,)
        )
        conn.commit()
        count += 1

    return count
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/stages/test_merger.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: Stage 3 merger — anchor-phrase redaction recovery"
```

---

*End of Chunk 4*

---

## Chunk 5: Stage 4 — PDF Processor

Stage 4 downloads original PDFs and checks for soft redactions — cases where Acrobat added a black overlay rectangle but the underlying text was never removed from the PDF stream. PyMuPDF can extract the hidden text directly. Results feed back into Stage 3 by inserting a new merge job.

### Task 11: stages/pdf_processor.py

**Files:**
- Modify: `core/db.py` (add three PDF-related helper functions)
- Create: `stages/pdf_processor.py`
- Create: `tests/stages/test_pdf_processor.py`

- [ ] **Step 1: Add PDF helper functions to core/db.py**

Append these three functions to the end of `core/db.py`. These keep all SQL in one place per the architecture spec.

```python
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
```

- [ ] **Step 2: Write failing tests**

Create `tests/stages/test_pdf_processor.py`:

```python
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
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/stages/test_pdf_processor.py -v
```

Expected: `ImportError` — `stages.pdf_processor` does not exist yet.

- [ ] **Step 4: Implement stages/pdf_processor.py**

```python
"""Stage 4: PDF Processor — download PDFs and strip soft/overlay redactions.

A "soft redaction" is one where a black rectangle was drawn over text in Acrobat
but the underlying text was never removed from the PDF stream.
PyMuPDF can extract the hidden text by reading the page stream directly.

Logic reference: PIPELINE.md — Phase 5
"""

import json
import fitz  # PyMuPDF
import httpx
from typing import Optional
from core.db import get_doc_group, get_document_for_pdf, append_soft_redaction_text, mark_pdf_processed
from core.queue import enqueue

# Jmail PDF base URL — confirmed pattern from API exploration.
# Format: https://data.jmail.world/v1/files/{source}/{batch}/{filename}
# NOTE: This pattern must be verified against the live API during implementation.
# If the URL pattern differs, update build_pdf_url() and re-run the integration test.
_PDF_BASE_URL = "https://data.jmail.world/v1/files"


def build_pdf_url(doc_id: int, source: str, batch: str, original_filename: str) -> str:
    """Construct the Jmail download URL for a document's original PDF."""
    return f"{_PDF_BASE_URL}/{source}/{batch}/{original_filename}"


def extract_soft_redactions(pdf_bytes: bytes) -> list[dict]:
    """Detect and extract text hidden under black rectangle overlays.

    Returns list of dicts: [{page: int, rect: tuple, text: str}, ...]
    Returns empty list if no soft redactions found or PDF is invalid.

    Logic reference: PIPELINE.md — Phase 5, steps 1–2
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return []

    recovered = []
    for page_num, page in enumerate(doc):
        # Find all filled black rectangles on the page (redaction overlays)
        drawings = page.get_drawings()
        for drawing in drawings:
            if drawing.get("fill") != (0.0, 0.0, 0.0):
                continue  # Not a black fill
            rect = fitz.Rect(drawing["rect"])
            if rect.width < 5 or rect.height < 5:
                continue  # Too small to be a redaction

            # Extract text from the PDF stream in that rectangle area
            hidden_text = page.get_textbox(rect).strip()
            if hidden_text:
                recovered.append({
                    "page": page_num,
                    "rect": tuple(rect),
                    "text": hidden_text
                })

    doc.close()
    return recovered


def process_pdf_for_document(conn, doc_id: int) -> None:
    """Download a document's PDF, check for soft redactions, queue merge job if found.

    Logic reference: PIPELINE.md — Phase 5, steps 1–3
    """
    doc = get_document_for_pdf(conn, doc_id)
    if not doc or not doc["pdf_url"]:
        return  # No PDF URL — skip silently

    try:
        response = httpx.get(doc["pdf_url"], timeout=30)
        response.raise_for_status()
        pdf_bytes = response.content
    except Exception:
        return  # Network failure — will be retried by the job queue on next run

    soft_redactions = extract_soft_redactions(pdf_bytes)

    if soft_redactions:
        recovered_text = "\n".join(r["text"] for r in soft_redactions)
        append_soft_redaction_text(conn, doc_id, recovered_text)

        group_id = get_doc_group(conn, doc_id)
        if group_id is not None:
            enqueue(conn, stage="merge", payload={"group_id": group_id}, priority=50)

    mark_pdf_processed(conn, doc_id)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/stages/test_pdf_processor.py -v
```

Expected: All 8 tests PASS. Note: `test_extract_soft_redactions_finds_text_under_black_rectangle` may require tuning if PyMuPDF's `get_textbox` does not recover text from the synthetic test PDF — in that case, the implementation needs testing against real Jmail PDFs with actual soft redactions.

- [ ] **Step 6: Commit**

```bash
git add core/db.py stages/pdf_processor.py tests/stages/test_pdf_processor.py
git commit -m "feat: Stage 4 PDF processor — soft redaction detection and extraction"
```

---

*End of Chunk 5*

---

## Chunk 6: Stage 5 — Output Generator + Daemon + Polling

This chunk completes the tool: output PDF generation with real highlights, the background daemon with cooperative shutdown, and polling for new Jmail batches.

**Pre-requisite:** Add `recovered_segments` and `soft_recovered_count` to `merge_results` schema, and update the merger to populate them. These are done in Task 12a before the output generator is written.

### Task 12a: Schema and merger updates for recovered_segments

**Files:**
- Modify: `core/db.py`
- Modify: `stages/merger.py`

- [ ] **Step 1: Add recovered_segments column to DB schema**

In `core/db.py`, update the `SCHEMA` string — add two columns to `merge_results`:

```sql
recovered_segments TEXT,             -- JSON: [{text, source_doc_id, stage}, ...]
soft_recovered_count INTEGER DEFAULT 0,
```

Append these helper functions to the end of `core/db.py`:

```python
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


def get_pending_pdf_document(conn) -> Optional[dict]:
    row = conn.execute(
        "SELECT id FROM documents WHERE pdf_processed = 0 AND pdf_url IS NOT NULL LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def get_known_batch_ids(conn) -> set:
    return {r["batch_id"] for r in conn.execute(
        "SELECT batch_id FROM release_batches"
    ).fetchall()}


def insert_release_batch(conn, batch_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO release_batches (batch_id, fully_indexed) VALUES (?, 0)",
        (batch_id,)
    )
```

- [ ] **Step 2: Update upsert_merge_result to accept recovered_segments**

Replace the existing `upsert_merge_result` function in `core/db.py` with:

```python
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
```

- [ ] **Step 3: Update stages/merger.py to capture recovered_segments**

In `merge_group()`, add tracking before the positions loop and update the return:

```python
# Add before the "for pos, marker in reversed(positions):" loop:
recovered_segments = []

# Inside the recovery success block, after "merged = merged[:pos] + ...":
recovered_segments.append({
    "text": recovered,
    "source_doc_id": donor_id,
    "stage": "merge"
})

# Update the return dict to include recovered_segments:
return {
    "merged_text": merged,
    "recovered_count": recovered_count,
    "total_redacted": total_redacted,
    "source_doc_ids": [base_id] + source_doc_ids,
    "recovered_segments": recovered_segments,
}
```

In `run_merger()`, pass `recovered_segments` to `upsert_merge_result`:

```python
upsert_merge_result(
    conn, group_id,
    result["merged_text"],
    result["recovered_count"],
    result["total_redacted"],
    result["source_doc_ids"],
    recovered_segments=result["recovered_segments"]
)
```

- [ ] **Step 4: Run all existing tests to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: All existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/db.py stages/merger.py
git commit -m "feat: add recovered_segments to merge_results for highlight support"
```

---

### Task 12b: stages/output_generator.py

**Files:**
- Create: `stages/output_generator.py`
- Create: `tests/stages/test_output_generator.py`

- [ ] **Step 6: Write failing tests**

Create `tests/stages/test_output_generator.py`:

```python
import pytest
import fitz
import json
from pathlib import Path
from core.db import (
    init_db, get_connection, upsert_document,
    create_match_group, add_group_member, upsert_merge_result
)
from stages.output_generator import YELLOW, GREEN, build_output_path, generate_output_pdf


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return str(d)


def seed_group(conn, base_text, merged_text, recovered_count, segments=None,
               source="doj", batch="VOL00001"):
    for doc_id, text in [(1, base_text), (2, "donor text")]:
        upsert_document(conn, {
            "id": doc_id, "source": source, "release_batch": batch,
            "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
            "size_bytes": 500, "description": "", "extracted_text": text
        })
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    upsert_merge_result(conn, g, merged_text, recovered_count=recovered_count,
                        total_redacted=max(recovered_count, 1), source_doc_ids=[1, 2],
                        recovered_segments=segments or [])
    conn.commit()
    return g


def test_build_output_path_uses_source_batch_and_doc_id(output_dir):
    path = build_output_path(output_dir, source="doj", batch="VOL00008", doc_id=12345)
    assert path.endswith("12345_merged.pdf")
    assert "doj" in path
    assert "VOL00008" in path


def test_build_output_path_creates_parent_directories(output_dir):
    path = build_output_path(output_dir, source="house_oversight",
                             batch="DataSet11", doc_id=99)
    assert Path(path).parent.exists()


def test_generate_output_pdf_creates_file(conn, output_dir):
    segments = [{"text": "Bill Clinton", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="The passenger list included [REDACTED] on the flight.",
                   merged_text="The passenger list included Bill Clinton on the flight.",
                   recovered_count=1, segments=segments)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    assert path is not None
    assert Path(path).exists()
    assert Path(path).suffix == ".pdf"


def test_generate_output_pdf_contains_recovered_text(conn, output_dir):
    segments = [{"text": "Donald Trump", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="Attendees: [REDACTED]. Location: Palm Beach.",
                   merged_text="Attendees: Donald Trump. Location: Palm Beach.",
                   recovered_count=1, segments=segments)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    doc = fitz.open(path)
    full_text = "".join(page.get_text() for page in doc)
    assert "Donald Trump" in full_text
    doc.close()


def test_generate_output_pdf_footnote_has_sources_and_recovery_method(conn, output_dir):
    segments = [{"text": "recovered", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="Text with [REDACTED] inside.",
                   merged_text="Text with recovered inside.",
                   recovered_count=1, segments=segments)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    doc = fitz.open(path)
    last_page_text = doc[-1].get_text()
    assert "SOURCES" in last_page_text
    assert "Unobfuscator" in last_page_text
    assert "Recovery method" in last_page_text
    doc.close()


def test_generate_output_pdf_returns_none_when_no_recoveries(conn, output_dir):
    g = seed_group(conn,
                   base_text="Clean text.",
                   merged_text="Clean text.",
                   recovered_count=0)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    assert path is None


def test_generate_output_pdf_marks_output_generated(conn, output_dir):
    segments = [{"text": "the President", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="The [REDACTED] attended the meeting.",
                   merged_text="The the President attended the meeting.",
                   recovered_count=1, segments=segments)
    generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                        redaction_markers=["[REDACTED]"])
    conn.commit()
    row = conn.execute(
        "SELECT output_generated FROM merge_results WHERE group_id = ?", (g,)
    ).fetchone()
    assert row["output_generated"] == 1
```

- [ ] **Step 7: Run tests to confirm they fail**

```bash
pytest tests/stages/test_output_generator.py -v
```

Expected: `ImportError` — `stages.output_generator` does not exist yet.

- [ ] **Step 8: Implement stages/output_generator.py**

```python
"""Stage 5: Output Generator — produce highlighted PDFs with footnote pages.

Only generates a file when at least one redaction was recovered.
Yellow highlight = cross-document merge recovery (Stage 3).
Green highlight  = soft redaction removal (Stage 4).

Logic reference: PIPELINE.md — Phase 6
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

from core.db import (
    get_merge_result, get_documents_by_ids,
    mark_output_generated, get_pending_output_groups
)

# Highlight colours as RGB tuples for PyMuPDF
YELLOW = (1.0, 1.0, 0.0)
GREEN = (0.0, 1.0, 0.0)

_JMAIL_DOC_BASE = "https://data.jmail.world/v1/documents"


def build_output_path(output_dir: str, source: str, batch: str, doc_id: int) -> str:
    """Build output file path and create parent directories."""
    path = Path(output_dir) / source / batch / f"{doc_id}_merged.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _apply_highlights(page: fitz.Page, segments: list[dict]) -> None:
    """Search for each recovered segment on the page and add a highlight annotation.

    Yellow for Stage 3 (cross-doc merge), Green for Stage 4 (soft redaction removal).
    """
    for seg in segments:
        text = seg.get("text", "").strip()
        stage = seg.get("stage", "merge")
        if not text:
            continue
        colour = GREEN if stage == "pdf" else YELLOW
        for rect in page.search_for(text):
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=colour)
            annot.update()


def _build_footnote_page(
    doc: fitz.Document,
    source_docs: list[dict],
    recovered_count: int,
    soft_recovered_count: int,
) -> None:
    """Append a footnote page listing all source documents and recovery stats."""
    page = doc.new_page()
    today = date.today().isoformat()

    method_parts = []
    if recovered_count - soft_recovered_count > 0:
        method_parts.append("cross-document merge")
    if soft_recovered_count > 0:
        method_parts.append("soft redaction removal")
    method = " + ".join(method_parts) if method_parts else "cross-document merge"

    lines = [
        "=" * 60,
        f"SOURCES — Unobfuscator v1.0 — {today}",
        "=" * 60,
        "",
        "This document was reconstructed from the following sources:",
        "",
    ]
    for i, src in enumerate(source_docs, 1):
        lines.append(
            f"[{i}] Document ID {src['id']} — {src['source']} "
            f"{src['release_batch']} — {src['original_filename']}"
        )
        lines.append(f"    {_JMAIL_DOC_BASE}/{src['id']}")
        lines.append("")

    lines += [
        f"Redactions recovered:      {recovered_count}",
        f"Soft redactions removed:   {soft_recovered_count}",
        f"Recovery method: {method}",
        "=" * 60,
    ]

    y = 50
    for line in lines:
        page.insert_text((40, y), line, fontsize=9)
        y += 14


def generate_output_pdf(
    conn,
    group_id: int,
    output_dir: str,
    redaction_markers: list[str],
) -> Optional[str]:
    """Generate a highlighted output PDF for a merge group.

    Returns the output file path, or None if no redactions were recovered.
    Logic reference: PIPELINE.md — Phase 6
    """
    merge_row = get_merge_result(conn, group_id)
    if not merge_row or merge_row["recovered_count"] == 0:
        return None

    merged_text = merge_row["merged_text"]
    source_doc_ids = json.loads(merge_row["source_doc_ids"])
    recovered_count = merge_row["recovered_count"]
    soft_recovered_count = merge_row.get("soft_recovered_count") or 0
    recovered_segments = json.loads(merge_row.get("recovered_segments") or "[]")

    source_docs = get_documents_by_ids(conn, source_doc_ids)
    base_doc = source_docs[0] if source_docs else {}

    output_path = build_output_path(
        output_dir,
        source=base_doc.get("source", "unknown"),
        batch=base_doc.get("release_batch", "unknown"),
        doc_id=source_doc_ids[0] if source_doc_ids else group_id
    )

    pdf = fitz.open()
    page = pdf.new_page()
    rect = fitz.Rect(40, 40, page.rect.width - 40, page.rect.height - 60)
    page.insert_textbox(rect, merged_text, fontsize=10)

    _apply_highlights(page, recovered_segments)
    _build_footnote_page(pdf, source_docs, recovered_count, soft_recovered_count)

    pdf.save(output_path)
    pdf.close()

    mark_output_generated(conn, group_id)
    return output_path


def run_output_generator(conn, output_dir: str, redaction_markers: list[str]) -> int:
    """Generate output PDFs for all pending merge results. Returns count generated."""
    count = 0
    for row in get_pending_output_groups(conn):
        path = generate_output_pdf(conn, row["group_id"], output_dir, redaction_markers)
        if path:
            count += 1
    conn.commit()
    return count
```

- [ ] **Step 9: Run tests**

```bash
pytest tests/stages/test_output_generator.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 10: Commit**

```bash
git add stages/output_generator.py tests/stages/test_output_generator.py
git commit -m "feat: Stage 5 output generator — highlighted PDFs with footnotes"
```

---

### Task 13: Daemon + Polling + Wired CLI

**Files:**
- Modify: `unobfuscator.py`
- Create: `tests/test_daemon.py`

- [ ] **Step 11: Add daemon helpers to unobfuscator.py**

Add these imports after the existing `import click` line at the top of `unobfuscator.py`:

```python
import os
import signal
import time
from typing import Optional
from core.db import (
    init_db, get_connection, get_known_batch_ids, insert_release_batch,
    get_pending_pdf_document
)
from core.config import load_config, get as cfg_get
from core.api import fetch_release_batches, fetch_person_document_ids
from core.queue import enqueue, get_queue_stats
from stages.indexer import run_indexer_batch
from stages.matcher import (
    run_phase0_email_fastpath, run_phase2_lsh_candidates,
    run_phase3_verify_and_group
)
from stages.merger import run_merger
from stages.pdf_processor import process_pdf_for_document
from stages.output_generator import run_output_generator
```

Then add these module-level variables and helper functions before the `@click.group()` decorator:

```python
PID_FILE = ".unobfuscator.pid"
_shutdown_requested = False


def _write_pid() -> None:
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _read_pid() -> Optional[int]:
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def _remove_pid() -> None:
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def _run_one_cycle(conn, cfg: dict) -> None:
    """Run one full pass of all 5 stages using only core.db helpers — no inline SQL."""
    global _shutdown_requested
    markers = cfg_get(cfg, "redaction_markers", default=[])
    min_overlap = cfg_get(cfg, "matching.min_overlap_chars", default=200)
    threshold = cfg_get(cfg, "matching.similarity_threshold", default=0.70)
    min_headers = cfg_get(cfg, "matching.email_header_min_matches", default=2)
    output_dir = cfg_get(cfg, "output_dir", default="./output")

    if _shutdown_requested:
        return

    for batch_id in get_known_batch_ids(conn):
        if _shutdown_requested:
            return
        run_indexer_batch(conn, batch_id=batch_id, redaction_markers=markers)

    if _shutdown_requested:
        return

    run_phase0_email_fastpath(conn, min_header_matches=min_headers)
    candidates = run_phase2_lsh_candidates(conn, threshold=threshold)
    run_phase3_verify_and_group(conn, candidates, redaction_markers=markers,
                                min_overlap_chars=min_overlap)

    if _shutdown_requested:
        return

    run_merger(conn, redaction_markers=markers)

    if not _shutdown_requested:
        pdf_doc = get_pending_pdf_document(conn)
        if pdf_doc:
            process_pdf_for_document(conn, doc_id=pdf_doc["id"])

    if not _shutdown_requested:
        run_output_generator(conn, output_dir=output_dir, redaction_markers=markers)

    conn.commit()


def _poll_for_new_batches(conn, cfg: dict) -> None:
    """Check Jmail for new release batches and queue indexing for any new ones."""
    known = get_known_batch_ids(conn)
    try:
        current = set(fetch_release_batches())
    except Exception:
        return  # Network failure — try again next poll

    for batch_id in current - known:
        insert_release_batch(conn, batch_id)
        enqueue(conn, stage="index", payload={"batch_id": batch_id})
    conn.commit()
```

- [ ] **Step 12: Replace start, stop, status stubs with real implementations**

Replace the stub command bodies with the following. Keep the `@cli.command()` decorators unchanged.

`start` body:

```python
    global _shutdown_requested
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    poll_interval = cfg_get(cfg, "polling.interval_minutes", default=60) * 60

    init_db(db_path)
    conn = get_connection(db_path)
    _shutdown_requested = False
    _write_pid()
    console.print(f"[green]Daemon started (PID {os.getpid()})[/green]")

    def _handle_signal(sig, frame):
        global _shutdown_requested
        _shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not _shutdown_requested:
            _run_one_cycle(conn, cfg)
            stats = get_queue_stats(conn)
            if stats.get("pending", 0) == 0 and not _shutdown_requested:
                console.print(
                    f"[dim]All tasks complete. Checking for updates in "
                    f"{poll_interval // 60:.0f} min...[/dim]"
                )
                for _ in range(int(poll_interval)):
                    if _shutdown_requested:
                        break
                    time.sleep(1)
                if not _shutdown_requested:
                    _poll_for_new_batches(conn, cfg)
    finally:
        _remove_pid()
        console.print("[yellow]Daemon stopped.[/yellow]")
```

`stop` body:

```python
    pid = _read_pid()
    if pid is None:
        console.print("[red]No daemon running.[/red]")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(
            f"[yellow]Stop signal sent to PID {pid}. "
            "Daemon will finish its current stage then exit.[/yellow]"
        )
    except ProcessLookupError:
        console.print("[red]Daemon not found — cleaning up.[/red]")
        _remove_pid()
```

`status` body (replace the single `console.print` placeholder):

```python
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    if not os.path.exists(db_path):
        console.print("[red]Database not found. Run 'unobfuscator start' first.[/red]")
        return
    conn = get_connection(db_path)

    if doc:
        from core.db import get_documents_by_ids
        rows = get_documents_by_ids(conn, [doc])
        console.print(rows[0] if rows else f"[red]Document {doc} not found.[/red]")
        return

    pid = _read_pid()
    daemon_label = f"running (PID {pid})" if pid else "stopped"

    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    indexed = conn.execute("SELECT COUNT(*) FROM documents WHERE text_processed=1").fetchone()[0]
    fps = conn.execute("SELECT COUNT(*) FROM document_fingerprints").fetchone()[0]
    groups = conn.execute("SELECT COUNT(*) FROM merge_results").fetchone()[0]
    pdf_done = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_processed=1").fetchone()[0]
    pdf_total = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE pdf_url IS NOT NULL"
    ).fetchone()[0]
    output_count = conn.execute(
        "SELECT COUNT(*) FROM merge_results WHERE output_generated=1"
    ).fetchone()[0]
    recovered = conn.execute(
        "SELECT COALESCE(SUM(recovered_count),0) FROM merge_results"
    ).fetchone()[0]
    output_dir = cfg_get(cfg, "output_dir", default="./output")

    from rich.table import Table
    t = Table(title="Unobfuscator — Status", show_header=False)
    t.add_column("", style="bold")
    t.add_column("")
    t.add_row("Daemon", daemon_label)
    t.add_row("Stage 1 Indexer", f"{indexed:,} / {total:,} docs")
    t.add_row("Stage 2 Matcher", f"{fps:,} fingerprints built")
    t.add_row("Stage 3 Merger", f"{groups:,} groups merged")
    t.add_row("Stage 4 PDF Processor", f"{pdf_done:,} / {pdf_total:,} PDFs done")
    t.add_row("Stage 5 Output", f"{output_count:,} files written")
    t.add_row("Recovered redactions", f"{recovered:,} total")
    t.add_row("Output directory", output_dir)
    console.print(t)
```

- [ ] **Step 13: Replace search stub with real implementation**

Replace the `search` command body:

```python
    cfg = load_config(ctx.obj["config_path"])
    db_path = cfg_get(cfg, "db_path", default="./data/unobfuscator.db")
    init_db(db_path)
    conn = get_connection(db_path)

    if person:
        console.print(f"[dim]Looking up '{person}' in Jmail people dataset...[/dim]")
        try:
            doc_ids = fetch_person_document_ids(person)
            if doc_ids:
                for did in doc_ids:
                    enqueue(conn, stage="index",
                            payload={"doc_id": did, "output_dir": output},
                            priority=100)
                console.print(
                    f"[green]Queued {len(doc_ids)} documents for '{person}'[/green]"
                )
            else:
                console.print(f"[yellow]No documents found for '{person}'[/yellow]")
        except Exception as e:
            console.print(f"[red]People lookup failed: {e}[/red]")
    else:
        payload: dict = {}
        if query:
            payload["query"] = query
        if batch:
            payload["batch"] = batch
        if doc_id:
            payload["doc_id"] = doc_id
        if output:
            payload["output_dir"] = output
        enqueue(conn, stage="index", payload=payload, priority=100)
        console.print(f"[green]Search queued: {payload}[/green]")

    conn.commit()

    if wait:
        console.print("[dim]Waiting for daemon to process...[/dim]")
        while True:
            stats = get_queue_stats(conn)
            if stats.get("pending", 0) == 0 and stats.get("running", 0) == 0:
                break
            time.sleep(2)
        ctx.invoke(status)
```

- [ ] **Step 14: Write daemon smoke test**

Create `tests/test_daemon.py`:

```python
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
```

- [ ] **Step 15: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 16: Final commit**

```bash
git add unobfuscator.py tests/test_daemon.py
git commit -m "feat: daemon with cooperative shutdown, polling, and wired CLI"
```

---

*End of Chunk 6*

