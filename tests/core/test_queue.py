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
