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
    base = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    rows = conn.execute("""
        SELECT status, COUNT(*) as count FROM jobs GROUP BY status
    """).fetchall()
    base.update({row["status"]: row["count"] for row in rows})
    return base
