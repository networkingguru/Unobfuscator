import pytest
import json
from core.db import (
    init_db, get_connection, upsert_document,
    create_match_group, add_group_member, upsert_merge_result,
    get_all_recovery_groups,
)


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


def seed_group(conn, group_docs, merged_text, recovered_count,
               segments, source="doj", batch="VOL00001"):
    """Seed a match group with documents, merge result, and segments."""
    for doc_id, text in group_docs:
        upsert_document(conn, {
            "id": doc_id, "source": source, "release_batch": batch,
            "original_filename": f"{doc_id}.pdf", "page_count": 1,
            "size_bytes": 500, "description": "", "extracted_text": text,
            "pdf_url": f"https://example.com/{doc_id}.pdf",
        })
    g = create_match_group(conn)
    for doc_id, _ in group_docs:
        add_group_member(conn, g, doc_id, 0.9)
    upsert_merge_result(conn, g, merged_text, recovered_count=recovered_count,
                        total_redacted=max(recovered_count, 1),
                        source_doc_ids=[d[0] for d in group_docs],
                        recovered_segments=segments)
    conn.commit()
    return g


def test_get_all_recovery_groups_returns_groups_with_recoveries(conn):
    seed_group(conn,
               group_docs=[("doc-a", "base"), ("doc-b", "donor")],
               merged_text="merged", recovered_count=1,
               segments=[{"text": "SARAH KELLEN", "source_doc_id": "doc-b", "stage": "merge"}])
    # Group with zero recoveries should be excluded
    seed_group(conn,
               group_docs=[("doc-c", "base2"), ("doc-d", "donor2")],
               merged_text="no recovery", recovered_count=0, segments=[])

    rows = get_all_recovery_groups(conn)
    assert len(rows) == 1
    assert rows[0]["recovered_count"] > 0


def test_get_all_recovery_groups_empty_db(conn):
    rows = get_all_recovery_groups(conn)
    assert rows == []
