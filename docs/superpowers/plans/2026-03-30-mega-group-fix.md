# Mega-Group Fix: Sub-Cluster, Release Orphans, Prevent Re-Formation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the merger hang caused by mega-groups (60K-82K members) by sub-clustering genuine matches within them, releasing orphan documents for re-evaluation, and preventing transitive group merging from re-forming mega-groups.

**Architecture:** Three-layer fix: (1) Pre-merge clustering uses existing MinHash fingerprints + within-group LSH to find genuine sub-clusters in seconds, then merges each sub-cluster with the existing `merge_group()` logic unchanged. (2) Orphan documents (86% of mega-group members that have no genuine match) are released from the group so the LSH pipeline can re-evaluate them against the full corpus. (3) `_assign_to_group()` is modified to record cross-group pairs in a new `verified_pairs` table instead of calling `merge_groups()`, preventing transitive chaining from creating new mega-groups. A new merger step processes cross-group pairs for redaction recovery.

**Tech Stack:** Python 3.12, SQLite (WAL mode), datasketch (MinHash/MinHashLSH), numpy

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `core/db.py` | Modify | Add `verified_pairs` table to schema, add helper functions |
| `stages/merger.py` | Modify | Add `cluster_and_merge_group()`, cross-group pair merging |
| `stages/matcher.py` | Modify | Change `_assign_to_group()` to record pairs instead of merging groups |
| `tests/core/test_db.py` | Modify | Tests for verified_pairs CRUD |
| `tests/stages/test_merger.py` | Modify | Tests for sub-clustering and cross-group merging |
| `tests/stages/test_matcher.py` | Modify | Tests for the new `_assign_to_group()` behavior |

---

### Task 1: Add `verified_pairs` Table and DB Helpers

**Files:**
- Modify: `core/db.py:10-90` (schema), `core/db.py:157-191` (group helpers)
- Test: `tests/core/test_db.py`

- [ ] **Step 1: Write failing tests for verified_pairs CRUD**

Add to `tests/core/test_db.py`:

```python
from core.db import (
    insert_verified_pair, get_verified_pairs_for_doc,
    get_cross_group_pairs, get_unmerged_cross_group_pairs,
)


def test_insert_verified_pair_stores_record(conn):
    for i in [1, 2]:
        upsert_document(conn, {**SAMPLE_DOC, "id": i, "original_filename": f"{i}.pdf"})
    conn.commit()
    insert_verified_pair(conn, "1", "2", similarity=0.85, phase="phase3")
    conn.commit()
    pairs = get_verified_pairs_for_doc(conn, "1")
    assert len(pairs) == 1
    assert pairs[0]["doc_id_a"] == "1"
    assert pairs[0]["doc_id_b"] == "2"
    assert pairs[0]["similarity"] == 0.85
    assert pairs[0]["phase"] == "phase3"


def test_insert_verified_pair_normalizes_order(conn):
    """doc_id_a < doc_id_b regardless of insertion order."""
    for i in [1, 2]:
        upsert_document(conn, {**SAMPLE_DOC, "id": i, "original_filename": f"{i}.pdf"})
    conn.commit()
    insert_verified_pair(conn, "2", "1", similarity=0.9, phase="phase0")
    conn.commit()
    pairs = get_verified_pairs_for_doc(conn, "1")
    assert pairs[0]["doc_id_a"] == "1"
    assert pairs[0]["doc_id_b"] == "2"


def test_insert_verified_pair_ignores_duplicate(conn):
    for i in [1, 2]:
        upsert_document(conn, {**SAMPLE_DOC, "id": i, "original_filename": f"{i}.pdf"})
    conn.commit()
    insert_verified_pair(conn, "1", "2", similarity=0.85, phase="phase3")
    insert_verified_pair(conn, "1", "2", similarity=0.90, phase="phase3")
    conn.commit()
    pairs = get_verified_pairs_for_doc(conn, "1")
    assert len(pairs) == 1  # deduped


def test_get_cross_group_pairs_finds_pairs_in_different_groups(conn):
    for i in [1, 2]:
        upsert_document(conn, {**SAMPLE_DOC, "id": i, "original_filename": f"{i}.pdf"})
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, "1", 1.0)
    add_group_member(conn, g2, "2", 1.0)
    insert_verified_pair(conn, "1", "2", similarity=0.85, phase="phase3")
    conn.commit()
    pairs = get_cross_group_pairs(conn)
    assert len(pairs) == 1
    assert pairs[0]["doc_id_a"] == "1"
    assert pairs[0]["doc_id_b"] == "2"


def test_get_unmerged_cross_group_pairs_excludes_merged(conn):
    for i in [1, 2]:
        upsert_document(conn, {**SAMPLE_DOC, "id": i, "original_filename": f"{i}.pdf"})
    conn.commit()
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, "1", 1.0)
    add_group_member(conn, g2, "2", 1.0)
    insert_verified_pair(conn, "1", "2", similarity=0.85, phase="phase3")
    conn.commit()
    # Mark pair as merged
    conn.execute("UPDATE verified_pairs SET pair_merged = 1 WHERE doc_id_a = '1' AND doc_id_b = '2'")
    conn.commit()
    pairs = get_unmerged_cross_group_pairs(conn)
    assert len(pairs) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/core/test_db.py::test_insert_verified_pair_stores_record tests/core/test_db.py::test_insert_verified_pair_normalizes_order tests/core/test_db.py::test_insert_verified_pair_ignores_duplicate tests/core/test_db.py::test_get_cross_group_pairs_finds_pairs_in_different_groups tests/core/test_db.py::test_get_unmerged_cross_group_pairs_excludes_merged -v`

Expected: All FAIL with `ImportError` (functions don't exist yet).

- [ ] **Step 3: Add `verified_pairs` table to schema**

In `core/db.py`, add to the `SCHEMA` string after the `match_group_members` table (after line 48):

```python
CREATE TABLE IF NOT EXISTS verified_pairs (
    doc_id_a TEXT NOT NULL REFERENCES documents(id),
    doc_id_b TEXT NOT NULL REFERENCES documents(id),
    similarity REAL,
    phase TEXT NOT NULL,
    pair_merged BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id_a, doc_id_b),
    CHECK (doc_id_a < doc_id_b)
);

CREATE INDEX IF NOT EXISTS idx_verified_pairs_doc_a ON verified_pairs(doc_id_a);
CREATE INDEX IF NOT EXISTS idx_verified_pairs_doc_b ON verified_pairs(doc_id_b);
CREATE INDEX IF NOT EXISTS idx_verified_pairs_unmerged ON verified_pairs(pair_merged) WHERE pair_merged = 0;
```

Also add a migration in `_migrate_text_recovery_columns()` (or add a new migration function called from `init_db`) to create this table on existing databases:

```python
try:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verified_pairs (
            doc_id_a TEXT NOT NULL,
            doc_id_b TEXT NOT NULL,
            similarity REAL,
            phase TEXT NOT NULL,
            pair_merged BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (doc_id_a, doc_id_b),
            CHECK (doc_id_a < doc_id_b)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_pairs_doc_a ON verified_pairs(doc_id_a)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_pairs_doc_b ON verified_pairs(doc_id_b)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_pairs_unmerged ON verified_pairs(pair_merged) WHERE pair_merged = 0")
except Exception:
    pass
```

- [ ] **Step 4: Add helper functions**

In `core/db.py`, add after the `merge_groups()` function (after line 191):

```python
def insert_verified_pair(conn, doc_a: str, doc_b: str, similarity: float, phase: str) -> None:
    """Record a verified document pair. Order is normalized (a < b)."""
    a, b = (str(doc_a), str(doc_b)) if str(doc_a) < str(doc_b) else (str(doc_b), str(doc_a))
    conn.execute("""
        INSERT OR IGNORE INTO verified_pairs (doc_id_a, doc_id_b, similarity, phase)
        VALUES (?, ?, ?, ?)
    """, (a, b, similarity, phase))


def get_verified_pairs_for_doc(conn, doc_id: str) -> list[dict]:
    """Return all verified pairs involving a document."""
    rows = conn.execute("""
        SELECT * FROM verified_pairs
        WHERE doc_id_a = ? OR doc_id_b = ?
    """, (str(doc_id), str(doc_id))).fetchall()
    return [dict(r) for r in rows]


def get_cross_group_pairs(conn) -> list[dict]:
    """Return verified pairs where the two docs are in different groups."""
    rows = conn.execute("""
        SELECT vp.*, ma.group_id AS group_a, mb.group_id AS group_b
        FROM verified_pairs vp
        JOIN match_group_members ma ON ma.doc_id = vp.doc_id_a
        JOIN match_group_members mb ON mb.doc_id = vp.doc_id_b
        WHERE ma.group_id != mb.group_id
    """).fetchall()
    return [dict(r) for r in rows]


def get_unmerged_cross_group_pairs(conn) -> list[dict]:
    """Return unmerged verified pairs where the two docs are in different groups."""
    rows = conn.execute("""
        SELECT vp.*, ma.group_id AS group_a, mb.group_id AS group_b
        FROM verified_pairs vp
        JOIN match_group_members ma ON ma.doc_id = vp.doc_id_a
        JOIN match_group_members mb ON mb.doc_id = vp.doc_id_b
        WHERE ma.group_id != mb.group_id
          AND vp.pair_merged = 0
    """).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 5: Update imports in test file**

Add to the imports at the top of `tests/core/test_db.py`:

```python
from core.db import (
    insert_verified_pair, get_verified_pairs_for_doc,
    get_cross_group_pairs, get_unmerged_cross_group_pairs,
)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/core/test_db.py -v`

Expected: All tests PASS including the 5 new ones. All existing tests still pass.

- [ ] **Step 7: Commit**

```bash
cd /root/Unobfuscator
git add core/db.py tests/core/test_db.py
git commit -m "feat: add verified_pairs table for cross-group pair tracking"
```

---

### Task 2: Modify `_assign_to_group()` to Record Pairs Instead of Merging Groups

**Files:**
- Modify: `stages/matcher.py:732-748`
- Test: `tests/stages/test_matcher.py`

- [ ] **Step 1: Write failing test for the new behavior**

Add to `tests/stages/test_matcher.py`. First check existing test structure and imports, then add:

```python
from core.db import (
    create_match_group, add_group_member, get_doc_group,
    insert_verified_pair, get_verified_pairs_for_doc,
)
from stages.matcher import _assign_to_group


def _seed_doc(conn, doc_id, text="sample text"):
    from core.db import upsert_document
    upsert_document(conn, {
        "id": doc_id, "source": "test", "release_batch": "VOL00001",
        "original_filename": f"{doc_id}.pdf", "page_count": 1,
        "size_bytes": 100, "description": "", "extracted_text": text,
    })


def test_assign_to_group_records_pair_instead_of_merging(conn):
    """When both docs are in different groups, record a verified pair, don't merge."""
    _seed_doc(conn, "a")
    _seed_doc(conn, "b")
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, "a", 1.0)
    add_group_member(conn, g2, "b", 1.0)
    conn.commit()

    _assign_to_group(conn, "a", "b", similarity=0.85)
    conn.commit()

    # Groups should NOT have been merged
    assert get_doc_group(conn, "a") == g1
    assert get_doc_group(conn, "b") == g2

    # But a verified pair should be recorded
    pairs = get_verified_pairs_for_doc(conn, "a")
    assert len(pairs) == 1
    assert pairs[0]["similarity"] == 0.85


def test_assign_to_group_still_adds_to_existing_group(conn):
    """When only one doc is grouped, add the other to that group (unchanged behavior)."""
    _seed_doc(conn, "a")
    _seed_doc(conn, "b")
    g1 = create_match_group(conn)
    add_group_member(conn, g1, "a", 1.0)
    conn.commit()

    _assign_to_group(conn, "a", "b", similarity=0.9)
    conn.commit()

    assert get_doc_group(conn, "b") == g1


def test_assign_to_group_still_creates_new_group(conn):
    """When neither doc is grouped, create a new group (unchanged behavior)."""
    _seed_doc(conn, "a")
    _seed_doc(conn, "b")
    conn.commit()

    _assign_to_group(conn, "a", "b", similarity=0.75)
    conn.commit()

    assert get_doc_group(conn, "a") is not None
    assert get_doc_group(conn, "a") == get_doc_group(conn, "b")
```

NOTE: You will need to adapt these tests to match the existing test file's fixture pattern. Read `tests/stages/test_matcher.py` to check the existing `conn` fixture and imports before writing.

- [ ] **Step 2: Run tests to verify the first test fails**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_matcher.py::test_assign_to_group_records_pair_instead_of_merging -v`

Expected: FAIL — `_assign_to_group` currently calls `merge_groups()`.

- [ ] **Step 3: Modify `_assign_to_group()` in `stages/matcher.py`**

Replace lines 732-748 with:

```python
def _assign_to_group(conn, doc_a, doc_b, similarity: float) -> None:
    """Assign two documents to a shared match group.

    When both docs are already in different groups, record a verified pair
    instead of merging the groups. This prevents transitive chaining that
    creates mega-groups (82K+ members) from unrelated documents.
    """
    from core.db import insert_verified_pair

    group_a = get_doc_group(conn, doc_a)
    group_b = get_doc_group(conn, doc_b)

    if group_a is not None and group_b is not None:
        if group_a != group_b:
            # Record cross-group pair instead of merging groups
            insert_verified_pair(conn, doc_a, doc_b, similarity, phase="match")
        # else: same group, nothing to do
    elif group_a is not None:
        add_group_member(conn, group_a, doc_b, similarity)
    elif group_b is not None:
        add_group_member(conn, group_b, doc_a, similarity)
    else:
        new_group = create_match_group(conn)
        add_group_member(conn, new_group, doc_a, 1.0)
        add_group_member(conn, new_group, doc_b, similarity)
```

Also update the imports at the top of `matcher.py` (line 19-22): add `insert_verified_pair` if not already imported via the local import above (the local import is fine and avoids circular issues).

- [ ] **Step 4: Run all tests to verify**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_matcher.py tests/core/test_db.py -v`

Expected: All PASS. The `test_merge_groups_reassigns_members` test in `test_db.py` still passes because it tests `merge_groups()` directly, not `_assign_to_group()`.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All tests PASS. No existing test should call `_assign_to_group` expecting merge behavior.

- [ ] **Step 6: Commit**

```bash
cd /root/Unobfuscator
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "feat: replace transitive group merging with cross-group pair tracking

_assign_to_group() no longer calls merge_groups() when both docs are
in different groups. Instead, it records the relationship in the new
verified_pairs table. This prevents the transitive chaining that
created mega-groups of 82K+ members from unrelated emails."
```

---

### Task 3: Add Sub-Clustering to the Merger for Large Groups

**Files:**
- Modify: `stages/merger.py:570-596` (run_merger)
- Modify: `stages/merger.py` (new functions)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for sub-cluster merge**

Add to `tests/stages/test_merger.py`:

```python
from stages.merger import cluster_and_merge_group

def test_cluster_and_merge_group_splits_unrelated_docs(conn):
    """A group with 2 unrelated sub-clusters should merge each independently."""
    # Sub-cluster 1: email about flights
    base_a = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight schedule\n\n"
        "The flight to [REDACTED] departs at 9am from Palm Beach."
    )
    donor_a = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight schedule\n\n"
        "The flight to Little St. James departs at 9am from Palm Beach."
    )
    # Sub-cluster 2: completely different email about dinner
    base_b = (
        "From: chef@example.com\nTo: staff@example.com\n"
        "Subject: Dinner menu\n\n"
        "Tonight's guest of honor is [REDACTED] arriving at 7pm."
    )
    donor_b = (
        "From: chef@example.com\nTo: staff@example.com\n"
        "Subject: Dinner menu\n\n"
        "Tonight's guest of honor is Prince Andrew arriving at 7pm."
    )
    seed_doc(conn, "a1", base_a)
    seed_doc(conn, "a2", donor_a)
    seed_doc(conn, "b1", base_b)
    seed_doc(conn, "b2", donor_b)
    conn.commit()

    g = create_match_group(conn)
    for doc_id in ["a1", "a2", "b1", "b2"]:
        add_group_member(conn, g, doc_id, 1.0)
    conn.commit()

    # Cluster threshold of 0.70 should split into 2 sub-clusters
    result = cluster_and_merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert "Little St. James" in result["merged_text"]
    assert "Prince Andrew" in result["merged_text"]
    assert result["recovered_count"] == 2


def test_cluster_and_merge_group_releases_orphans(conn):
    """Documents with no within-group match should be released (removed from group)."""
    # 2 related docs + 1 orphan
    base = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight\n\nThe passenger was [REDACTED] on the manifest."
    )
    donor = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight\n\nThe passenger was Bill Clinton on the manifest."
    )
    orphan = (
        "From: accountant@example.com\nTo: bank@example.com\n"
        "Subject: Wire transfer\n\nPlease transfer $50,000 to account [REDACTED]."
    )
    seed_doc(conn, "r1", base)
    seed_doc(conn, "r2", donor)
    seed_doc(conn, "orphan", orphan)
    conn.commit()

    g = create_match_group(conn)
    add_group_member(conn, g, "r1", 1.0)
    add_group_member(conn, g, "r2", 1.0)
    add_group_member(conn, g, "orphan", 1.0)
    conn.commit()

    result = cluster_and_merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)

    # Orphan should have been removed from the group
    orphan_group = conn.execute(
        "SELECT group_id FROM match_group_members WHERE doc_id = 'orphan'"
    ).fetchone()
    assert orphan_group is None, "Orphan doc should be released from group"

    # Related docs should still be in the group
    assert conn.execute(
        "SELECT group_id FROM match_group_members WHERE doc_id = 'r1'"
    ).fetchone() is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_cluster_and_merge_group_splits_unrelated_docs tests/stages/test_merger.py::test_cluster_and_merge_group_releases_orphans -v`

Expected: FAIL with `ImportError` (function doesn't exist yet).

- [ ] **Step 3: Implement `cluster_and_merge_group()`**

Add to `stages/merger.py` before `run_merger()` (before line 570). Add necessary imports at the top of the file:

```python
import struct
import numpy as np
from collections import defaultdict, deque
```

Then add the function:

```python
# Threshold for sub-clustering large groups.  Groups with more members
# than this are sub-clustered before merging to avoid O(n*d) donor iteration.
_CLUSTER_THRESHOLD = 50


def _load_group_fingerprints(conn, group_id: int, num_perm: int = 128) -> dict:
    """Load MinHash fingerprints for all members of a group."""
    from datasketch import MinHash
    rows = conn.execute("""
        SELECT f.doc_id, f.minhash_sig
        FROM match_group_members m
        JOIN document_fingerprints f ON f.doc_id = m.doc_id
        WHERE m.group_id = ?
    """, (group_id,)).fetchall()
    result = {}
    for row in rows:
        mh = MinHash(num_perm=num_perm)
        mh.hashvalues = np.frombuffer(row["minhash_sig"], dtype=np.uint64).copy()
        result[row["doc_id"]] = mh
    return result


def _find_clusters(fingerprints: dict, threshold: float = 0.70, num_perm: int = 128) -> tuple[list[set], set]:
    """Use LSH to find connected components (sub-clusters) within fingerprints.

    Returns (clusters, orphans) where clusters is a list of sets of doc_ids
    and orphans is the set of doc_ids with no match.
    """
    from datasketch import MinHashLSH

    if not fingerprints:
        return [], set()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for doc_id, mh in fingerprints.items():
        try:
            lsh.insert(doc_id, mh)
        except ValueError:
            pass  # duplicate key

    # Build adjacency graph
    graph = defaultdict(set)
    has_match = set()
    for doc_id, mh in fingerprints.items():
        neighbors = lsh.query(mh)
        for n in neighbors:
            if n != doc_id:
                graph[doc_id].add(n)
                graph[n].add(doc_id)
                has_match.add(doc_id)
                has_match.add(n)

    # Connected components via BFS
    visited = set()
    clusters = []
    for node in graph:
        if node in visited:
            continue
        cluster = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            cluster.add(n)
            queue.extend(graph[n] - visited)
        if len(cluster) >= 2:
            clusters.append(cluster)

    orphans = set(fingerprints.keys()) - has_match
    return clusters, orphans


def cluster_and_merge_group(
    conn, group_id: int, redaction_markers: list[str], anchor_length: int = 50,
    lsh_threshold: float = 0.70, num_perm: int = 128,
) -> dict:
    """Sub-cluster a large group by text similarity, merge each cluster, release orphans.

    1. Load pre-computed MinHash fingerprints for group members
    2. Build within-group LSH index to find genuine pairs
    3. Extract connected components as sub-clusters
    4. Merge each sub-cluster with existing merge_group() logic
    5. Remove orphan documents from the group
    6. Return aggregated results (best merged_text per sub-cluster)
    """
    member_ids = {row["doc_id"] for row in conn.execute(
        "SELECT doc_id FROM match_group_members WHERE group_id = ?", (group_id,)
    ).fetchall()}

    logger.info("Clustering group %d (%d members) before merge...", group_id, len(member_ids))

    fingerprints = _load_group_fingerprints(conn, group_id, num_perm=num_perm)
    # Members without fingerprints are treated as orphans
    members_without_fp = member_ids - set(fingerprints.keys())

    clusters, orphans = _find_clusters(fingerprints, threshold=lsh_threshold, num_perm=num_perm)
    orphans = orphans | members_without_fp

    logger.info("Group %d: %d sub-clusters, %d orphans (%.1f%%)",
                group_id, len(clusters), len(orphans),
                len(orphans) / max(len(member_ids), 1) * 100)

    # Merge each sub-cluster independently using temporary groups
    all_recovered_count = 0
    all_total_redacted = 0
    all_source_doc_ids = []
    all_recovered_segments = []
    best_merged_text = ""
    best_recovered = -1

    for cluster_doc_ids in clusters:
        # Create a temporary group for this cluster
        temp_group = conn.execute("INSERT INTO match_groups (merged) VALUES (0)").lastrowid
        for doc_id in cluster_doc_ids:
            conn.execute(
                "INSERT OR IGNORE INTO match_group_members (group_id, doc_id, similarity) VALUES (?, ?, 1.0)",
                (temp_group, doc_id)
            )

        result = merge_group(conn, temp_group, redaction_markers, anchor_length)

        all_recovered_count += result["recovered_count"]
        all_total_redacted += result["total_redacted"]
        all_source_doc_ids.extend(result["source_doc_ids"])
        all_recovered_segments.extend(result["recovered_segments"])

        # Track best merged text (the one with most recoveries)
        if result["recovered_count"] > best_recovered:
            best_recovered = result["recovered_count"]
            best_merged_text = result["merged_text"]

        # Clean up temporary group
        conn.execute("DELETE FROM match_group_members WHERE group_id = ?", (temp_group,))
        conn.execute("DELETE FROM match_groups WHERE group_id = ?", (temp_group,))

    # Release orphans: remove them from the original group
    if orphans:
        placeholders = ",".join("?" * len(orphans))
        conn.execute(
            f"DELETE FROM match_group_members WHERE group_id = ? AND doc_id IN ({placeholders})",
            (group_id, *orphans)
        )
        logger.info("Group %d: released %d orphan documents for re-evaluation",
                     group_id, len(orphans))

    return {
        "merged_text": best_merged_text,
        "recovered_count": all_recovered_count,
        "total_redacted": all_total_redacted,
        "source_doc_ids": all_source_doc_ids,
        "recovered_segments": all_recovered_segments,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_cluster_and_merge_group_splits_unrelated_docs tests/stages/test_merger.py::test_cluster_and_merge_group_releases_orphans -v`

Expected: Both PASS.

NOTE: These tests use small groups without pre-computed fingerprints, so `_load_group_fingerprints` will return empty dicts. The test documents need to have fingerprints seeded. If the tests fail because of missing fingerprints, add fingerprint seeding to the test setup:

```python
from datasketch import MinHash
from core.db import upsert_fingerprint

def _seed_doc_with_fingerprint(conn, doc_id, text, num_perm=128):
    """Seed a document and compute+store its MinHash fingerprint."""
    seed_doc(conn, doc_id, text)
    mh = MinHash(num_perm=num_perm)
    for shingle in [text[i:i+5] for i in range(len(text) - 4)]:
        mh.update(shingle.encode('utf-8'))
    upsert_fingerprint(conn, doc_id, mh.hashvalues.tobytes(), len(text) - 4)
```

Then use `_seed_doc_with_fingerprint` instead of `seed_doc` in the clustering tests.

- [ ] **Step 5: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /root/Unobfuscator
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: add sub-clustering for large match groups before merging

Groups with >50 members are sub-clustered using within-group LSH on
pre-computed MinHash fingerprints. Each genuine sub-cluster is merged
independently with existing merge_group() logic. Orphan documents
(no match within group) are released for re-evaluation by the LSH pipeline."
```

---

### Task 4: Wire Sub-Clustering into `run_merger()` and Add Cross-Group Pair Merging

**Files:**
- Modify: `stages/merger.py:570-596` (run_merger)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for run_merger routing large groups to clustering**

Add to `tests/stages/test_merger.py`:

```python
def test_run_merger_routes_large_group_to_clustering(conn):
    """Groups exceeding _CLUSTER_THRESHOLD should use cluster_and_merge_group."""
    # Create a group with enough members to trigger clustering
    from stages.merger import _CLUSTER_THRESHOLD

    # We need _CLUSTER_THRESHOLD + 1 docs to trigger clustering.
    # For the test, temporarily set the threshold low.
    import stages.merger as merger_mod
    original_threshold = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 3  # trigger on 4+ members

    try:
        # 2 related docs + 2 orphans
        base = (
            "From: jeffrey@example.com\nTo: pilot@example.com\n"
            "Subject: Flight\n\nPassenger [REDACTED] on the manifest."
        )
        donor = (
            "From: jeffrey@example.com\nTo: pilot@example.com\n"
            "Subject: Flight\n\nPassenger Bill Clinton on the manifest."
        )
        orphan1 = "From: a@b.com\nCompletely different content alpha."
        orphan2 = "From: c@d.com\nCompletely different content beta."

        _seed_doc_with_fingerprint(conn, "m1", base)
        _seed_doc_with_fingerprint(conn, "m2", donor)
        _seed_doc_with_fingerprint(conn, "m3", orphan1)
        _seed_doc_with_fingerprint(conn, "m4", orphan2)
        conn.commit()

        g = create_match_group(conn)
        for doc_id in ["m1", "m2", "m3", "m4"]:
            add_group_member(conn, g, doc_id, 1.0)
        conn.commit()

        count = run_merger(conn, REDACTION_MARKERS, anchor_length=50)
        assert count == 1

        # Group should be marked as merged
        row = conn.execute("SELECT merged FROM match_groups WHERE group_id = ?", (g,)).fetchone()
        assert row["merged"] == 1

        # Orphans should have been released
        orphan_rows = conn.execute(
            "SELECT doc_id FROM match_group_members WHERE group_id = ? AND doc_id IN ('m3', 'm4')",
            (g,)
        ).fetchall()
        assert len(orphan_rows) == 0
    finally:
        merger_mod._CLUSTER_THRESHOLD = original_threshold
```

- [ ] **Step 2: Write failing test for cross-group pair merging**

Add to `tests/stages/test_merger.py`:

```python
from stages.merger import run_cross_group_merger

def test_run_cross_group_merger_recovers_redactions(conn):
    """Cross-group pairs should recover redactions between docs in different groups."""
    base = (
        "The investigation found [REDACTED] at the scene on March 10. "
        "The evidence was collected by officer Johnson."
    )
    donor = (
        "The investigation found John Smith at the scene on March 10. "
        "The evidence was collected by officer Johnson."
    )
    seed_doc(conn, "x1", base)
    seed_doc(conn, "x2", donor)
    conn.commit()

    # Put them in separate groups
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, "x1", 1.0)
    add_group_member(conn, g2, "x2", 1.0)
    conn.commit()

    # Record the cross-group pair
    from core.db import insert_verified_pair
    insert_verified_pair(conn, "x1", "x2", similarity=0.9, phase="phase3")
    conn.commit()

    count = run_cross_group_merger(conn, REDACTION_MARKERS, anchor_length=50)
    assert count == 1

    # The pair should be marked as merged
    row = conn.execute(
        "SELECT pair_merged FROM verified_pairs WHERE doc_id_a = 'x1' AND doc_id_b = 'x2'"
    ).fetchone()
    assert row["pair_merged"] == 1

    # A merge result should exist for the group containing the redacted doc
    mr = conn.execute(
        "SELECT recovered_count FROM merge_results WHERE group_id = ?", (g1,)
    ).fetchone()
    assert mr is not None
    assert mr["recovered_count"] >= 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_run_merger_routes_large_group_to_clustering tests/stages/test_merger.py::test_run_cross_group_merger_recovers_redactions -v`

Expected: FAIL.

- [ ] **Step 4: Modify `run_merger()` to route large groups to clustering**

Replace `run_merger()` in `stages/merger.py` (lines 570-596):

```python
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
        member_count = conn.execute(
            "SELECT COUNT(*) FROM match_group_members WHERE group_id = ?", (group_id,)
        ).fetchone()[0]

        if member_count > _CLUSTER_THRESHOLD:
            logger.info("Group %d has %d members — using sub-cluster merge",
                        group_id, member_count)
            result = cluster_and_merge_group(conn, group_id, redaction_markers, anchor_length)
        else:
            result = merge_group(conn, group_id, redaction_markers, anchor_length)

        upsert_merge_result(
            conn, group_id,
            result["merged_text"],
            result["recovered_count"],
            result["total_redacted"],
            result["source_doc_ids"],
            recovered_segments=result.get("recovered_segments", [])
        )
        conn.execute(
            "UPDATE match_groups SET merged = 1 WHERE group_id = ?", (group_id,)
        )
        conn.commit()
        count += 1

    return count
```

- [ ] **Step 5: Implement `run_cross_group_merger()`**

Add to `stages/merger.py` after `run_merger()`:

```python
def run_cross_group_merger(conn, redaction_markers: list[str], anchor_length: int = 50) -> int:
    """Attempt redaction recovery on cross-group verified pairs.

    For each unmerged pair where docs are in different groups, load both
    documents and attempt anchor-based recovery. Updates the merge_result
    for the group containing the more-redacted document.

    Returns count of pairs processed.
    """
    from core.db import get_unmerged_cross_group_pairs

    pairs = get_unmerged_cross_group_pairs(conn)
    if not pairs:
        return 0

    logger.info("Processing %d cross-group verified pairs", len(pairs))
    count = 0

    for pair in pairs:
        doc_a_id = pair["doc_id_a"]
        doc_b_id = pair["doc_id_b"]
        group_a = pair["group_a"]
        group_b = pair["group_b"]

        text_a = conn.execute(
            "SELECT extracted_text FROM documents WHERE id = ?", (doc_a_id,)
        ).fetchone()
        text_b = conn.execute(
            "SELECT extracted_text FROM documents WHERE id = ?", (doc_b_id,)
        ).fetchone()
        if not text_a or not text_b:
            conn.execute(
                "UPDATE verified_pairs SET pair_merged = 1 WHERE doc_id_a = ? AND doc_id_b = ?",
                (doc_a_id, doc_b_id)
            )
            count += 1
            continue

        text_a = text_a["extracted_text"] or ""
        text_b = text_b["extracted_text"] or ""

        # Determine which doc is more redacted
        def redaction_count(text):
            return sum(text.count(m) for m in redaction_markers)

        count_a = redaction_count(text_a)
        count_b = redaction_count(text_b)

        if count_a == 0 and count_b == 0:
            # Neither has redactions — nothing to recover
            conn.execute(
                "UPDATE verified_pairs SET pair_merged = 1 WHERE doc_id_a = ? AND doc_id_b = ?",
                (doc_a_id, doc_b_id)
            )
            count += 1
            continue

        # Use more-redacted as base, less-redacted as donor
        if count_a >= count_b:
            base_id, base_text, donor_id, donor_text = doc_a_id, text_a, doc_b_id, text_b
            target_group = group_a
        else:
            base_id, base_text, donor_id, donor_text = doc_b_id, text_b, doc_a_id, text_a
            target_group = group_b

        # Get existing merge result for the target group (if any)
        existing = conn.execute(
            "SELECT merged_text, recovered_count, total_redacted, source_doc_ids, recovered_segments "
            "FROM merge_results WHERE group_id = ?", (target_group,)
        ).fetchone()

        if existing and existing["merged_text"]:
            # Use existing merged text as base (it may have prior recoveries)
            merged_base = existing["merged_text"]
            prior_count = existing["recovered_count"]
            prior_total = existing["total_redacted"]
            prior_sources = json.loads(existing["source_doc_ids"]) if existing["source_doc_ids"] else []
            prior_segments = json.loads(existing["recovered_segments"]) if existing["recovered_segments"] else []
        else:
            merged_base = base_text
            prior_count = 0
            prior_total = len(find_redaction_positions(base_text, redaction_markers))
            prior_sources = [base_id]
            prior_segments = []

        # Attempt recovery using the cross-group donor
        positions = find_redaction_positions(merged_base, redaction_markers)
        if not positions:
            conn.execute(
                "UPDATE verified_pairs SET pair_merged = 1 WHERE doc_id_a = ? AND doc_id_b = ?",
                (doc_a_id, doc_b_id)
            )
            count += 1
            continue

        recovered_count = 0
        merged = merged_base
        new_segments = []

        for pos, marker in reversed(positions):
            left_anchor, right_anchor = extract_anchors(
                merged, pos, len(marker), anchor_length, redaction_markers
            )
            alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
            if len(alpha_content) < 8:
                continue

            recovered = find_text_between_anchors(donor_text, left_anchor, right_anchor)
            if recovered and _is_real_recovery(recovered, redaction_markers):
                merged = merged[:pos] + recovered + merged[pos + len(marker):]
                recovered_count += 1
                new_segments.append({
                    "text": recovered,
                    "source_doc_id": donor_id,
                    "stage": "cross_group",
                    "confidence": "high",
                    "anchor_alpha_len": len(alpha_content),
                })

        if recovered_count > 0:
            all_sources = prior_sources + ([donor_id] if donor_id not in prior_sources else [])
            all_segments = prior_segments + new_segments
            upsert_merge_result(
                conn, target_group,
                merged,
                prior_count + recovered_count,
                prior_total,
                all_sources,
                recovered_segments=all_segments,
            )
            logger.info("Cross-group pair (%s, %s): recovered %d redactions for group %d",
                        doc_a_id, doc_b_id, recovered_count, target_group)

        conn.execute(
            "UPDATE verified_pairs SET pair_merged = 1 WHERE doc_id_a = ? AND doc_id_b = ?",
            (doc_a_id, doc_b_id)
        )
        count += 1

        if count % 100 == 0:
            conn.commit()

    conn.commit()
    return count
```

Add `import json` to the top of merger.py if not already present.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py -v`

Expected: All tests PASS.

- [ ] **Step 7: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
cd /root/Unobfuscator
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: wire sub-clustering into run_merger, add cross-group pair merger

run_merger() routes groups above _CLUSTER_THRESHOLD to
cluster_and_merge_group(). New run_cross_group_merger() processes
verified pairs where docs are in different groups, recovering
redactions without merging the groups themselves."
```

---

### Task 5: Wire Cross-Group Merger into the Daemon Loop

**Files:**
- Modify: `unobfuscator.py` (daemon loop, Stage 3 section)

- [ ] **Step 1: Read the daemon loop**

Read `unobfuscator.py` around lines 303-327 (Stage 3 section) to find the exact integration point.

- [ ] **Step 2: Add cross-group merger call after run_merger()**

In the Stage 3 section of `_run_one_cycle()`, after the `run_merger()` call and its logging, add:

```python
from stages.merger import run_cross_group_merger

# Cross-group pair merging: recover redactions between docs in different groups
cross_count = run_cross_group_merger(conn, redaction_markers, anchor_length)
if cross_count > 0:
    logger.info("Stage 3: processed %d cross-group pairs", cross_count)
```

Move the import to the top of the file with the other merger imports.

- [ ] **Step 3: Run the daemon smoke test**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/test_daemon.py -v`

Expected: PASS (or no daemon tests fail).

- [ ] **Step 4: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd /root/Unobfuscator
git add unobfuscator.py
git commit -m "feat: add cross-group pair merging to daemon Stage 3 loop"
```

---

### Task 6: Integration Test with Real Mega-Group Scenario

**Files:**
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write integration test simulating the mega-group scenario**

```python
def test_mega_group_scenario_end_to_end(conn):
    """Simulate the real scenario: 6 docs, 2 genuine clusters, 2 orphans, 1 cross-group pair.

    This exercises the full flow:
    1. run_merger sub-clusters the group, merges each cluster, releases orphans
    2. run_cross_group_merger recovers redactions from cross-group pairs
    """
    from core.db import insert_verified_pair
    import stages.merger as merger_mod
    original_threshold = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 3  # trigger clustering on 4+ members

    try:
        # Cluster A: flight docs (2 members)
        flight_redacted = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan\n\n"
            "Departing [REDACTED] at 0800. Passengers: Jeffrey, [REDACTED]."
        )
        flight_clean = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan\n\n"
            "Departing Palm Beach at 0800. Passengers: Jeffrey, Prince Andrew."
        )
        # Cluster B: legal docs (2 members)
        legal_redacted = (
            "CASE NO. 2005-0042\nPlaintiff: [REDACTED]\n"
            "The deposition was taken on March 10, 2005."
        )
        legal_clean = (
            "CASE NO. 2005-0042\nPlaintiff: Virginia Giuffre\n"
            "The deposition was taken on March 10, 2005."
        )
        # Orphans
        orphan1 = "Completely unrelated document about weather forecasts in Idaho."
        orphan2 = "Dinner reservation for 8 at restaurant Le Bernardin tonight."

        for doc_id, text in [
            ("f1", flight_redacted), ("f2", flight_clean),
            ("l1", legal_redacted), ("l2", legal_clean),
            ("o1", orphan1), ("o2", orphan2),
        ]:
            _seed_doc_with_fingerprint(conn, doc_id, text)
        conn.commit()

        # All in one mega-group (simulating transitive chaining)
        g = create_match_group(conn)
        for doc_id in ["f1", "f2", "l1", "l2", "o1", "o2"]:
            add_group_member(conn, g, doc_id, 1.0)
        conn.commit()

        # Also a doc in a separate group that has a verified cross-group pair with f1
        cross_donor_text = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan\n\n"
            "Departing Teterboro at 0800. Passengers: Jeffrey, Prince Andrew."
        )
        _seed_doc_with_fingerprint(conn, "cross1", cross_donor_text)
        conn.commit()
        g2 = create_match_group(conn)
        add_group_member(conn, g2, "cross1", 1.0)
        insert_verified_pair(conn, "f1", "cross1", similarity=0.8, phase="phase3")
        conn.commit()

        # Step 1: run_merger handles the mega-group via clustering
        merge_count = run_merger(conn, REDACTION_MARKERS, anchor_length=50)
        assert merge_count >= 1

        # Orphans released
        for orphan_id in ["o1", "o2"]:
            row = conn.execute(
                "SELECT group_id FROM match_group_members WHERE doc_id = ?", (orphan_id,)
            ).fetchone()
            assert row is None, f"Orphan {orphan_id} should be released"

        # Recoveries from within-group clustering
        mr = conn.execute(
            "SELECT recovered_count, merged_text FROM merge_results WHERE group_id = ?", (g,)
        ).fetchone()
        assert mr is not None
        assert mr["recovered_count"] >= 1  # At least some within-group recoveries

        # Step 2: cross-group merger picks up the verified pair
        cross_count = run_cross_group_merger(conn, REDACTION_MARKERS, anchor_length=50)
        assert cross_count >= 1

    finally:
        merger_mod._CLUSTER_THRESHOLD = original_threshold
```

- [ ] **Step 2: Run the integration test**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_mega_group_scenario_end_to_end -v`

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
cd /root/Unobfuscator
git add tests/stages/test_merger.py
git commit -m "test: add end-to-end integration test for mega-group fix"
```

---

### Task 7: File GitHub Issue and Update Memory

- [ ] **Step 1: Create GitHub issue for the mega-group bug**

```bash
gh issue create --title "Merger hangs on mega-groups (82K+ members) due to transitive group chaining" \
  --body "## Problem
Match groups with 60K-82K members are created by transitive chaining in \`_assign_to_group()\`. When Phase 0 or Phase 3 finds a pair where both docs are already in different groups, it calls \`merge_groups()\`, which transitively chains unrelated documents. Group 19334 has 82,396 members with 86% having <0.70 Jaccard similarity to any other member.

## Root Cause
\`_assign_to_group()\` (matcher.py:732-748) calls \`merge_groups()\` unconditionally when both docs are in different groups. Phase 0 email header matching creates large initial groups, and Phase 3 cross-group verification causes cascading merges.

## Fix
1. Sub-cluster large groups using within-group LSH before merging
2. Release orphan documents (no genuine match) for re-evaluation
3. Replace transitive group merging with cross-group pair tracking (verified_pairs table)

## Impact
- Groups 19334, 48654, 51709 (82K, 70K, 61K members) blocked all merger progress
- 70K+ orphan docs trapped, unable to match with their actual counterparts
- 9.6% of mega-group members have legitimate matches outside the group they're locked in"
```

- [ ] **Step 2: Update project memory**

Update `/root/.claude/projects/-root-Unobfuscator/memory/project_merger_bisect_fix.md` to mark the bisect fix as complete and reference this new fix.

Create a new memory file for this fix at `/root/.claude/projects/-root-Unobfuscator/memory/project_mega_group_fix.md`.

- [ ] **Step 3: Commit plan document**

```bash
cd /root/Unobfuscator
git add docs/superpowers/plans/2026-03-30-mega-group-fix.md
git commit -m "docs: add implementation plan for mega-group fix"
```
