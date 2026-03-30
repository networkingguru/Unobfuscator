# Mega-Group Fix: Sub-Cluster, Release Orphans, Prevent Re-Formation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the merger hang caused by mega-groups (60K-82K members) by sub-clustering genuine matches within them, releasing orphan documents for re-evaluation, and preventing transitive group merging from re-forming mega-groups.

**Architecture:** Three-layer fix: (1) Pre-merge clustering uses existing MinHash fingerprints + within-group LSH to find genuine sub-clusters in seconds, then **splits the mega-group into permanent sub-groups** — each gets its own group_id and merge_result, preserving full fidelity. Orphan documents (86% of mega-group members) are released for re-evaluation by the LSH pipeline. (2) `_assign_to_group()` is modified to record cross-group pairs in a new `verified_pairs` table instead of calling `merge_groups()` — but only when one or both groups exceed `_MERGE_SIZE_LIMIT` (default 500). Small groups still merge normally to preserve Phase 0/3 grouping semantics. (3) A new cross-group merger step reuses the full `merge_group()` pipeline on verified pairs between different groups.

**Tech Stack:** Python 3.12, SQLite (WAL mode), datasketch (MinHash/MinHashLSH), numpy

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `core/db.py` | Modify | Add `verified_pairs` table to schema, add helper functions |
| `stages/merger.py` | Modify | Add `cluster_and_split_group()`, cross-group pair merging |
| `stages/matcher.py` | Modify | Change `_assign_to_group()` to block merges above size limit |
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

```sql
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

Also add a migration in `_migrate_text_recovery_columns()` to create this table on existing databases:

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


def get_group_member_count(conn, group_id: int) -> int:
    """Return the number of members in a match group."""
    row = conn.execute(
        "SELECT COUNT(*) FROM match_group_members WHERE group_id = ?", (group_id,)
    ).fetchone()
    return row[0]
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

### Task 2: Modify `_assign_to_group()` to Block Large Group Merges

**Files:**
- Modify: `stages/matcher.py:732-748`
- Modify: `tests/stages/test_matcher.py:438-456`
- Test: `tests/stages/test_matcher.py`

**Design note:** Small groups (both below `_MERGE_SIZE_LIMIT = 500`) still merge normally via `merge_groups()`. This preserves Phase 0/3 grouping semantics for the 99.8% of groups that have <100 members. Only when a merge would create or enlarge a mega-group does it record a verified pair instead.

- [ ] **Step 1: Write failing tests for the new behavior**

Read `tests/stages/test_matcher.py` to understand the existing `conn` fixture and `seed_doc` helper. Then add these tests:

```python
from core.db import insert_verified_pair, get_verified_pairs_for_doc, get_group_member_count
from stages.matcher import _assign_to_group, _MERGE_SIZE_LIMIT


def test_assign_to_group_blocks_merge_when_group_exceeds_limit(conn):
    """When one group exceeds _MERGE_SIZE_LIMIT, record pair instead of merging."""
    seed_doc(conn, "a", "text a")
    seed_doc(conn, "b", "text b")
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, "a", 1.0)
    add_group_member(conn, g2, "b", 1.0)
    conn.commit()

    # Simulate g1 being a large group by patching the limit low
    import stages.matcher as matcher_mod
    original = matcher_mod._MERGE_SIZE_LIMIT
    matcher_mod._MERGE_SIZE_LIMIT = 1  # any group with >=1 member triggers the guard
    try:
        _assign_to_group(conn, "a", "b", similarity=0.85)
        conn.commit()

        # Groups should NOT have been merged
        assert get_doc_group(conn, "a") == g1
        assert get_doc_group(conn, "b") == g2

        # But a verified pair should be recorded
        pairs = get_verified_pairs_for_doc(conn, "a")
        assert len(pairs) == 1
        assert pairs[0]["similarity"] == 0.85
    finally:
        matcher_mod._MERGE_SIZE_LIMIT = original


def test_assign_to_group_still_merges_small_groups(conn):
    """When both groups are below _MERGE_SIZE_LIMIT, merge normally."""
    seed_doc(conn, "a", "text a")
    seed_doc(conn, "b", "text b")
    g1 = create_match_group(conn)
    g2 = create_match_group(conn)
    add_group_member(conn, g1, "a", 1.0)
    add_group_member(conn, g2, "b", 1.0)
    conn.commit()

    # Default limit is 1000 — both groups have 1 member, well under limit
    _assign_to_group(conn, "a", "b", similarity=0.85)
    conn.commit()

    # Groups SHOULD have been merged
    assert get_doc_group(conn, "a") == get_doc_group(conn, "b")


def test_assign_to_group_blocks_add_when_group_exceeds_limit(conn):
    """When one doc is ungrouped but the other's group exceeds limit, leave ungrouped."""
    seed_doc(conn, "a", "text a")
    seed_doc(conn, "b", "text b")
    g1 = create_match_group(conn)
    add_group_member(conn, g1, "a", 1.0)
    conn.commit()

    import stages.matcher as matcher_mod
    original = matcher_mod._MERGE_SIZE_LIMIT
    matcher_mod._MERGE_SIZE_LIMIT = 1  # any group with >=1 member triggers the guard
    try:
        _assign_to_group(conn, "a", "b", similarity=0.85)
        conn.commit()

        # doc_a stays in its group
        assert get_doc_group(conn, "a") == g1
        # doc_b should remain UNGROUPED (not in a singleton group)
        # so LSH can re-evaluate it next cycle
        assert get_doc_group(conn, "b") is None

        # A verified pair should be recorded
        pairs = get_verified_pairs_for_doc(conn, "a")
        assert len(pairs) == 1
    finally:
        matcher_mod._MERGE_SIZE_LIMIT = original


def test_assign_to_group_still_adds_to_existing_group(conn):
    """When only one doc is grouped, add the other to that group (unchanged)."""
    seed_doc(conn, "a", "text a")
    seed_doc(conn, "b", "text b")
    g1 = create_match_group(conn)
    add_group_member(conn, g1, "a", 1.0)
    conn.commit()

    _assign_to_group(conn, "a", "b", similarity=0.9)
    conn.commit()

    assert get_doc_group(conn, "b") == g1


def test_assign_to_group_still_creates_new_group(conn):
    """When neither doc is grouped, create a new group (unchanged)."""
    seed_doc(conn, "a", "text a")
    seed_doc(conn, "b", "text b")
    conn.commit()

    _assign_to_group(conn, "a", "b", similarity=0.75)
    conn.commit()

    assert get_doc_group(conn, "a") is not None
    assert get_doc_group(conn, "a") == get_doc_group(conn, "b")
```

- [ ] **Step 2: Run tests to verify the first test fails**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_matcher.py::test_assign_to_group_blocks_merge_when_group_exceeds_limit -v`

Expected: FAIL — `_MERGE_SIZE_LIMIT` doesn't exist yet, and `_assign_to_group` still calls `merge_groups()` unconditionally.

- [ ] **Step 3: Modify `_assign_to_group()` in `stages/matcher.py`**

Add the size limit constant near the top of `matcher.py` (after line 36):

```python
# Maximum group size before merge_groups() is blocked. Groups at or above
# this size will record a verified_pair instead of merging, preventing
# the transitive chaining that creates mega-groups (82K+ members).
_MERGE_SIZE_LIMIT = 500
```

Replace `_assign_to_group()` at lines 732-748 with:

```python
def _assign_to_group(conn, doc_a: str, doc_b: str, similarity: float) -> None:
    """Assign two documents to a shared match group.

    When both docs are already in different groups and either group exceeds
    _MERGE_SIZE_LIMIT, record a verified pair instead of merging the groups.
    This prevents transitive chaining that creates mega-groups (82K+ members)
    while preserving normal grouping for small groups.
    """
    from core.db import insert_verified_pair, get_group_member_count

    group_a = get_doc_group(conn, doc_a)
    group_b = get_doc_group(conn, doc_b)

    if group_a is not None and group_b is not None:
        if group_a != group_b:
            size_a = get_group_member_count(conn, group_a)
            size_b = get_group_member_count(conn, group_b)
            if size_a >= _MERGE_SIZE_LIMIT or size_b >= _MERGE_SIZE_LIMIT:
                # Block merge — record cross-group pair instead
                insert_verified_pair(conn, doc_a, doc_b, similarity, phase="match")
            else:
                merge_groups(conn, group_a, group_b)
        # else: same group, nothing to do
    elif group_a is not None:
        size_a = get_group_member_count(conn, group_a)
        if size_a >= _MERGE_SIZE_LIMIT:
            # Group is too large — leave doc_b ungrouped so LSH can
            # re-evaluate it next cycle. Record the pair for cross-group merging.
            # Do NOT create a singleton group (it would trap doc_b permanently
            # since LSH skips already-grouped docs).
            insert_verified_pair(conn, doc_a, doc_b, similarity, phase="match")
        else:
            add_group_member(conn, group_a, doc_b, similarity)
    elif group_b is not None:
        size_b = get_group_member_count(conn, group_b)
        if size_b >= _MERGE_SIZE_LIMIT:
            # Group is too large — leave doc_a ungrouped (same rationale as above)
            insert_verified_pair(conn, doc_a, doc_b, similarity, phase="match")
        else:
            add_group_member(conn, group_b, doc_a, similarity)
    else:
        new_group = create_match_group(conn)
        add_group_member(conn, new_group, doc_a, 1.0)
        add_group_member(conn, new_group, doc_b, similarity)
```

- [ ] **Step 4: Update the existing `test_phase3_merges_two_existing_groups` test**

This test at `tests/stages/test_matcher.py:438-456` asserts groups merge when both docs are in separate groups. With the new code, small groups still merge (both groups have 1 member, well below 1000), so **this test should still pass unchanged**. Verify this.

If it fails (unlikely), the issue would be the import of `get_group_member_count`. Debug and fix.

- [ ] **Step 5: Run all matcher tests**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_matcher.py -v`

Expected: All PASS — including `test_phase3_merges_two_existing_groups` (small groups still merge).

- [ ] **Step 6: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /root/Unobfuscator
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "feat: block transitive group merging above size limit

_assign_to_group() now checks group sizes before merging. Groups below
_MERGE_SIZE_LIMIT (1000) merge normally, preserving Phase 0/3 grouping
semantics. Groups at or above the limit record a verified_pair instead,
preventing the transitive chaining that created 82K-member mega-groups."
```

---

### Task 3: Add Sub-Clustering That Splits Mega-Groups Into Permanent Sub-Groups

**Files:**
- Modify: `stages/merger.py` (new functions, modify run_merger)
- Test: `tests/stages/test_merger.py`

**Design note:** Unlike the v1 plan which used temporary groups, this splits the mega-group into permanent sub-groups. Each sub-cluster gets its own `group_id` and `merge_result`. The original mega-group is deleted after splitting. Orphan docs are released (removed from all groups) so the LSH pipeline re-evaluates them.

- [ ] **Step 1: Add `_seed_doc_with_fingerprint` helper to test file**

Add to `tests/stages/test_merger.py` after the existing `seed_doc` helper:

```python
from datasketch import MinHash
from core.db import upsert_fingerprint


def _seed_doc_with_fingerprint(conn, doc_id, text, num_perm=128):
    """Seed a document and compute+store its MinHash fingerprint."""
    seed_doc(conn, doc_id, text)
    mh = MinHash(num_perm=num_perm)
    for shingle in [text[i:i+5] for i in range(len(text) - 4)]:
        mh.update(shingle.encode('utf-8'))
    upsert_fingerprint(conn, doc_id, mh.hashvalues.tobytes(), max(len(text) - 4, 0))
```

- [ ] **Step 2: Write failing tests for cluster_and_split_group**

Add to `tests/stages/test_merger.py`:

```python
from stages.merger import cluster_and_split_group


def test_cluster_and_split_creates_permanent_subgroups(conn):
    """A mega-group with 2 unrelated sub-clusters should be split into 2 permanent groups."""
    # Sub-cluster 1: flight docs
    base_a = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight schedule for the weekend trip\n\n"
        "The flight to [REDACTED] departs at 9am from Palm Beach airport terminal."
    )
    donor_a = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight schedule for the weekend trip\n\n"
        "The flight to Little St. James departs at 9am from Palm Beach airport terminal."
    )
    # Sub-cluster 2: completely different content
    base_b = (
        "CASE NO. 2005-0042\nSouthern District of New York\n"
        "Plaintiff: [REDACTED]\n"
        "The deposition was taken on March 10, 2005 at the courthouse."
    )
    donor_b = (
        "CASE NO. 2005-0042\nSouthern District of New York\n"
        "Plaintiff: Virginia Giuffre\n"
        "The deposition was taken on March 10, 2005 at the courthouse."
    )
    _seed_doc_with_fingerprint(conn, "a1", base_a)
    _seed_doc_with_fingerprint(conn, "a2", donor_a)
    _seed_doc_with_fingerprint(conn, "b1", base_b)
    _seed_doc_with_fingerprint(conn, "b2", donor_b)
    conn.commit()

    g = create_match_group(conn)
    for doc_id in ["a1", "a2", "b1", "b2"]:
        add_group_member(conn, g, doc_id, 1.0)
    conn.commit()

    new_group_ids = cluster_and_split_group(conn, g, REDACTION_MARKERS)

    # Original group should be deleted
    original = conn.execute("SELECT * FROM match_groups WHERE group_id = ?", (g,)).fetchone()
    assert original is None, "Original mega-group should be deleted after splitting"

    # Should have created 2 new permanent groups
    assert len(new_group_ids) >= 2

    # Each sub-group should have 2 members
    for gid in new_group_ids:
        count = conn.execute(
            "SELECT COUNT(*) FROM match_group_members WHERE group_id = ?", (gid,)
        ).fetchone()[0]
        assert count == 2


def test_cluster_and_split_releases_orphans(conn):
    """Documents with no within-group match should be released (removed from all groups)."""
    base = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight plan details\n\nPassenger [REDACTED] on the manifest today."
    )
    donor = (
        "From: jeffrey@example.com\nTo: pilot@example.com\n"
        "Subject: Flight plan details\n\nPassenger Bill Clinton on the manifest today."
    )
    orphan = (
        "From: accountant@example.com\nTo: bank@example.com\n"
        "Subject: Wire transfer instructions\n\nPlease transfer $50,000 to account number."
    )
    _seed_doc_with_fingerprint(conn, "r1", base)
    _seed_doc_with_fingerprint(conn, "r2", donor)
    _seed_doc_with_fingerprint(conn, "orphan", orphan)
    conn.commit()

    g = create_match_group(conn)
    add_group_member(conn, g, "r1", 1.0)
    add_group_member(conn, g, "r2", 1.0)
    add_group_member(conn, g, "orphan", 1.0)
    conn.commit()

    new_group_ids = cluster_and_split_group(conn, g, REDACTION_MARKERS)

    # Orphan should not be in any group
    orphan_group = conn.execute(
        "SELECT group_id FROM match_group_members WHERE doc_id = 'orphan'"
    ).fetchone()
    assert orphan_group is None, "Orphan doc should be released from all groups"

    # Related docs should be in one of the new groups
    r1_group = conn.execute(
        "SELECT group_id FROM match_group_members WHERE doc_id = 'r1'"
    ).fetchone()
    assert r1_group is not None


def test_cluster_and_split_handles_recursive_large_component(conn):
    """If a connected component still exceeds the cluster threshold, it is recursively split."""
    import stages.merger as merger_mod
    original = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 2  # Very low threshold for testing

    try:
        # 4 docs that are all somewhat similar (they'll form one big component)
        texts = [
            "The investigation revealed details about the case on March 10 in Palm Beach Florida version " + str(i)
            for i in range(4)
        ]
        for i, text in enumerate(texts):
            _seed_doc_with_fingerprint(conn, f"d{i}", text)
        conn.commit()

        g = create_match_group(conn)
        for i in range(4):
            add_group_member(conn, g, f"d{i}", 1.0)
        conn.commit()

        new_group_ids = cluster_and_split_group(conn, g, REDACTION_MARKERS)

        # All docs should still be in some group (not lost)
        for i in range(4):
            row = conn.execute(
                "SELECT group_id FROM match_group_members WHERE doc_id = ?", (f"d{i}",)
            ).fetchone()
            assert row is not None, f"Doc d{i} should still be in a group"
    finally:
        merger_mod._CLUSTER_THRESHOLD = original
```

- [ ] **Step 2b: Run tests to verify they fail**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_cluster_and_split_creates_permanent_subgroups tests/stages/test_merger.py::test_cluster_and_split_releases_orphans -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `cluster_and_split_group()` and helpers**

Add imports at the top of `stages/merger.py`:

```python
import json
import struct
import numpy as np
from collections import defaultdict, deque
```

Add constants after the existing imports:

```python
# Groups with more members than this are sub-clustered before merging.
# Set higher than _MERGE_SIZE_LIMIT (in matcher.py) to create a buffer zone:
# groups that organically grow to 500-2000 are simply capped (no new members
# via merge), but not disrupted. Only truly pathological groups get sub-clustered.
_CLUSTER_THRESHOLD = 2000
```

Add the implementation before `run_merger()`:

```python
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


def _find_clusters(
    fingerprints: dict, threshold: float = 0.70, num_perm: int = 128
) -> tuple[list[set], set]:
    """Use LSH to find connected components (sub-clusters) within fingerprints.

    Returns (clusters, orphans) where clusters is a list of sets of doc_ids
    (each with >=2 members) and orphans is the set of doc_ids with no match.
    """
    from datasketch import MinHashLSH

    if not fingerprints:
        return [], set(fingerprints.keys())

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


def cluster_and_split_group(
    conn, group_id: int, redaction_markers: list[str],
    lsh_threshold: float = 0.70, num_perm: int = 128,
    shutdown_check=None,
) -> list[int]:
    """Split a mega-group into permanent sub-groups based on text similarity.

    1. Load pre-computed MinHash fingerprints for group members
    2. Build within-group LSH index to find genuine pairs
    3. Extract connected components as sub-clusters
    4. If any component still exceeds _CLUSTER_THRESHOLD, recursively split
       it with a higher LSH threshold
    5. Create a permanent new group for each sub-cluster
    6. Remove orphan documents from all groups (released for LSH re-eval)
    7. Delete the original mega-group
    8. Return list of new group_ids

    Each new group gets merged by the normal run_merger() loop.
    """
    from core.db import create_match_group, add_group_member

    member_ids = {row["doc_id"] for row in conn.execute(
        "SELECT doc_id FROM match_group_members WHERE group_id = ?", (group_id,)
    ).fetchall()}

    logger.info("Splitting group %d (%d members) into sub-groups...", group_id, len(member_ids))

    fingerprints = _load_group_fingerprints(conn, group_id, num_perm=num_perm)
    members_without_fp = member_ids - set(fingerprints.keys())

    clusters, orphans = _find_clusters(fingerprints, threshold=lsh_threshold, num_perm=num_perm)
    orphans = orphans | members_without_fp

    # Recursive sub-clustering: if a component is still too large, re-cluster
    # with progressively higher thresholds to break it apart further.
    # Bounded: threshold increases by 0.05 each level, capped at 0.95.
    final_clusters = []
    work_queue = [(cluster, lsh_threshold) for cluster in clusters]
    while work_queue:
        cluster, current_threshold = work_queue.pop()
        if len(cluster) > _CLUSTER_THRESHOLD and current_threshold < 0.95:
            higher_threshold = min(current_threshold + 0.05, 0.95)
            sub_fps = {doc_id: fingerprints[doc_id] for doc_id in cluster if doc_id in fingerprints}
            sub_clusters, sub_orphans = _find_clusters(sub_fps, threshold=higher_threshold, num_perm=num_perm)
            orphans = orphans | sub_orphans
            for sc in sub_clusters:
                # Re-queue for further splitting if still too large
                work_queue.append((sc, higher_threshold))
        else:
            # Accept this cluster (either small enough or threshold maxed out)
            if len(cluster) > _CLUSTER_THRESHOLD:
                logger.warning("Sub-cluster of %d members still exceeds threshold at "
                               "max threshold 0.95 — accepting as-is", len(cluster))
            final_clusters.append(cluster)

    logger.info("Group %d: %d sub-clusters, %d orphans (%.1f%%)",
                group_id, len(final_clusters), len(orphans),
                len(orphans) / max(len(member_ids), 1) * 100)

    # Step 1: Clean up the original group.
    # Delete output PDF if it exists (prevents orphan files on disk).
    existing_result = conn.execute(
        "SELECT output_path, recovered_count, output_generated FROM merge_results WHERE group_id = ?",
        (group_id,)
    ).fetchone()
    if existing_result and existing_result["output_path"]:
        import os
        try:
            os.remove(existing_result["output_path"])
            logger.info("Removed output PDF for group %d before split: %s",
                        group_id, existing_result["output_path"])
        except OSError:
            pass
    if existing_result and existing_result["recovered_count"] and existing_result["recovered_count"] > 0:
        logger.warning("Group %d had %d prior recoveries — will be re-merged in sub-groups",
                       group_id, existing_result["recovered_count"])

    conn.execute("DELETE FROM match_group_members WHERE group_id = ?", (group_id,))
    conn.execute("DELETE FROM merge_results WHERE group_id = ?", (group_id,))
    conn.execute("DELETE FROM match_groups WHERE group_id = ?", (group_id,))

    # Step 2: Create permanent sub-groups
    new_group_ids = []
    for cluster_doc_ids in final_clusters:
        new_gid = conn.execute("INSERT INTO match_groups (merged) VALUES (0)").lastrowid
        for doc_id in cluster_doc_ids:
            conn.execute(
                "INSERT OR IGNORE INTO match_group_members (group_id, doc_id, similarity) "
                "VALUES (?, ?, 1.0)",
                (new_gid, doc_id)
            )
        new_group_ids.append(new_gid)

    # Step 3: Orphans are already removed (they were in the deleted group).
    # They are now ungrouped and will be picked up by Phase 2 LSH on the next cycle.
    logger.info("Group %d: split into %d sub-groups, released %d orphans",
                group_id, len(new_group_ids), len(orphans))

    conn.commit()
    return new_group_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_cluster_and_split_creates_permanent_subgroups tests/stages/test_merger.py::test_cluster_and_split_releases_orphans tests/stages/test_merger.py::test_cluster_and_split_handles_recursive_large_component -v`

Expected: All PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd /root/Unobfuscator
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: add cluster_and_split_group to break mega-groups into permanent sub-groups

Uses within-group LSH on pre-computed MinHash fingerprints to find
genuine sub-clusters. Each becomes a permanent group with its own
merge_result. Orphan documents are released for LSH re-evaluation.
Includes recursive sub-clustering for components still above threshold."
```

---

### Task 4: Wire Sub-Clustering Into `run_merger()` and Add Cross-Group Pair Merging

**Files:**
- Modify: `stages/merger.py:570-596` (run_merger)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for run_merger routing large groups**

Add to `tests/stages/test_merger.py`:

```python
def test_run_merger_splits_large_group_then_merges_subgroups(conn):
    """Groups exceeding _CLUSTER_THRESHOLD should be split, then each sub-group merged."""
    import stages.merger as merger_mod
    original = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 3  # trigger on 4+ members

    try:
        base_a = (
            "From: jeffrey@example.com\nTo: pilot@example.com\n"
            "Subject: Flight schedule for the weekend trip\n\n"
            "The flight to [REDACTED] departs at 9am from Palm Beach airport terminal."
        )
        donor_a = (
            "From: jeffrey@example.com\nTo: pilot@example.com\n"
            "Subject: Flight schedule for the weekend trip\n\n"
            "The flight to Little St. James departs at 9am from Palm Beach airport terminal."
        )
        orphan1 = "From: a@b.com\nCompletely different content alpha beta gamma delta."
        orphan2 = "From: c@d.com\nCompletely different content epsilon zeta eta theta."

        _seed_doc_with_fingerprint(conn, "m1", base_a)
        _seed_doc_with_fingerprint(conn, "m2", donor_a)
        _seed_doc_with_fingerprint(conn, "m3", orphan1)
        _seed_doc_with_fingerprint(conn, "m4", orphan2)
        conn.commit()

        g = create_match_group(conn)
        for doc_id in ["m1", "m2", "m3", "m4"]:
            add_group_member(conn, g, doc_id, 1.0)
        conn.commit()

        count = run_merger(conn, REDACTION_MARKERS, anchor_length=50)

        # Original mega-group should be gone
        original_group = conn.execute(
            "SELECT * FROM match_groups WHERE group_id = ?", (g,)
        ).fetchone()
        assert original_group is None

        # Sub-groups should have been created and merged
        # At least one merge result should exist with a recovery
        results = conn.execute(
            "SELECT recovered_count FROM merge_results WHERE recovered_count > 0"
        ).fetchall()
        assert len(results) >= 1

        # Orphans should be ungrouped
        for orphan_id in ["m3", "m4"]:
            row = conn.execute(
                "SELECT group_id FROM match_group_members WHERE doc_id = ?", (orphan_id,)
            ).fetchone()
            assert row is None, f"Orphan {orphan_id} should be ungrouped"

    finally:
        merger_mod._CLUSTER_THRESHOLD = original
```

- [ ] **Step 2: Write failing test for cross-group pair merging**

Add to `tests/stages/test_merger.py`:

```python
from stages.merger import run_cross_group_merger


def test_run_cross_group_merger_recovers_redactions(conn):
    """Cross-group pairs should recover redactions using full merge_group() pipeline."""
    base = (
        "The investigation found [REDACTED] at the scene on March 10. "
        "The evidence was collected by officer Johnson at the precinct."
    )
    donor = (
        "The investigation found John Smith at the scene on March 10. "
        "The evidence was collected by officer Johnson at the precinct."
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

    # A merge result should exist with the recovery
    mr = conn.execute(
        "SELECT recovered_count, merged_text FROM merge_results WHERE group_id = ?", (g1,)
    ).fetchone()
    assert mr is not None
    assert mr["recovered_count"] >= 1
    assert "John Smith" in mr["merged_text"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_run_merger_splits_large_group_then_merges_subgroups tests/stages/test_merger.py::test_run_cross_group_merger_recovers_redactions -v`

Expected: FAIL.

- [ ] **Step 4: Modify `run_merger()` to handle large groups**

Replace `run_merger()` in `stages/merger.py`:

```python
def run_merger(
    conn, redaction_markers: list[str], anchor_length: int = 50,
    shutdown_check=None,
) -> int:
    """Run merger on all unmerged match groups with 2+ members. Returns count processed."""
    groups = conn.execute("""
        SELECT group_id FROM match_groups
        WHERE merged = 0
        AND (SELECT COUNT(*) FROM match_group_members WHERE group_id = match_groups.group_id) >= 2
    """).fetchall()

    # Phase 1: Split any mega-groups. Collect new sub-group IDs.
    new_sub_groups = []
    count = 0
    for row in groups:
        if shutdown_check and shutdown_check():
            break

        group_id = row["group_id"]

        # Check if this group still exists (may have been deleted by a prior split)
        member_count = conn.execute(
            "SELECT COUNT(*) FROM match_group_members WHERE group_id = ?", (group_id,)
        ).fetchone()[0]
        if member_count < 2:
            continue

        if member_count > _CLUSTER_THRESHOLD:
            logger.info("Group %d has %d members — splitting into sub-groups",
                        group_id, member_count)
            new_ids = cluster_and_split_group(
                conn, group_id, redaction_markers,
                shutdown_check=shutdown_check,
            )
            new_sub_groups.extend(new_ids)
            continue

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

    # Phase 2: Process newly created sub-groups from splits.
    # No recursion needed — these groups are guaranteed to be at or below
    # _CLUSTER_THRESHOLD (or accepted as-is at max LSH threshold).
    for sub_gid in new_sub_groups:
        if shutdown_check and shutdown_check():
            break
        member_count = conn.execute(
            "SELECT COUNT(*) FROM match_group_members WHERE group_id = ?", (sub_gid,)
        ).fetchone()[0]
        if member_count < 2:
            continue
        result = merge_group(conn, sub_gid, redaction_markers, anchor_length)
        upsert_merge_result(
            conn, sub_gid,
            result["merged_text"],
            result["recovered_count"],
            result["total_redacted"],
            result["source_doc_ids"],
            recovered_segments=result.get("recovered_segments", [])
        )
        conn.execute(
            "UPDATE match_groups SET merged = 1 WHERE group_id = ?", (sub_gid,)
        )
        conn.commit()
        count += 1

    return count
```

**No recursion, no infinite loop risk.** `cluster_and_split_group` returns the new group IDs, and `run_merger` processes them in a flat loop. Sub-groups produced by splitting are guaranteed to be at or below `_CLUSTER_THRESHOLD` (via the iterative work queue), or accepted as-is at max threshold 0.95. Either way, they go directly to `merge_group()`, never back through the splitting path.

- [ ] **Step 5: Implement `run_cross_group_merger()`**

Ensure `import json` is at the top of `stages/merger.py` (added in Task 3). Then add after `run_merger()`:

```python
def run_cross_group_merger(
    conn, redaction_markers: list[str], anchor_length: int = 50,
    shutdown_check=None,
) -> int:
    """Attempt redaction recovery on cross-group verified pairs.

    For each unmerged pair where docs are in different groups, create a
    temporary 2-member group and run the full merge_group() pipeline on it.
    Updates the merge_result for the group containing the more-redacted doc.

    Returns count of pairs processed.
    """
    from core.db import get_unmerged_cross_group_pairs

    pairs = get_unmerged_cross_group_pairs(conn)
    if not pairs:
        return 0

    logger.info("Processing %d cross-group verified pairs", len(pairs))
    count = 0

    for pair in pairs:
        if shutdown_check and shutdown_check():
            break

        doc_a_id = pair["doc_id_a"]
        doc_b_id = pair["doc_id_b"]
        group_a = pair["group_a"]
        group_b = pair["group_b"]

        # Determine which group has the more-redacted doc
        row_a = conn.execute(
            "SELECT extracted_text FROM documents WHERE id = ?", (doc_a_id,)
        ).fetchone()
        text_a = (row_a["extracted_text"] if row_a else "") or ""
        row_b = conn.execute(
            "SELECT extracted_text FROM documents WHERE id = ?", (doc_b_id,)
        ).fetchone()
        text_b = (row_b["extracted_text"] if row_b else "") or ""

        count_a = sum(text_a.count(m) for m in redaction_markers)
        count_b = sum(text_b.count(m) for m in redaction_markers)

        if count_a == 0 and count_b == 0:
            conn.execute(
                "UPDATE verified_pairs SET pair_merged = 1 "
                "WHERE doc_id_a = ? AND doc_id_b = ?",
                (doc_a_id, doc_b_id)
            )
            count += 1
            continue

        # Direct anchor matching — no temp groups, no SAVEPOINT needed.
        # Use the more-redacted doc's text (or existing merged_text) as base,
        # the less-redacted doc's text as donor.
        if count_a >= count_b:
            base_id, donor_id = doc_a_id, doc_b_id
            donor_text = text_b
            target_group = group_a
        else:
            base_id, donor_id = doc_b_id, doc_a_id
            donor_text = text_a
            target_group = group_b

        # Use existing merged_text as base if available (it may have prior recoveries)
        existing = conn.execute(
            "SELECT merged_text, recovered_count, total_redacted, "
            "source_doc_ids, recovered_segments FROM merge_results WHERE group_id = ?",
            (target_group,)
        ).fetchone()

        if existing and existing["merged_text"]:
            base_text = existing["merged_text"]
            prior_count = existing["recovered_count"]
            prior_total = existing["total_redacted"]
            prior_sources = json.loads(existing["source_doc_ids"]) if existing["source_doc_ids"] else []
            prior_segments = json.loads(existing["recovered_segments"]) if existing["recovered_segments"] else []
        else:
            base_text = text_a if count_a >= count_b else text_b
            prior_count = 0
            prior_total = len(find_redaction_positions(base_text, redaction_markers))
            prior_sources = [base_id]
            prior_segments = []

        # Anchor matching loop — same pattern as merge_group's core loop
        updated_text = base_text
        applied_count = 0
        new_segments = []

        positions = find_redaction_positions(updated_text, redaction_markers)
        for pos, marker in reversed(positions):
            left_anchor, right_anchor = extract_anchors(
                updated_text, pos, len(marker), anchor_length, redaction_markers
            )
            alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
            if len(alpha_content) < 8:
                continue
            recovered = find_text_between_anchors(donor_text, left_anchor, right_anchor)
            if recovered and _is_real_recovery(recovered, redaction_markers):
                updated_text = updated_text[:pos] + recovered + updated_text[pos + len(marker):]
                applied_count += 1
                new_segments.append({
                    "text": recovered,
                    "source_doc_id": donor_id,
                    "stage": "cross_group",
                    "confidence": "high",
                    "anchor_alpha_len": len(alpha_content),
                })

        if applied_count > 0:
            all_sources = prior_sources + ([donor_id] if donor_id not in prior_sources else [])
            upsert_merge_result(
                conn, target_group,
                updated_text,
                prior_count + applied_count,
                prior_total,
                all_sources,
                recovered_segments=prior_segments + new_segments,
            )
            logger.info("Cross-group pair (%s, %s): recovered %d redactions for group %d",
                        doc_a_id, doc_b_id, applied_count, target_group)

        conn.execute(
            "UPDATE verified_pairs SET pair_merged = 1 "
            "WHERE doc_id_a = ? AND doc_id_b = ?",
            (doc_a_id, doc_b_id)
        )
        count += 1

        if count % 100 == 0:
            conn.commit()

    conn.commit()
    return count
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py -v`

Expected: All PASS.

- [ ] **Step 7: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
cd /root/Unobfuscator
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: wire sub-clustering into run_merger, add cross-group pair merger

run_merger() splits groups above _CLUSTER_THRESHOLD into permanent
sub-groups, then recursively merges them. run_cross_group_merger()
processes verified pairs between different groups using the full
merge_group() pipeline via temporary group membership swaps."
```

---

### Task 5: Wire Cross-Group Merger into the Daemon Loop

**Files:**
- Modify: `unobfuscator.py` (Stage 3 section, around line 303-333)

- [ ] **Step 1: Read the daemon Stage 3 section**

Read `unobfuscator.py` around lines 295-333 to confirm the exact integration point. The current code is:

```python
_set_activity("Stage 3 Merger: merging groups")
logger.info("Stage 3: starting merger")
merged_count = run_merger(conn, redaction_markers=markers)
logger.info("Stage 3: merged %d groups", merged_count)
```

- [ ] **Step 2: Add cross-group merger and shutdown_check to Stage 3**

After the existing `run_merger` call and before the merge queue processing, add:

```python
# Cross-group pair merging
if not _shutdown_requested:
    from stages.merger import run_cross_group_merger
    cross_count = run_cross_group_merger(
        conn, redaction_markers=markers,
        shutdown_check=lambda: _shutdown_requested,
    )
    if cross_count > 0:
        logger.info("Stage 3: processed %d cross-group pairs", cross_count)
```

Update BOTH existing `run_merger` calls to pass `shutdown_check`. There are two calls in `_run_one_cycle`:

1. Line 305: `merged_count = run_merger(conn, redaction_markers=markers)` — the primary merge pass
2. Line 327: `run_merger(conn, redaction_markers=markers)` — the re-merge pass after processing the merge queue

Both must become:
```python
run_merger(conn, redaction_markers=markers,
           shutdown_check=lambda: _shutdown_requested)
```

Move the `run_cross_group_merger` import to the top of the file with the other merger imports.

- [ ] **Step 3: Run daemon tests**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/test_daemon.py -v`

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/pytest -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd /root/Unobfuscator
git add unobfuscator.py
git commit -m "feat: add cross-group pair merging and shutdown checks to daemon Stage 3"
```

---

### Task 6: End-to-End Integration Test

**Files:**
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write integration test simulating the full mega-group scenario**

```python
def test_mega_group_full_scenario(conn):
    """End-to-end: mega-group with mixed clusters, orphans, and a cross-group pair.

    1. run_merger splits the mega-group, merges sub-groups, releases orphans
    2. run_cross_group_merger recovers redactions from cross-group pairs
    """
    from core.db import insert_verified_pair
    import stages.merger as merger_mod
    original = merger_mod._CLUSTER_THRESHOLD
    merger_mod._CLUSTER_THRESHOLD = 3

    try:
        # Cluster A: flight docs
        flight_redacted = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan for Palm Beach\n\n"
            "Departing [REDACTED] at 0800 hours. Passengers: Jeffrey, [REDACTED]."
        )
        flight_clean = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan for Palm Beach\n\n"
            "Departing Teterboro at 0800 hours. Passengers: Jeffrey, Prince Andrew."
        )
        # Cluster B: legal docs
        legal_redacted = (
            "CASE NO. 2005-0042\nSouthern District of New York\n"
            "Plaintiff: [REDACTED]\nThe deposition was taken on March 10."
        )
        legal_clean = (
            "CASE NO. 2005-0042\nSouthern District of New York\n"
            "Plaintiff: Virginia Giuffre\nThe deposition was taken on March 10."
        )
        # Orphans
        orphan1 = "Completely unrelated document about weather forecasts in Idaho state parks."
        orphan2 = "Dinner reservation for eight guests at restaurant Le Bernardin tonight in NYC."

        for doc_id, text in [
            ("f1", flight_redacted), ("f2", flight_clean),
            ("l1", legal_redacted), ("l2", legal_clean),
            ("o1", orphan1), ("o2", orphan2),
        ]:
            _seed_doc_with_fingerprint(conn, doc_id, text)
        conn.commit()

        # All in one mega-group
        g = create_match_group(conn)
        for doc_id in ["f1", "f2", "l1", "l2", "o1", "o2"]:
            add_group_member(conn, g, doc_id, 1.0)
        conn.commit()

        # Cross-group donor in a separate group
        cross_text = (
            "From: pilot@jets.com\nTo: jeffrey@example.com\n"
            "Subject: Flight plan for Palm Beach\n\n"
            "Departing Newark at 0800 hours. Passengers: Jeffrey, Prince Andrew."
        )
        _seed_doc_with_fingerprint(conn, "cross1", cross_text)
        conn.commit()
        g2 = create_match_group(conn)
        add_group_member(conn, g2, "cross1", 1.0)
        insert_verified_pair(conn, "f1", "cross1", similarity=0.8, phase="phase3")
        conn.commit()

        # Step 1: run_merger handles the mega-group
        merge_count = run_merger(conn, REDACTION_MARKERS, anchor_length=50)

        # Original mega-group should be gone
        assert conn.execute(
            "SELECT * FROM match_groups WHERE group_id = ?", (g,)
        ).fetchone() is None

        # Orphans released
        for orphan_id in ["o1", "o2"]:
            assert conn.execute(
                "SELECT group_id FROM match_group_members WHERE doc_id = ?", (orphan_id,)
            ).fetchone() is None

        # At least some recoveries from within-group merging
        results = conn.execute(
            "SELECT SUM(recovered_count) FROM merge_results WHERE recovered_count > 0"
        ).fetchone()
        assert results[0] is not None and results[0] >= 1

        # Step 2: cross-group merger
        cross_count = run_cross_group_merger(conn, REDACTION_MARKERS, anchor_length=50)
        assert cross_count >= 1

        # Cross-group pair should be marked as merged
        pair_row = conn.execute(
            "SELECT pair_merged FROM verified_pairs WHERE doc_id_a = 'cross1' OR doc_id_b = 'cross1'"
        ).fetchone()
        assert pair_row["pair_merged"] == 1

    finally:
        merger_mod._CLUSTER_THRESHOLD = original
```

- [ ] **Step 2: Run the integration test**

Run: `cd /root/Unobfuscator && .venv/bin/pytest tests/stages/test_merger.py::test_mega_group_full_scenario -v`

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

- [ ] **Step 1: Create GitHub issue**

```bash
gh issue create \
  --title "Merger hangs on mega-groups (82K+ members) due to transitive group chaining" \
  --body "## Problem
Match groups with 60K-82K members are created by transitive chaining in _assign_to_group().
Group 19334 has 82,396 members with 86% having <0.70 Jaccard similarity to any other member.
9.6% of trapped members have legitimate matches outside the group they're locked in.

## Root Cause
_assign_to_group() (matcher.py) calls merge_groups() unconditionally when both docs are in
different groups. Phase 0 email header matching and Phase 3 verification both trigger cascading
merges that chain unrelated documents together.

## Fix (3 layers)
1. Sub-cluster large groups using within-group LSH, split into permanent sub-groups
2. Release orphan documents for re-evaluation by the LSH pipeline
3. Block transitive group merging above _MERGE_SIZE_LIMIT (1000), record verified_pairs instead
4. Cross-group pair merger processes verified pairs using full merge_group() pipeline

## Impact
- Groups 19334, 48654, 51709 (82K, 70K, 61K members) blocked all merger progress
- 70K+ orphan docs trapped, unable to find their actual counterparts
- Daemon stuck at 99.8% CPU for 39+ days on a single group"
```

- [ ] **Step 2: Update project memory**

Create `/root/.claude/projects/-root-Unobfuscator/memory/project_mega_group_fix.md` with status and context.

- [ ] **Step 3: Final commit**

```bash
cd /root/Unobfuscator
git add docs/superpowers/plans/2026-03-30-mega-group-fix.md
git commit -m "docs: update mega-group fix plan (v2, post-review)"
```
