# Anchor Uniqueness Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate false recoveries caused by degenerate anchors while preserving all legitimate recoveries.

**Architecture:** Two guards added to the existing merger pipeline — an anchor quality floor in `merge_group()` that skips anchors with insufficient alphanumeric content, and a uniqueness check in `_find_between_exact()` that rejects ambiguous matches. Then a full re-merge of all groups.

**Tech Stack:** Python, SQLite, pytest

**Spec:** `docs/superpowers/specs/2026-03-27-anchor-uniqueness-design.md`

---

### File Structure

- **Modify:** `stages/merger.py` — add uniqueness check to `_find_between_exact()`, add quality floor to `merge_group()`
- **Modify:** `tests/stages/test_merger.py` — add tests for both new guards
- **No new files created**

---

### Task 1: Test the Uniqueness Check

**Files:**
- Modify: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test — ambiguous anchors return None**

Add to `tests/stages/test_merger.py`:

```python
def test_find_between_exact_rejects_ambiguous_match():
    """When anchor pair matches multiple positions in donor, return None."""
    # "Hello" ... "World" appears twice — ambiguous
    donor = "Hello dear friend World. Some filler. Hello dear enemy World."
    result = find_text_between_anchors(donor, "Hello", "World")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_find_between_exact_rejects_ambiguous_match -v`
Expected: FAIL — currently returns `"dear friend"` (first match)

- [ ] **Step 3: Write failing test — unique anchors still work**

Add to `tests/stages/test_merger.py`:

```python
def test_find_between_exact_accepts_unique_match():
    """When anchor pair matches exactly once, return the text between them."""
    donor = "Hello dear friend World. Some filler text here."
    result = find_text_between_anchors(donor, "Hello", "World")
    assert result == "dear friend"
```

- [ ] **Step 4: Run test to verify it passes (this should already pass)**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_find_between_exact_accepts_unique_match -v`
Expected: PASS (existing behavior)

- [ ] **Step 5: Write failing test — empty left anchor with ambiguous right anchor**

Add to `tests/stages/test_merger.py`:

```python
def test_find_between_exact_rejects_ambiguous_right_anchor_when_left_empty():
    """When left_anchor is empty and right_anchor matches multiple times, return None."""
    donor = "First section MARKER middle section MARKER end"
    result = find_text_between_anchors(donor, "", "MARKER")
    assert result is None
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_find_between_exact_rejects_ambiguous_right_anchor_when_left_empty -v`
Expected: FAIL — currently returns `"First section"`

---

### Task 2: Implement the Uniqueness Check

**Files:**
- Modify: `stages/merger.py:117-136`

- [ ] **Step 1: Add uniqueness check to `_find_between_exact`**

Replace the `_find_between_exact` function in `stages/merger.py` with:

```python
def _find_between_exact(
    text: str, left_anchor: str, right_anchor: str
) -> Optional[str]:
    """Exact string search for text between two anchors.

    Returns None if the anchor pair matches multiple positions (ambiguous).
    """
    if not left_anchor:
        right_pos = text.find(right_anchor)
        if right_pos == -1:
            return None
        # Uniqueness: check right_anchor doesn't appear again
        if text.find(right_anchor, right_pos + 1) != -1:
            return None
        return text[:right_pos].strip()

    left_pos = text.find(left_anchor)
    if left_pos == -1:
        return None

    search_from = left_pos + len(left_anchor)
    right_pos = text.find(right_anchor, search_from)
    if right_pos == -1:
        return None

    # Uniqueness: check the anchor pair doesn't match a second position
    second_left = text.find(left_anchor, left_pos + 1)
    if second_left != -1:
        second_right = text.find(right_anchor, second_left + len(left_anchor))
        if second_right != -1:
            return None  # ambiguous

    return text[search_from:right_pos].strip()
```

- [ ] **Step 2: Run all three new tests to verify they pass**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_find_between_exact_rejects_ambiguous_match tests/stages/test_merger.py::test_find_between_exact_accepts_unique_match tests/stages/test_merger.py::test_find_between_exact_rejects_ambiguous_right_anchor_when_left_empty -v`
Expected: All PASS

- [ ] **Step 3: Run full existing test suite to check for regressions**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py -v`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
cd /root/Unobfuscator && git add stages/merger.py tests/stages/test_merger.py && git commit -m "fix: add uniqueness check to _find_between_exact to reject ambiguous anchor matches"
```

---

### Task 3: Test the Anchor Quality Floor

**Files:**
- Modify: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test — consecutive redactions with punctuation-only anchors produce zero recoveries**

Add to `tests/stages/test_merger.py`:

```python
def test_merge_group_skips_redactions_with_degenerate_anchors(conn):
    """Consecutive redactions with only quote markers between them should not be recovered."""
    base_text = (
        "From: sender@example.com\n"
        ">>>>>> Employment Counselor\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>> [REDACTED]\n"
        ">>>>>>>\n"
        ">>>>>>>> On Jan 27, 2016, Michelle wrote:"
    )
    donor_text = (
        "From: sender@example.com\n"
        ">>>>>> Employment Counselor\n"
        ">>>>>> Regal Domestics, Inc.\n"
        ">>>>>> 123 Main Street\n"
        ">>>>>> New York, NY 10001\n"
        ">>>>>> (555) 123-4567\n"
        ">>>>>> michelle@regal.com\n"
        ">>>>>>>\n"
        ">>>>>>>> On Jan 27, 2016, Michelle wrote:"
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Should NOT recover all 5 with the same wrong text
    # The first redaction has a good left anchor ("Employment Counselor") so it may recover
    # The rest have only ">>>>>>" as anchors — too weak
    assert result["recovered_count"] <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_merge_group_skips_redactions_with_degenerate_anchors -v`
Expected: FAIL — currently recovers all 5

- [ ] **Step 3: Write test — boundary threshold (7 alphanumeric = skip, 8 = pass)**

Add to `tests/stages/test_merger.py`:

```python
def test_anchor_quality_floor_boundary(conn):
    """Anchors with exactly 7 alphanumeric chars are skipped; 8 chars pass."""
    # "Abcdefg" = 7 alpha chars on left, nothing on right -> should skip
    base_7 = "Abcdefg [REDACTED] end of text with no matching content"
    # "Abcdefgh" = 8 alpha chars on left, nothing on right -> should attempt recovery
    base_8 = "Abcdefgh [REDACTED] end of text with no matching content"
    donor = "Abcdefgh RECOVERED end of text with no matching content"

    seed_doc(conn, 1, base_7)
    seed_doc(conn, 2, donor)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()
    result_7 = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result_7["recovered_count"] == 0, "7 alphanumeric chars should be below quality floor"

    seed_doc(conn, 3, base_8)
    seed_doc(conn, 4, donor)
    conn.commit()
    g2 = create_match_group(conn)
    add_group_member(conn, g2, 3, 1.0)
    add_group_member(conn, g2, 4, 0.9)
    conn.commit()
    result_8 = merge_group(conn, g2, REDACTION_MARKERS, anchor_length=50)
    assert result_8["recovered_count"] == 1, "8 alphanumeric chars should pass quality floor"
```

- [ ] **Step 4: Run test to verify it fails (quality floor not yet implemented)**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_anchor_quality_floor_boundary -v`
Expected: FAIL (the 7-char case currently recovers)

---

### Task 4: Implement the Anchor Quality Floor

**Files:**
- Modify: `stages/merger.py:254-256`

- [ ] **Step 1: Add logger and quality floor check in `merge_group` loop**

In `stages/merger.py`, add a logger at the top of the file (after the existing imports):

```python
import logging

logger = logging.getLogger(__name__)
```

Then in the `merge_group` function, add the quality check after extracting anchors. Replace this block:

```python
    # Process in reverse order so string positions remain valid after substitution
    for pos, marker in reversed(positions):
        left_anchor, right_anchor = extract_anchors(base_text, pos, len(marker), anchor_length, redaction_markers)

        for donor_id, donor_text in donors:
```

With:

```python
    # Process in reverse order so string positions remain valid after substitution
    for pos, marker in reversed(positions):
        left_anchor, right_anchor = extract_anchors(base_text, pos, len(marker), anchor_length, redaction_markers)

        # Anchor quality floor: skip if combined anchors lack alphanumeric content
        alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
        if len(alpha_content) < 8:
            logger.debug("Skipping redaction at pos %d: anchor quality too low (%d alphanumeric chars)", pos, len(alpha_content))
            continue

        for donor_id, donor_text in donors:
```

- [ ] **Step 2: Run the degenerate anchors test to verify it passes**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_merge_group_skips_redactions_with_degenerate_anchors -v`
Expected: PASS

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /root/Unobfuscator && git add stages/merger.py tests/stages/test_merger.py && git commit -m "fix: add anchor quality floor to skip degenerate punctuation-only anchors"
```

---

### Task 5: Adversarial Tests

**Files:**
- Modify: `tests/stages/test_merger.py`

- [ ] **Step 1: Write adversarial test — table-based consecutive redactions**

```python
def test_merge_group_rejects_table_column_degenerate_anchors(conn):
    """Redactions in table columns with only pipe separators should not false-recover."""
    base_text = (
        "| Department | Contact |\n"
        "| East Coast | [REDACTED] |\n"
        "| West Coast | [REDACTED] |\n"
        "| Customer Service | [REDACTED] |\n"
    )
    donor_text = (
        "| Department | Contact |\n"
        "| East Coast | 555-0101 |\n"
        "| West Coast | 555-0102 |\n"
        "| Customer Service | 555-0103 |\n"
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Each row has unique left anchors ("East Coast |", "West Coast |", "Customer Service |")
    # so these SHOULD recover correctly — they have good anchors
    # But the right anchors are all "|\n|" which is ambiguous
    # The uniqueness check should catch if anchors match multiple positions
    recovered_texts = [s["text"] for s in result["recovered_segments"]]
    # No single text should appear more than once
    from collections import Counter
    counts = Counter(recovered_texts)
    for text, count in counts.items():
        assert count == 1, f"Text {repr(text)} recovered {count} times — likely false recovery"
```

- [ ] **Step 2: Run adversarial test**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_merge_group_rejects_table_column_degenerate_anchors -v`
Expected: PASS

- [ ] **Step 3: Write adversarial test — real group 7082 pattern**

```python
def test_merge_group_rejects_consecutive_redactions_real_pattern(conn):
    """Reproduce the group 7082 pattern: 56 consecutive [REDACTED] with >>>>> separators."""
    lines = [">>>>>> [REDACTED]"] * 20
    base_text = (
        "From: Michelle\nEmployment Counselor\nRegal Domestics, Inc.\n"
        + "\n".join(lines)
        + "\n>>>>>>>\n>>>>>>>> On Jan 27, 2016, at 1:05 PM, Michelle\n>> wrote:"
    )
    donor_text = (
        "From: Michelle\nEmployment Counselor\nRegal Domestics, Inc.\n"
        ">>>>>> 123 Main St\n>>>>>> Suite 100\n>>>>>> New York, NY\n"
        ">>>>>> (555) 111-2222\n>>>>>> michelle@regal.com\n"
        ">>>>>> Mon-Fri 9am-5pm\n>>>>>> Fax: (555) 111-3333\n"
        ">>>>>> Emergency: (555) 111-4444\n>>>>>> Regional Office\n"
        ">>>>>> PO Box 12345\n>>>>>> New York, NY 10001\n"
        ">>>>>> Licensed & Bonded\n>>>>>> Est. 1995\n"
        ">>>>>> www.regaldomestics.com\n>>>>>> info@regal.com\n"
        ">>>>>> NAICS: 561311\n>>>>>> EIN: 12-3456789\n"
        ">>>>>> Member: APSA\n>>>>>> Accredited: BBB A+\n"
        ">>>>>>>\n>>>>>>>> On Jan 27, 2016, at 1:05 PM, Michelle\n>> wrote:"
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Should NOT recover 20 copies of the same text
    recovered_texts = [s["text"] for s in result["recovered_segments"]]
    from collections import Counter
    counts = Counter(recovered_texts)
    for text, count in counts.items():
        assert count <= 1, f"Text {repr(text[:60])} recovered {count} times — false recovery"
    # The vast majority should be skipped due to degenerate anchors
    assert result["recovered_count"] <= 2
```

- [ ] **Step 4: Run adversarial test**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py::test_merge_group_rejects_consecutive_redactions_real_pattern -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /root/Unobfuscator && python -m pytest tests/stages/test_merger.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /root/Unobfuscator && git add tests/stages/test_merger.py && git commit -m "test: add adversarial tests for degenerate anchor false recovery patterns"
```

---

### Task 6: Live Data Validation

**Files:**
- No files modified — validation only

- [ ] **Step 1: Snapshot current recovery counts for comparison**

```bash
cd /root/Unobfuscator && python3 -c "
import sqlite3, json
from collections import Counter
conn = sqlite3.connect('data/unobfuscator.db')
conn.row_factory = sqlite3.Row
row = conn.execute('SELECT COUNT(*) as groups, SUM(recovered_count) as total FROM merge_results WHERE recovered_count > 0').fetchone()
print(f'BEFORE: {row[\"groups\"]} groups, {row[\"total\"]} total recoveries')
# Count suspect false recoveries
rows = conn.execute('SELECT group_id, recovered_segments FROM merge_results WHERE recovered_count > 3').fetchall()
false_count = 0
for r in rows:
    segs = json.loads(r['recovered_segments'])
    texts = [s['text'] for s in segs]
    counts = Counter(texts)
    mc_text, mc_count = counts.most_common(1)[0]
    if mc_count >= 3 and mc_count / len(texts) > 0.5:
        false_count += mc_count - 1
print(f'BEFORE: ~{false_count} estimated false recoveries')
"
```

- [ ] **Step 2: Reset all merge results**

```bash
cd /root/Unobfuscator && python3 -c "
import sqlite3
conn = sqlite3.connect('data/unobfuscator.db')
conn.execute('UPDATE match_groups SET merged = 0')
conn.execute('DELETE FROM merge_results')
conn.commit()
print('Reset complete')
print(f'Unmerged groups: {conn.execute(\"SELECT COUNT(*) FROM match_groups WHERE merged = 0\").fetchone()[0]}')
"
```

- [ ] **Step 3: Re-run the merger on all groups**

```bash
cd /root/Unobfuscator && python3 -c "
import sqlite3, sys, time, yaml
sys.path.insert(0, '.')
from stages.merger import run_merger

conn = sqlite3.connect('data/unobfuscator.db')
conn.row_factory = sqlite3.Row

with open('config.yaml') as f:
    config = yaml.safe_load(f)
markers = config.get('redaction_markers', ['[REDACTED]', '[b(6)]'])

start = time.time()
count = run_merger(conn, markers)
elapsed = time.time() - start
print(f'Merged {count} groups in {elapsed:.1f}s')
"
```

Expected: ~44K groups in ~10 minutes

- [ ] **Step 4: Validate — compare recovery counts and check for remaining false recoveries**

```bash
cd /root/Unobfuscator && python3 -c "
import sqlite3, json
from collections import Counter
conn = sqlite3.connect('data/unobfuscator.db')
conn.row_factory = sqlite3.Row
row = conn.execute('SELECT COUNT(*) as groups, SUM(recovered_count) as total FROM merge_results WHERE recovered_count > 0').fetchone()
print(f'AFTER: {row[\"groups\"]} groups, {row[\"total\"]} total recoveries')

# Check for remaining suspect patterns
rows = conn.execute('SELECT group_id, recovered_count, recovered_segments FROM merge_results WHERE recovered_count > 3').fetchall()
false_count = 0
suspect_groups = []
for r in rows:
    segs = json.loads(r['recovered_segments'])
    texts = [s['text'] for s in segs]
    counts = Counter(texts)
    mc_text, mc_count = counts.most_common(1)[0]
    if mc_count >= 3 and mc_count / len(texts) > 0.5:
        false_count += mc_count - 1
        suspect_groups.append((r['group_id'], mc_count, repr(mc_text[:60])))

print(f'AFTER: ~{false_count} estimated false recoveries')
if suspect_groups:
    print('Remaining suspect groups:')
    for gid, count, text in suspect_groups[:10]:
        print(f'  Group {gid}: {count}x {text}')
else:
    print('No remaining suspect groups!')
"
```

Expected: ~0 false recoveries, total recoveries slightly lower than before (lost ambiguous ones)

- [ ] **Step 5: Spot-check group 7082 specifically**

```bash
cd /root/Unobfuscator && python3 -c "
import sqlite3, json
conn = sqlite3.connect('data/unobfuscator.db')
conn.row_factory = sqlite3.Row
row = conn.execute('SELECT recovered_count, total_redacted, recovered_segments FROM merge_results WHERE group_id = 7082').fetchone()
if row:
    print(f'Group 7082: {row[\"recovered_count\"]}/{row[\"total_redacted\"]} recovered')
    segs = json.loads(row['recovered_segments'])
    for s in segs:
        print(f'  {repr(s[\"text\"][:80])}')
else:
    print('Group 7082: no recoveries (all skipped)')
"
```

Expected: 0-2 recoveries (not 57)

- [ ] **Step 6: Commit data validation results as a note, then final commit**

Note: Use the actual BEFORE/AFTER numbers from Steps 1 and 4 in the commit message:

```bash
cd /root/Unobfuscator && git commit -m "fix: eliminate false recoveries from degenerate anchors

Add anchor quality floor (min 8 alphanumeric chars) and donor uniqueness
check to reject ambiguous anchor matches. Re-merged all groups.

BEFORE: [X] recoveries, ~[Y] false. AFTER: [A] recoveries, ~[B] false."
```

---

### Task 7: Reset Output Generation Flags

**Files:**
- No code changes — database update only

- [ ] **Step 1: Reset output_generated for affected groups so PDFs regenerate**

```bash
cd /root/Unobfuscator && python3 -c "
import sqlite3
conn = sqlite3.connect('data/unobfuscator.db')
# Reset output flags so Stage 5 regenerates PDFs with corrected merge results
updated = conn.execute('UPDATE merge_results SET output_generated = 0 WHERE output_generated = 1').rowcount
conn.commit()
print(f'Reset output_generated on {updated} groups — PDFs will regenerate on next daemon cycle')
"
```

Expected: Resets output flags so the daemon regenerates corrected PDFs
