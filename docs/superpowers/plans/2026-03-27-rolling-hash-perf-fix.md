# Rolling Hash Performance Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the rolling hash seed explosion that causes timeouts (and silent false negatives) on repetitive and large documents by (A) stripping Unicode redaction blocks before hashing and (B) scaling `min_seg` with document size.

**Architecture:** Two changes to `find_longest_common_substring` in `stages/matcher.py`. First, collapse runs of `█` (U+2588, the OCR redaction character) to a single space in `strip_markers` — this eliminates the dominant source of seed collisions (~16K heavily-redacted docs). Second, scale `min_seg` from 10 up to a cap of 50 based on the shorter document's length — this bounds seed counts for large PDFs with repetitive boilerplate. Both changes are in the entry-point function; the rolling hash algorithm itself is unchanged.

**Tech Stack:** Python stdlib (`re`), existing test infrastructure

**Root cause:** The `█` character (Unicode full block) is OCR output for redacted content, but it's not in the `redaction_markers` config. A doc that's 97% `█` generates ~30K identical 10-char windows. Even with `_MAX_BUCKET=256`, two such docs produce ~7.7M seeds. For large PDFs (150K-250K chars) with moderate repetition, 18M+ seeds were observed, triggering the 30-second circuit breaker and causing silent false negatives.

**Conservative `min_seg` formula:** `min_seg = max(10, min(50, shorter_len // 2000))`

| Shorter doc length | min_seg | Effect |
|---|---|---|
| ≤ 20K | 10 | No change from current behavior |
| 50K | 25 | Reduces seed collisions ~2.5× |
| 100K | 50 (cap) | Reduces seed collisions ~5× |
| 200K | 50 (cap) | Same cap, still well below 200-char rejection threshold |

The cap of 50 is conservative: Phase 3 rejects pairs with < 200 chars of common text, so any shared passage worth keeping is at least 4× longer than the largest `min_seg`.

---

### Task 1: Strip `█` blocks in marker stripping

**Files:**
- Modify: `stages/matcher.py:406-409` (the `strip_markers` inner function)
- Test: `tests/stages/test_matcher.py`

- [ ] **Step 1: Write failing test for `█`-block stripping**

Add after line 339 in `tests/stages/test_matcher.py` (after `test_rolling_hash_consistent_with_dp`):

```python
def test_rolling_hash_handles_unicode_redaction_blocks():
    """Documents with long █ (U+2588) redaction blocks must not cause seed explosion."""
    import time
    # Simulate OCR-redacted docs: email header + massive █ block + small unique tail
    redacted_block = "█" * 20000
    a = "From: sender@example.com\nTo: recipient@example.com\n" + redacted_block + "\nUnique passage alpha about Teterboro departure logs."
    b = "From: sender@example.com\nTo: recipient@example.com\n" + redacted_block + "\nUnique passage bravo about Teterboro departure logs."
    t0 = time.monotonic()
    common = find_longest_common_substring(a, b, REDACTION_MARKERS)
    elapsed = time.monotonic() - t0
    # Must complete fast — the █ block should be collapsed, not hashed
    assert elapsed < 2.0, f"Took {elapsed:.1f}s on █-heavy docs"
    # Should still find the real shared text (email headers + "Teterboro departure logs")
    assert "sender@example.com" in common or "Teterboro" in common
```

- [ ] **Step 2: Run test to verify it fails or is slow**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/stages/test_matcher.py::test_rolling_hash_handles_unicode_redaction_blocks -v 2>&1 | tail -10`

Expected: Either exceeds 2.0s timeout assertion or takes noticeably long.

- [ ] **Step 3: Add `█`-block collapsing to `strip_markers`**

In `stages/matcher.py`, replace the `strip_markers` inner function (lines 406-409):

```python
    def strip_markers(t):
        for m in redaction_markers:
            t = t.replace(m, " ")
        # Collapse Unicode full-block redaction runs (█, U+2588) from OCR output
        t = re.sub(r"\u2588+", " ", t)
        return re.sub(r"\s+", " ", t).strip()
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/stages/test_matcher.py::test_rolling_hash_handles_unicode_redaction_blocks -v 2>&1 | tail -10`

Expected: PASS, completes in < 1s.

- [ ] **Step 5: Run existing rolling hash tests to check for regressions**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/stages/test_matcher.py -v -k "rolling_hash" 2>&1 | tail -15`

Expected: `test_rolling_hash_handles_unicode_redaction_blocks` PASS. The two pre-existing performance tests may still fail (they test non-█ repetition — Task 2 addresses that).

- [ ] **Step 6: Commit**

```bash
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "fix: collapse █ redaction blocks before rolling hash to prevent seed explosion

OCR output uses █ (U+2588) for redactions, but this character wasn't in
redaction_markers. A 30K-char █ block generates ~30K identical 10-char
windows, causing millions of seeds and circuit breaker timeouts. Collapsing
█ runs to spaces before hashing eliminates this — same as we already do
for string markers like [REDACTED]."
```

---

### Task 2: Scale `min_seg` with document size

**Files:**
- Modify: `stages/matcher.py:417-421` (the `min_seg` / routing section of `find_longest_common_substring`)
- Test: `tests/stages/test_matcher.py`

- [ ] **Step 1: Write failing test for adaptive `min_seg`**

Add after `test_rolling_hash_handles_unicode_redaction_blocks` in `tests/stages/test_matcher.py`:

```python
def test_rolling_hash_performance_large_docs_with_boilerplate():
    """Large docs (100K+) with repetitive boilerplate must complete in bounded time."""
    import time
    # Simulate legal PDF: repeated boilerplate header/footer per page + unique content
    boilerplate = "UNITED STATES DISTRICT COURT SOUTHERN DISTRICT OF NEW YORK " * 20  # ~1.2K per page
    pages_a = []
    pages_b = []
    for i in range(80):  # 80 pages ≈ 100K chars
        pages_a.append(boilerplate + f"Page {i} unique content alpha section {i * 7}. ")
        pages_b.append(boilerplate + f"Page {i} unique content bravo section {i * 13}. ")
    # 20 pages of genuinely shared deposition testimony
    shared_testimony = "The witness testified that the meeting occurred on March tenth at the residence. " * 15  # ~1.3K per page
    for i in range(20):
        shared_page = shared_testimony + f"Exhibit {i} referenced. "
        pages_a.append(shared_page)
        pages_b.append(shared_page)
    a = "\n".join(pages_a)
    b = "\n".join(pages_b)
    assert len(a) > 100000, f"Test doc too short: {len(a)}"
    t0 = time.monotonic()
    common = find_longest_common_substring(a, b, REDACTION_MARKERS)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"Took {elapsed:.1f}s on 100K+ boilerplate docs"
    # Should find the shared testimony
    assert "witness testified" in common
```

- [ ] **Step 2: Run test to verify it is slow**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/stages/test_matcher.py::test_rolling_hash_performance_large_docs_with_boilerplate -v 2>&1 | tail -10`

Expected: Either slow (> 5s) or times out.

- [ ] **Step 3: Scale `min_seg` based on document size**

In `stages/matcher.py`, replace lines 417-421:

```python
    # Collect all common substrings of length >= min_seg using DP
    min_seg = 10
    max_chars = 2000
    if len(a) > max_chars or len(b) > max_chars:
        return _collect_common_segments_rolling_hash(a, b, min_seg)
```

With:

```python
    # Collect all common substrings of length >= min_seg.
    # Scale min_seg with document size to bound rolling-hash seed counts.
    # Cap at 50 — well below the 200-char Phase 3 rejection threshold.
    shorter = min(len(a), len(b))
    min_seg = max(10, min(50, shorter // 2000))
    max_chars = 2000
    if len(a) > max_chars or len(b) > max_chars:
        return _collect_common_segments_rolling_hash(a, b, min_seg)
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/stages/test_matcher.py::test_rolling_hash_performance_large_docs_with_boilerplate -v 2>&1 | tail -10`

Expected: PASS, completes in < 5s.

- [ ] **Step 5: Run ALL rolling hash and performance tests**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/stages/test_matcher.py -v -k "rolling_hash" 2>&1 | tail -20`

Expected: ALL rolling hash tests PASS, including the two previously-failing performance tests (`test_rolling_hash_repeated_content_completes_fast` and `test_rolling_hash_performance_50k`).

- [ ] **Step 6: Commit**

```bash
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "fix: scale min_seg with document size to bound rolling hash seeds

Large docs (100K+) with repetitive boilerplate generated millions of seeds
at min_seg=10, triggering the 30s circuit breaker and silently dropping
matches. Scale min_seg from 10 up to 50 based on the shorter document's
length (shorter // 2000). The cap of 50 is well below the 200-char Phase 3
rejection threshold, so no fidelity loss."
```

---

### Task 3: Full regression test and production verification

**Files:**
- Test: `tests/stages/test_matcher.py` (full suite)
- Verify: production database

- [ ] **Step 1: Run the full test suite**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/ -v 2>&1 | tail -30`

Expected: All tests PASS. No regressions in Phase 0/2/3/merger/output tests.

- [ ] **Step 2: Run the rolling-hash consistency test to verify fidelity**

Run: `cd /root/Unobfuscator && .venv/bin/python3 -m pytest tests/stages/test_matcher.py::test_rolling_hash_consistent_with_dp -v 2>&1 | tail -10`

Expected: PASS — rolling hash and DP paths still produce equivalent results for the same input.

- [ ] **Step 3: Verify on production data — the previously-timed-out doc pair**

```bash
cd /root/Unobfuscator && .venv/bin/python3 -c "
import logging, time, re
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
from core.db import get_connection
from stages.matcher import find_longest_common_substring

conn = get_connection('./data/unobfuscator.db')
# Find two large docs in the 150K-250K range to test
rows = conn.execute('''
    SELECT id, extracted_text FROM documents
    WHERE extracted_text IS NOT NULL AND LENGTH(extracted_text) BETWEEN 150000 AND 250000
    LIMIT 2
''').fetchall()
if len(rows) == 2:
    markers = ['[REDACTED]', '[Redacted]', '[REDACTED TEXT]', '[REDACTED PER]', '[Sender information redacted]']
    t0 = time.monotonic()
    common = find_longest_common_substring(rows[0]['extracted_text'], rows[1]['extracted_text'], markers)
    elapsed = time.monotonic() - t0
    print(f'Docs: {rows[0][\"id\"]} ({len(rows[0][\"extracted_text\"]):,} chars) vs {rows[1][\"id\"]} ({len(rows[1][\"extracted_text\"]):,} chars)')
    print(f'Common text: {len(common):,} chars in {elapsed:.1f}s')
    assert elapsed < 30, f'Still timing out: {elapsed:.1f}s'
    print('PASS — no timeout')
else:
    print(f'Only found {len(rows)} docs in range — skipping')
conn.close()
"
```

Expected: Completes in < 30s (ideally < 5s). No circuit breaker timeout.

- [ ] **Step 4: Test a heavily-redacted doc pair from production**

```bash
cd /root/Unobfuscator && .venv/bin/python3 -c "
import time
from core.db import get_connection
from stages.matcher import find_longest_common_substring

conn = get_connection('./data/unobfuscator.db')
# Find two heavily-redacted HOUSE_OVERSIGHT docs
rows = conn.execute('''
    SELECT id, extracted_text FROM documents
    WHERE id LIKE 'HOUSE_OVERSIGHT%'
    AND extracted_text IS NOT NULL AND LENGTH(extracted_text) > 10000
    AND extracted_text LIKE '%████%'
    LIMIT 2
''').fetchall()
if len(rows) == 2:
    markers = ['[REDACTED]', '[Redacted]', '[REDACTED TEXT]', '[REDACTED PER]', '[Sender information redacted]']
    t0 = time.monotonic()
    common = find_longest_common_substring(rows[0]['extracted_text'], rows[1]['extracted_text'], markers)
    elapsed = time.monotonic() - t0
    print(f'Docs: {rows[0][\"id\"]} vs {rows[1][\"id\"]}')
    print(f'Common text: {len(common):,} chars in {elapsed:.2f}s')
    print('PASS' if elapsed < 2.0 else f'SLOW: {elapsed:.1f}s')
else:
    print(f'Only found {len(rows)} matching docs')
conn.close()
"
```

Expected: Completes in < 2s.

- [ ] **Step 5: Commit plan doc**

```bash
git add docs/superpowers/plans/2026-03-27-rolling-hash-perf-fix.md
git commit -m "docs: add rolling hash performance fix implementation plan"
```
