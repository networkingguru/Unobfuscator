# Enhanced Merge Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dramatically increase redaction recovery rates when near-duplicate unredacted copies exist, while maintaining zero false positives.

**Architecture:** Four complementary techniques layered into `stages/merger.py`:
1. **Reverse merge** — when a donor has 0 redactions + sufficient similarity/length, use it as the base
2. **Wider adaptive anchors** — when 50-char anchors fail uniqueness, retry at 100, 150, 200 chars
3. **Line-level alignment merge** — use `difflib.SequenceMatcher` on lines to align texts, confirm candidates via anchor context
4. **Multi-pass merge with chain walking** — up to 3 passes; each pass uses prior recoveries as anchor context for adjacent redactions; early exit when no progress

**Tech Stack:** Python stdlib (`difflib.SequenceMatcher`), existing `stages/merger.py`, `core/db.py`

**Review fixes incorporated:** All 16 issues from review cycle 1 + 4 issues from cycle 2 addressed. Key fixes: line-level alignment with char-level intra-line SequenceMatcher for multi-marker lines, confirmation uses AND logic with positional donor line index, tests redesigned to genuinely fail first.

---

### Task 1: Reverse Merge — Select Least-Redacted Base

When a group member has 0 redaction markers, sufficient text length (within 2x of the most-redacted member), and high similarity, use it as the merge base. Count redactions from the redacted members as `total_redacted` for reporting.

**Files:**
- Modify: `stages/merger.py:256-285` (merge_group function, base selection)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for reverse merge**

```python
def test_reverse_merge_uses_unredacted_doc_as_base(conn):
    """When a donor has 0 redactions, it becomes the base; merged text has no markers."""
    redacted_text = (
        "The attendees included [REDACTED] and Prince Andrew. "
        "The flight departed from [REDACTED] at 9am. "
        "They arrived at [REDACTED] by noon."
    )
    unredacted_text = (
        "The attendees included Bill Clinton and Prince Andrew. "
        "The flight departed from Palm Beach at 9am. "
        "They arrived at Little St. James by noon."
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, unredacted_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.95)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Unredacted doc becomes base — merged text IS the unredacted doc
    assert "[REDACTED]" not in result["merged_text"]
    assert "Bill Clinton" in result["merged_text"]
    assert "Palm Beach" in result["merged_text"]
    assert "Little St. James" in result["merged_text"]
    # total_redacted reflects the redacted member's count (for reporting)
    assert result["total_redacted"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py::test_reverse_merge_uses_unredacted_doc_as_base -v`
Expected: FAIL — current logic picks the redacted doc as base

- [ ] **Step 3: Write failing test for structural reverse merge (Issue #1 scenario)**

```python
def test_reverse_merge_recovers_when_normal_merge_fails(conn):
    """Docs with structural differences: using unredacted as base avoids anchor failures."""
    redacted_text = (
        "(U) Interview of [REDACTED]\n"
        "(U) Interview of [REDACTED]\n"
        "(U) Proffer of [REDACTED]\n"
        "EFTA02730496\n--- PAGE BREAK ---\n"
    )
    unredacted_text = (
        "(U) Interview of Miguel Monge\n"
        "(U) Interview of Darrius Dupree\n"
        "(U) Interview of John MARRUGO\n"
        "(U) Proffer of Jason Mojica\n"
        "(U) Proffer of Christian Perez\n"
        "(U) Proffer of Freddy Caraballo\n"
        "EFTA02730942\n--- PAGE BREAK ---\n"
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, unredacted_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.85)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert "[REDACTED]" not in result["merged_text"]
    assert "Jason Mojica" in result["merged_text"]
    # total_redacted comes from the redacted member
    assert result["total_redacted"] == 3
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py::test_reverse_merge_recovers_when_normal_merge_fails -v`
Expected: FAIL

- [ ] **Step 5: Write test that reverse merge does NOT trigger for poor structural matches**

```python
def test_reverse_merge_skips_when_length_mismatch(conn):
    """A short unredacted doc should not become base for a much longer redacted doc."""
    long_redacted = (
        "Section 1: The investigation found [REDACTED] at the scene.\n" * 20
    )
    short_unredacted = "Cover page: Case File 2005-0042\n"
    seed_doc(conn, 1, long_redacted)
    seed_doc(conn, 2, short_unredacted)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.3)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Should NOT use the short doc as base — falls back to standard merge
    assert result["total_redacted"] == 20
    # Merged text is based on the long redacted doc (standard behavior)
    assert "Cover page" not in result["merged_text"][:50]
```

- [ ] **Step 6: Implement reverse merge in merge_group**

In `stages/merger.py`, modify `merge_group`. Replace the base selection block (lines 271-285):

```python
    members = sorted(rows, key=redaction_count, reverse=True)

    if not members:
        return {"merged_text": "", "recovered_count": 0, "total_redacted": 0,
                "source_doc_ids": [], "recovered_segments": []}

    most_redacted = members[0]
    least_redacted = min(members, key=redaction_count)
    most_count = redaction_count(most_redacted)
    least_count = redaction_count(least_redacted)

    # Reverse merge: if a member has 0 redactions and similar text length,
    # use it as base. Guard against cover pages / unrelated short docs.
    reverse_merge = False
    if least_count == 0 and most_count > 0:
        most_len = len(most_redacted["extracted_text"] or "")
        least_len = len(least_redacted["extracted_text"] or "")
        if least_len > 0 and most_len > 0:
            ratio = min(most_len, least_len) / max(most_len, least_len)
            if ratio > 0.3:  # within ~3x length — reasonable structural match
                reverse_merge = True

    if reverse_merge:
        base_id = least_redacted["id"]
        base_text = least_redacted["extracted_text"] or ""
        donors = [(row["id"], row["extracted_text"] or "") for row in members
                  if row["id"] != base_id]
        # total_redacted from the most-redacted member (for reporting)
        total_redacted = most_count
    else:
        base_id = most_redacted["id"]
        base_text = most_redacted["extracted_text"] or ""
        donors = [(row["id"], row["extracted_text"] or "") for row in members[1:]]
        total_redacted = None  # computed below from positions
```

Then update the `total_redacted` computation after `find_redaction_positions`:

```python
    positions = find_redaction_positions(base_text, redaction_markers)
    if total_redacted is None:
        total_redacted = len(positions)
```

- [ ] **Step 7: Run all merger tests**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: reverse merge — use unredacted doc as base when available"
```

---

### Task 2: Wider Adaptive Anchors

When the default 50-char anchor fails, retry with 100, 150, 200 char anchors. Wider anchors are more unique, reducing ambiguity rejections. Zero false-positive risk.

**Files:**
- Modify: `stages/merger.py` (anchor matching loop in merge_group)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test with genuinely ambiguous short anchors**

```python
def test_wider_anchors_recover_when_short_anchors_are_ambiguous(conn):
    """50-char anchors are identical for two redactions; wider context resolves them."""
    # Both paragraphs share identical 50-char context around the redaction.
    # The distinguishing "January/downtown" vs "February/uptown" is beyond 50 chars.
    shared_pre = "the private counsel representing the individual "  # 48 chars
    shared_suf = " during the confidential deposition proceedings. "  # 50 chars
    base_text = (
        "In January in the downtown office, " + shared_pre + "[REDACTED]" + shared_suf
        + "In February in the uptown office, " + shared_pre + "[REDACTED]" + shared_suf
    )
    donor_text = (
        "In January in the downtown office, " + shared_pre + "Alan Dershowitz" + shared_suf
        + "In February in the uptown office, " + shared_pre + "Leslie Wexner" + shared_suf
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # At 50 chars both anchors are identical — ambiguous, rejected.
    # At 100 chars the "January/downtown" vs "February/uptown" resolves it.
    assert result["recovered_count"] == 2
    assert "Alan Dershowitz" in result["merged_text"]
    assert "Leslie Wexner" in result["merged_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py::test_wider_anchors_recover_when_short_anchors_are_ambiguous -v`
Expected: FAIL — 50-char anchors are ambiguous, both rejected

- [ ] **Step 3: Implement adaptive anchor widening**

In `stages/merger.py`, replace the per-redaction anchor+donor loop with a version that tries wider anchors on failure. Replace the block from the `for pos, marker in reversed(positions):` loop body:

```python
    for pos, marker in reversed(positions):
        left_anchor, right_anchor = extract_anchors(
            merged, pos, len(marker), anchor_length, redaction_markers
        )
        alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
        if len(alpha_content) < 8:
            logger.debug("Skipping redaction at pos %d: anchor quality too low (%d alphanumeric chars)",
                         pos, len(alpha_content))
            continue

        # Try progressively wider anchors if the default width fails
        anchor_widths = [anchor_length, 100, 150, 200]
        recovered_this = False
        for width in anchor_widths:
            if recovered_this:
                break
            if width == anchor_length:
                left_w, right_w = left_anchor, right_anchor
            else:
                left_w, right_w = extract_anchors(
                    merged, pos, len(marker), width, redaction_markers
                )

            for donor_id, donor_text in donors:
                recovered = find_text_between_anchors(donor_text, left_w, right_w)
                if recovered and _is_real_recovery(recovered, redaction_markers):
                    merged = merged[:pos] + recovered + merged[pos + len(marker):]
                    recovered_count += 1
                    alpha_w = re.sub(r'[^a-zA-Z0-9]', '', left_w + right_w)
                    recovered_segments.append({
                        "text": recovered,
                        "source_doc_id": donor_id,
                        "stage": "merge",
                        "confidence": "high",
                        "anchor_alpha_len": len(alpha_w),
                    })
                    if donor_id not in source_doc_ids:
                        source_doc_ids.append(donor_id)
                    recovered_this = True
                    break
```

Note: `anchor_alpha_len` now uses `alpha_w` (computed from the actual matching anchors), not the original `alpha_content`.

- [ ] **Step 4: Run all merger tests**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: adaptive anchor widening — retry with 100/150/200 char anchors on failure"
```

---

### Task 3: Line-Level Alignment Merge with Anchor Confirmation

For remaining unrecovered redactions, use `difflib.SequenceMatcher` on **lines** (not characters) to align base and donor texts. For each redaction marker in the base, read the aligned region from the donor as a candidate. Then validate using **both** left and right anchor context (AND, not OR).

Line-level alignment is O(n) in practice and handles 200KB documents in <20ms.

**Files:**
- Modify: `stages/merger.py` (add `_alignment_recover`, `_confirm_alignment_candidate`, integrate)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for alignment-based recovery**

```python
def test_alignment_merge_recovers_with_structural_differences(conn):
    """Alignment handles different page breaks and shifted content."""
    redacted_text = (
        "MEMORANDUM\n"
        "From: Office of General Counsel\n"
        "Re: Investigation Update\n\n"
        "1. The subject [REDACTED] was identified at the location.\n"
        "2. Witness [REDACTED] provided testimony on the record.\n"
        "EFTA0001\n--- PAGE BREAK ---\n"
        "3. Evidence collected from [REDACTED] confirmed the timeline.\n"
    )
    # Same content but different page breaks and an added date line
    unredacted_text = (
        "MEMORANDUM\n"
        "From: Office of General Counsel\n"
        "Date: March 15, 2005\n"
        "Re: Investigation Update\n\n"
        "1. The subject John Smith was identified at the location.\n"
        "2. Witness Sarah Johnson provided testimony on the record.\n"
        "3. Evidence collected from Palm Beach PD confirmed the timeline.\n"
        "EFTA0002\n--- PAGE BREAK ---\n"
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, unredacted_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.85)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] >= 2
    assert "John Smith" in result["merged_text"]
    assert "Sarah Johnson" in result["merged_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py::test_alignment_merge_recovers_with_structural_differences -v`
Expected: FAIL — anchor matching fails due to structural differences

- [ ] **Step 3: Write test that alignment does NOT false-recover unrelated content**

```python
def test_alignment_merge_does_not_false_recover_unrelated_content(conn):
    """Alignment rejects candidates where surrounding context doesn't match."""
    redacted_text = (
        "Section A: The investigation found [REDACTED] at the scene.\n"
        "Section B: The report concluded with no further action.\n"
    )
    donor_text = (
        "Section A: The budget review found surplus funds at the office.\n"
        "Section B: The report concluded with no further action.\n"
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.7)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert "surplus funds" not in result["merged_text"]
    assert "[REDACTED]" in result["merged_text"]
```

- [ ] **Step 4: Implement `_alignment_recover` (line-level)**

Add to `stages/merger.py`:

```python
def _alignment_recover(
    base_text: str,
    donor_text: str,
    positions: list[tuple[int, str]],
    redaction_markers: list[str],
) -> dict[int, tuple[str, int]]:
    """Use line-level SequenceMatcher to propose recoveries for redaction positions.

    Returns dict mapping redaction position -> (candidate text, donor_line_char_offset).
    The donor_line_char_offset enables positional confirmation instead of global search.
    Works on lines for O(n) performance (handles 200KB+ docs in <20ms).
    """
    from difflib import SequenceMatcher

    # Split into lines, tracking char offset of each line
    base_lines = base_text.splitlines(keepends=True)
    donor_lines = donor_text.splitlines(keepends=True)

    # Build line-offset maps: line_index -> char_offset
    base_offsets = []
    offset = 0
    for line in base_lines:
        base_offsets.append(offset)
        offset += len(line)

    donor_offsets = []
    offset = 0
    for line in donor_lines:
        donor_offsets.append(offset)
        offset += len(line)

    # Map each redaction position to its line index
    pos_to_line = {}
    for pos, marker in positions:
        for i, line_offset in enumerate(base_offsets):
            line_end = line_offset + len(base_lines[i])
            if line_offset <= pos < line_end:
                pos_to_line[pos] = i
                break

    # Align lines
    sm = SequenceMatcher(None, base_lines, donor_lines)
    opcodes = sm.get_opcodes()

    # Build line mapping: base_line_index -> donor_line_index for 'replace' blocks
    base_to_donor_line = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            # Map each base line in the block to the corresponding donor lines
            for bi in range(i1, i2):
                # Proportional mapping within the replace block
                ratio = (bi - i1) / max(i2 - i1, 1)
                dj = j1 + int(ratio * (j2 - j1))
                dj = min(dj, j2 - 1)
                base_to_donor_line[bi] = dj

    candidates = {}
    for pos, marker in positions:
        line_idx = pos_to_line.get(pos)
        if line_idx is None:
            continue

        donor_line_idx = base_to_donor_line.get(line_idx)
        if donor_line_idx is None:
            continue

        # Get the donor line and extract the text that replaces the marker
        base_line = base_lines[line_idx]
        donor_line = donor_lines[donor_line_idx]

        # Find the marker's position within the base line
        line_offset = base_offsets[line_idx]
        pos_in_line = pos - line_offset

        # Use char-level SequenceMatcher on the individual line pair.
        # This handles multi-marker lines correctly (where prefix/suffix
        # matching breaks) and is fast since individual lines are short.
        line_sm = SequenceMatcher(None, base_line, donor_line)
        line_ops = line_sm.get_opcodes()

        # Find the opcode that covers this marker's position
        for tag, li1, li2, lj1, lj2 in line_ops:
            if tag == 'replace' and li1 <= pos_in_line < li2:
                # Check that the replaced region actually contains our marker
                if pos_in_line >= li1 and pos_in_line + len(marker) <= li2:
                    candidate = donor_line[lj1:lj2].strip()
                    if candidate and _is_real_recovery(candidate, redaction_markers):
                        candidates[pos] = (candidate, donor_offsets[donor_line_idx] + lj1)
                break

    return candidates
```

- [ ] **Step 5: Implement `_confirm_alignment_candidate` with AND logic**

```python
def _confirm_alignment_candidate(
    candidate: str,
    donor_text: str,
    base_text: str,
    pos: int,
    marker_len: int,
    redaction_markers: list[str],
    donor_line_offset: int = -1,
) -> bool:
    """Confirm an alignment candidate using anchor context (both sides must match).

    donor_line_offset: char offset of the donor line that produced this candidate.
    When provided, checks context around that specific position instead of
    searching globally (avoids matching wrong occurrence of repeated text).
    """
    context_len = 30
    left_ctx = base_text[max(0, pos - context_len):pos].strip()
    right_ctx = base_text[pos + marker_len:pos + marker_len + context_len].strip()

    if redaction_markers:
        for m in redaction_markers:
            idx = left_ctx.rfind(m)
            if idx >= 0:
                left_ctx = left_ctx[idx + len(m):].strip()
            idx = right_ctx.find(m)
            if idx >= 0:
                right_ctx = right_ctx[:idx].strip()

    # Use donor_line_offset if provided, else fall back to global search
    if donor_line_offset >= 0:
        cand_pos = donor_line_offset
    else:
        cand_pos = donor_text.find(candidate)
        if cand_pos == -1:
            return False

    search_window = 200
    donor_region = donor_text[max(0, cand_pos - search_window):cand_pos + len(candidate) + search_window]
    norm_region = _normalize_for_anchor(donor_region)

    # Both non-empty anchors must match (AND, not OR) for zero false positives
    left_ok = (not left_ctx) or left_ctx in donor_region or _normalize_for_anchor(left_ctx) in norm_region
    right_ok = (not right_ctx) or right_ctx in donor_region or _normalize_for_anchor(right_ctx) in norm_region

    return left_ok and right_ok
```

- [ ] **Step 6: Integrate alignment as fallback in merge_group**

After the anchor-based `for pos, marker in reversed(positions):` loop, add the alignment fallback. This goes BEFORE the confidence flagging block, AFTER the anchor loop:

```python
    # --- Alignment fallback for remaining redactions ---
    remaining_positions = find_redaction_positions(merged, redaction_markers)
    if remaining_positions and donors:
        for donor_id, donor_text in donors:
            if not remaining_positions:
                break
            alignment_candidates = _alignment_recover(
                merged, donor_text, remaining_positions, redaction_markers
            )
            newly_recovered = []
            # Process in reverse to maintain positions
            for pos, marker in reversed(remaining_positions):
                if pos in alignment_candidates:
                    candidate, donor_offset = alignment_candidates[pos]
                    if _confirm_alignment_candidate(
                        candidate, donor_text, merged, pos, len(marker),
                        redaction_markers, donor_line_offset=donor_offset
                    ):
                        merged = merged[:pos] + candidate + merged[pos + len(marker):]
                        recovered_count += 1
                        recovered_segments.append({
                            "text": candidate,
                            "source_doc_id": donor_id,
                            "stage": "alignment",
                            "confidence": "high",
                            "anchor_alpha_len": 0,
                        })
                        if donor_id not in source_doc_ids:
                            source_doc_ids.append(donor_id)
                        newly_recovered.append(pos)
            # Re-scan for remaining after this donor (positions shifted)
            remaining_positions = find_redaction_positions(merged, redaction_markers)
```

Note: after each donor's alignment replacements, we re-scan `merged` for remaining positions (fixing the position-shift bug from review issue #9).

- [ ] **Step 7: Run all merger tests**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: line-level alignment merge with anchor confirmation fallback"
```

---

### Task 4: Multi-Pass Merge with Chain Walking

Wrap the anchor + alignment logic in a multi-pass loop (up to 3 passes). Each pass uses the updated `merged` text (containing prior recoveries) as anchor context. Early exit when no progress. This enables chain walking: recovered text from pass N provides anchors for adjacent redactions in pass N+1.

**Files:**
- Modify: `stages/merger.py` (wrap merge loop)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for multi-pass chain recovery**

Design a test where pass-1 anchors are truncated by adjacent redaction markers, but after pass-1 recovers the "easy" ones, pass-2 has full anchor context. The key: redaction markers in the anchor window cause truncation (lines 66-76 of merger.py).

```python
def test_multi_pass_recovers_when_adjacent_recovery_provides_anchors(conn):
    """Pass 1 recovers outer redactions; pass 2 uses them as anchor context for inner ones."""
    # Layout: good_anchor [A] , [B] , [C] good_anchor
    # The separators between redactions are just ", " (2 chars, 0 alphanumeric).
    # Pass 1: A's left anchor is good ("confirmed that"), C's right anchor is good
    #         ("was already"). But B's anchors are truncated at adjacent markers,
    #         leaving only ", " on each side -> alpha_content < 8, skipped.
    # Pass 2: After A and C are recovered, B's anchors include "Bill Clinton, "
    #         (left) and ", Ghislaine Maxwell" (right) -> alpha > 8, matches.
    base_text = (
        "The lead investigator confirmed that [REDACTED], [REDACTED], [REDACTED] "
        "was already waiting according to the flight manifest records."
    )
    donor_text = (
        "The lead investigator confirmed that Bill Clinton, Prince Andrew, Ghislaine Maxwell "
        "was already waiting according to the flight manifest records."
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] == 3
    assert "Bill Clinton" in result["merged_text"]
    assert "Prince Andrew" in result["merged_text"]
    assert "Ghislaine Maxwell" in result["merged_text"]
```

- [ ] **Step 2: Run test to verify it fails (middle redaction skipped by quality floor)**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py::test_multi_pass_recovers_when_adjacent_recovery_provides_anchors -v`
Expected: FAIL — only 2 of 3 recovered (middle one's anchors truncated to ", " by adjacent markers, below quality floor)

- [ ] **Step 3: Implement multi-pass wrapper**

In `merge_group`, wrap the anchor loop + alignment fallback in a multi-pass structure. Replace the single-pass code with:

```python
    positions = find_redaction_positions(base_text, redaction_markers)
    if total_redacted is None:
        total_redacted = len(positions)
    recovered_count = 0
    source_doc_ids = []
    recovered_segments = []
    merged = base_text

    # Multi-pass: each pass uses prior recoveries as improved anchor context
    max_passes = 3
    for pass_num in range(max_passes):
        current_positions = find_redaction_positions(merged, redaction_markers)
        if not current_positions:
            break

        pre_pass_count = recovered_count

        # --- Anchor matching with adaptive widths ---
        for pos, marker in reversed(current_positions):
            left_anchor, right_anchor = extract_anchors(
                merged, pos, len(marker), anchor_length, redaction_markers
            )
            alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
            if len(alpha_content) < 8:
                logger.debug("Pass %d: skipping pos %d: anchor quality too low (%d)",
                             pass_num + 1, pos, len(alpha_content))
                continue

            anchor_widths = [anchor_length, 100, 150, 200]
            recovered_this = False
            for width in anchor_widths:
                if recovered_this:
                    break
                if width == anchor_length:
                    left_w, right_w = left_anchor, right_anchor
                else:
                    left_w, right_w = extract_anchors(
                        merged, pos, len(marker), width, redaction_markers
                    )

                for donor_id, donor_text in donors:
                    recovered = find_text_between_anchors(donor_text, left_w, right_w)
                    if recovered and _is_real_recovery(recovered, redaction_markers):
                        merged = merged[:pos] + recovered + merged[pos + len(marker):]
                        recovered_count += 1
                        alpha_w = re.sub(r'[^a-zA-Z0-9]', '', left_w + right_w)
                        recovered_segments.append({
                            "text": recovered,
                            "source_doc_id": donor_id,
                            "stage": "merge",
                            "confidence": "high",
                            "anchor_alpha_len": len(alpha_w),
                        })
                        if donor_id not in source_doc_ids:
                            source_doc_ids.append(donor_id)
                        recovered_this = True
                        break

        # --- Alignment fallback for remaining redactions ---
        remaining_positions = find_redaction_positions(merged, redaction_markers)
        if remaining_positions and donors:
            for donor_id, donor_text in donors:
                if not remaining_positions:
                    break
                alignment_candidates = _alignment_recover(
                    merged, donor_text, remaining_positions, redaction_markers
                )
                newly_recovered = []
                for pos, marker in reversed(remaining_positions):
                    if pos in alignment_candidates:
                        candidate, donor_offset = alignment_candidates[pos]
                        if _confirm_alignment_candidate(
                            candidate, donor_text, merged, pos, len(marker),
                            redaction_markers, donor_line_offset=donor_offset
                        ):
                            merged = merged[:pos] + candidate + merged[pos + len(marker):]
                            recovered_count += 1
                            recovered_segments.append({
                                "text": candidate,
                                "source_doc_id": donor_id,
                                "stage": "alignment",
                                "confidence": "high",
                                "anchor_alpha_len": 0,
                            })
                            if donor_id not in source_doc_ids:
                                source_doc_ids.append(donor_id)
                            newly_recovered.append(pos)
                # Re-scan remaining after this donor
                remaining_positions = find_redaction_positions(merged, redaction_markers)

        # Early exit if no progress this pass
        if recovered_count == pre_pass_count:
            break
```

This replaces everything between `positions = find_redaction_positions(...)` and the confidence-flagging block. The confidence flagging and return statement remain unchanged after the loop.

- [ ] **Step 4: Run all merger tests**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: multi-pass merge with chain walking (3 passes, early exit)"
```

---

### Task 5: Update PIPELINE.md

**Files:**
- Modify: `PIPELINE.md:91-121`

- [ ] **Step 1: Update Phase 4 documentation**

Replace the Phase 4 section with:

```markdown
## Phase 4 — Merging (Stage 3)

    FOR EACH match group with 2 or more members:

      1. Pick the "base" document:
         - IF any member has 0 redaction markers AND similar text length
           (within 3x): use the least-redacted member as base (reverse merge).
           total_redacted is taken from the most-redacted member for reporting.
         - OTHERWISE: use the member with the most redaction markers.

      2. Multi-pass anchor matching (up to 3 passes, early exit on no progress):
         FOR EACH redaction marker remaining in the merged text:
           a. Extract anchors (left + right context around the marker).
           b. Try progressively wider anchors: 50, 100, 150, 200 chars.
           c. FOR EACH donor member, search for the anchor pair.
              IF found uniquely: extract text between anchors as recovery.
           d. Each pass uses the updated merged text (including prior recoveries)
              as context, enabling chain walking through dense redaction blocks.

      3. Alignment fallback (after each anchor pass):
         FOR EACH remaining unrecovered redaction:
           a. Use difflib.SequenceMatcher on lines to align base and donor.
           b. Read the aligned region from the donor as a candidate.
           c. Confirm via anchor context (both left AND right must match).
           d. Only accept if surrounding context validates the candidate.

      4. Build merged_text and store in merge_results.
```

- [ ] **Step 2: Commit**

```bash
git add PIPELINE.md
git commit -m "docs: update Phase 4 with enhanced merge strategies"
```

---

### Task 6: Full Integration Test and Regression Check

**Files:**
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write integration test that exercises multiple techniques**

```python
def test_enhanced_merge_full_integration(conn):
    """Exercise multiple techniques: no single member has complete text."""
    # Member A: has names but locations redacted
    member_a = (
        "MEMORANDUM\nFrom: DOJ Office\nRe: Case Update\n\n"
        "1. Subject John Smith was seen at [REDACTED] on March 10.\n"
        "2. Witness Jane Doe confirmed John Smith was present at [REDACTED].\n"
        "3. Captain James Lee transported John Smith via private aircraft.\n"
        "4. Destination: [REDACTED]\n"
    )
    # Member B: has locations but names redacted
    member_b = (
        "MEMORANDUM\nFrom: DOJ Office\nRe: Case Update\n\n"
        "1. Subject [REDACTED] was seen at Mar-a-Lago on March 10.\n"
        "2. Witness [REDACTED] confirmed [REDACTED] was present at the Palm Beach estate.\n"
        "3. [REDACTED] transported [REDACTED] via private aircraft.\n"
        "4. Destination: Palm Beach International\n"
    )
    # Member C: structural variant (different page breaks) with partial info
    member_c = (
        "MEMORANDUM\nFrom: DOJ Office\nDate: March 15, 2005\nRe: Case Update\n\n"
        "1. Subject John Smith was seen at Mar-a-Lago on March 10.\n"
        "--- PAGE BREAK ---\n"
        "2. Witness Jane Doe confirmed John Smith was present at the Palm Beach estate.\n"
        "3. Captain James Lee transported John Smith via private aircraft.\n"
        "4. Destination: Palm Beach International\n"
    )
    seed_doc(conn, 1, member_a)
    seed_doc(conn, 2, member_b)
    seed_doc(conn, 3, member_c)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.85)
    add_group_member(conn, g, 3, 0.80)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Member C has 0 redactions — reverse merge uses it as base
    # All names and locations should be present
    assert "[REDACTED]" not in result["merged_text"]
    assert "John Smith" in result["merged_text"]
    assert "Mar-a-Lago" in result["merged_text"]
    assert "Jane Doe" in result["merged_text"]
    assert "Captain James Lee" in result["merged_text"]
    assert "Palm Beach International" in result["merged_text"]
    # total_redacted reflects the most-redacted member's count
    assert result["total_redacted"] >= 3
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/python -m pytest tests/stages/test_merger.py::test_enhanced_merge_full_integration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/stages/test_merger.py
git commit -m "test: add full integration test for enhanced merge strategies"
```
