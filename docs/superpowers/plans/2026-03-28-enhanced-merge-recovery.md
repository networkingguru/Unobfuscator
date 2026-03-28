# Enhanced Merge Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dramatically increase redaction recovery rates when near-duplicate unredacted copies exist, while maintaining zero false positives.

**Architecture:** Five complementary techniques layered into `stages/merger.py`:
1. **Reverse merge** — when a donor has 0 redactions + high similarity, use it as the base instead
2. **Sequence-alignment merge** — for high-similarity pairs (≥0.8), use `difflib.SequenceMatcher` to align texts and propose recoveries, confirmed by anchor validation
3. **Wider adaptive anchors** — when 50-char anchors fail uniqueness, automatically retry at 100, 150, 200 chars
4. **Two-pass merge** — after pass 1 recoveries, re-run anchor extraction using the improved text as context
5. **Anchor chain walking** — use recovered text from adjacent redactions to extend anchor context

**Tech Stack:** Python stdlib (`difflib.SequenceMatcher`), existing `stages/merger.py`, `core/db.py`

---

### Task 1: Reverse Merge — Select Least-Redacted Base

When a group member has 0 (or near-zero) redactions and the group has high average similarity, use that member as the merge base and reverse the merge direction: recover unique content from the *redacted* doc into the *unredacted* base.

**Files:**
- Modify: `stages/merger.py:256-285` (merge_group function, base selection)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for reverse merge selection**

```python
def test_merge_group_uses_unredacted_doc_as_base_when_available(conn):
    """When a donor has 0 redactions and high similarity, it becomes the base."""
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
    assert result["recovered_count"] == 3
    assert "[REDACTED]" not in result["merged_text"]
    assert "Bill Clinton" in result["merged_text"]
    assert "Palm Beach" in result["merged_text"]
    assert "Little St. James" in result["merged_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_merge_group_uses_unredacted_doc_as_base_when_available -v`
Expected: PASS (this may already pass with current logic since anchors exist — if so, proceed to the structural test below)

- [ ] **Step 3: Write failing test for structural reverse merge**

This tests the case where the current algorithm fails: the redacted doc has different structure than the unredacted one, so anchors don't match — but reverse merge (using unredacted as base) succeeds.

```python
def test_reverse_merge_recovers_when_normal_merge_fails(conn):
    """When docs have structural differences, using unredacted as base recovers more."""
    # Redacted doc has fewer entries (different page/rendering)
    redacted_text = (
        "(U) Interview of [REDACTED]\n"
        "(U) Interview of [REDACTED]\n"
        "(U) Proffer of [REDACTED]\n"
        "EFTA02730496\n--- PAGE BREAK ---\n"
    )
    # Unredacted doc has more entries and different page structure
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
    # With reverse merge, the unredacted doc becomes the base (0 redactions).
    # The merged text should be the unredacted version since it has no gaps.
    assert "[REDACTED]" not in result["merged_text"]
    assert "Jason Mojica" in result["merged_text"]
```

- [ ] **Step 4: Run test to verify it fails**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_reverse_merge_recovers_when_normal_merge_fails -v`
Expected: FAIL — current logic picks the redacted doc as base

- [ ] **Step 5: Implement reverse merge in merge_group**

In `stages/merger.py`, modify the `merge_group` function. After computing redaction counts for all members, check if any member has 0 redactions. If so, use the least-redacted member as the base and all others as donors.

Replace the base selection block (lines 271-285) with:

```python
    members = sorted(rows, key=redaction_count, reverse=True)

    if not members:
        return {"merged_text": "", "recovered_count": 0, "total_redacted": 0,
                "source_doc_ids": [], "recovered_segments": []}

    # Reverse merge: if any member has 0 redactions, use the least-redacted
    # as the base. The unredacted doc is the best foundation — we only need
    # to recover unique content from the redacted versions into it.
    least_redacted = min(members, key=redaction_count)
    most_redacted = members[0]

    if redaction_count(least_redacted) == 0 and redaction_count(most_redacted) > 0:
        base_id = least_redacted["id"]
        base_text = least_redacted["extracted_text"] or ""
        donors = [(row["id"], row["extracted_text"] or "") for row in members
                  if row["id"] != base_id]
    else:
        # Standard: most-redacted as base, others as donors
        base_id = most_redacted["id"]
        base_text = most_redacted["extracted_text"] or ""
        donors = [(row["id"], row["extracted_text"] or "") for row in members[1:]]
```

- [ ] **Step 6: Run tests to verify pass**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 7: Write test for reverse merge with no false positives**

```python
def test_reverse_merge_does_not_introduce_false_content(conn):
    """Reverse merge should not inject redaction markers or garbage into merged text."""
    redacted_text = (
        "Document about [REDACTED] at the meeting. "
        "Also [REDACTED] was present."
    )
    unredacted_text = (
        "Document about John Doe at the meeting. "
        "Also Jane Smith was present."
    )
    seed_doc(conn, 1, redacted_text)
    seed_doc(conn, 2, unredacted_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.95)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # No redaction markers in output
    for marker in REDACTION_MARKERS:
        assert marker not in result["merged_text"]
    # No garbage text
    assert "None" not in result["merged_text"]
    assert "null" not in result["merged_text"]
```

- [ ] **Step 8: Run test**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_reverse_merge_does_not_introduce_false_content -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: reverse merge — use unredacted doc as base when available"
```

---

### Task 2: Wider Adaptive Anchors

When the default 50-char anchor fails (returns None from `find_text_between_anchors`), automatically retry with progressively wider anchors: 100, 150, 200 chars. Wider anchors are more unique, reducing ambiguity rejections without any false-positive risk.

**Files:**
- Modify: `stages/merger.py:294-318` (anchor matching loop in merge_group)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test**

```python
def test_wider_anchors_recover_when_short_anchors_are_ambiguous(conn):
    """When 50-char anchors are ambiguous, wider context should resolve it."""
    # Two identical short-anchor regions but different at wider context
    base_text = (
        "On January 15th in the downtown office, the lawyer met with [REDACTED] about the case. "
        "On February 20th in the uptown office, the lawyer met with [REDACTED] about the deal."
    )
    donor_text = (
        "On January 15th in the downtown office, the lawyer met with Alan Dershowitz about the case. "
        "On February 20th in the uptown office, the lawyer met with Leslie Wexner about the deal."
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    assert result["recovered_count"] == 2
    assert "Alan Dershowitz" in result["merged_text"]
    assert "Leslie Wexner" in result["merged_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_wider_anchors_recover_when_short_anchors_are_ambiguous -v`
Expected: FAIL (short anchors "the lawyer met with" / "about the" are ambiguous)

- [ ] **Step 3: Implement adaptive anchor widening**

In `stages/merger.py`, modify the inner loop in `merge_group` where anchor matching is attempted. After the initial attempt with `anchor_length` fails, retry with wider anchors.

Replace the donor-search loop (the `for donor_id, donor_text in donors:` block, lines 304-318) with:

```python
        # Try progressively wider anchors if the default length fails
        anchor_widths = [anchor_length, 100, 150, 200]
        recovered_this = False
        for width in anchor_widths:
            if recovered_this:
                break
            left_anchor_w, right_anchor_w = extract_anchors(
                base_text, pos, len(marker), width, redaction_markers
            )
            # Re-check quality floor for wider anchors (already passed for first width)
            if width > anchor_length:
                alpha_w = re.sub(r'[^a-zA-Z0-9]', '', left_anchor_w + right_anchor_w)
                if len(alpha_w) < 8:
                    continue

            for donor_id, donor_text in donors:
                recovered = find_text_between_anchors(donor_text, left_anchor_w, right_anchor_w)
                if recovered and _is_real_recovery(recovered, redaction_markers):
                    merged = merged[:pos] + recovered + merged[pos + len(marker):]
                    recovered_count += 1
                    recovered_segments.append({
                        "text": recovered,
                        "source_doc_id": donor_id,
                        "stage": "merge",
                        "confidence": "high",
                        "anchor_alpha_len": len(re.sub(r'[^a-zA-Z0-9]', '',
                                                       left_anchor_w + right_anchor_w)),
                    })
                    if donor_id not in source_doc_ids:
                        source_doc_ids.append(donor_id)
                    recovered_this = True
                    break  # Found recovery — move to next redaction
```

Also update the initial anchor extraction + quality check (lines 296-302) to only compute once for the first width, and move the quality floor check inside the loop above (already included).

The full revised per-redaction block becomes:

```python
    for pos, marker in reversed(positions):
        # Initial quality check with default anchor width
        left_anchor, right_anchor = extract_anchors(
            base_text, pos, len(marker), anchor_length, redaction_markers
        )
        alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
        if len(alpha_content) < 8:
            logger.debug("Skipping redaction at pos %d: anchor quality too low (%d alphanumeric chars)",
                         pos, len(alpha_content))
            continue

        # Try progressively wider anchors if the default length fails
        anchor_widths = [anchor_length, 100, 150, 200]
        recovered_this = False
        for width in anchor_widths:
            if recovered_this:
                break
            if width == anchor_length:
                left_w, right_w = left_anchor, right_anchor
            else:
                left_w, right_w = extract_anchors(
                    base_text, pos, len(marker), width, redaction_markers
                )

            for donor_id, donor_text in donors:
                recovered = find_text_between_anchors(donor_text, left_w, right_w)
                if recovered and _is_real_recovery(recovered, redaction_markers):
                    merged = merged[:pos] + recovered + merged[pos + len(marker):]
                    recovered_count += 1
                    recovered_segments.append({
                        "text": recovered,
                        "source_doc_id": donor_id,
                        "stage": "merge",
                        "confidence": "high",
                        "anchor_alpha_len": len(alpha_content),
                    })
                    if donor_id not in source_doc_ids:
                        source_doc_ids.append(donor_id)
                    recovered_this = True
                    break
```

- [ ] **Step 4: Run tests to verify pass**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: adaptive anchor widening — retry with 100/150/200 char anchors on failure"
```

---

### Task 3: Sequence Alignment Merge with Anchor Confirmation

For high-similarity document pairs (both share substantial common text), use `difflib.SequenceMatcher` to align the base and donor texts structurally. For each redaction marker in the base, read the aligned region from the donor as a candidate recovery. Then validate the candidate using anchor matching as a confirmation step.

**Files:**
- Modify: `stages/merger.py` (add new function + integrate into merge_group)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for alignment-based recovery**

```python
def test_alignment_merge_recovers_with_structural_differences(conn):
    """Alignment-based merge handles different page breaks and shifted content."""
    # Documents share content but have different structure
    redacted_text = (
        "MEMORANDUM\n"
        "From: Office of General Counsel\n"
        "Re: Investigation Update\n\n"
        "1. The subject [REDACTED] was identified at the location.\n"
        "2. Witness [REDACTED] provided testimony.\n"
        "EFTA0001\n--- PAGE BREAK ---\n"
        "3. Evidence collected from [REDACTED] confirmed the timeline.\n"
    )
    # Same content but different page breaks and added lines
    unredacted_text = (
        "MEMORANDUM\n"
        "From: Office of General Counsel\n"
        "Date: March 15, 2005\n"
        "Re: Investigation Update\n\n"
        "1. The subject John Smith was identified at the location.\n"
        "2. Witness Sarah Johnson provided testimony.\n"
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

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_alignment_merge_recovers_with_structural_differences -v`
Expected: FAIL — anchor matching fails due to different page breaks

- [ ] **Step 3: Implement alignment-based recovery function**

Add a new function `_alignment_recover` to `stages/merger.py`:

```python
def _alignment_recover(
    base_text: str,
    donor_text: str,
    positions: list[tuple[int, str]],
    redaction_markers: list[str],
) -> dict[int, str]:
    """Use SequenceMatcher alignment to propose recoveries for redaction positions.

    Returns a dict mapping redaction position -> candidate recovered text.
    Only returns candidates where the alignment context confirms the match.
    """
    from difflib import SequenceMatcher

    # Build a version of base_text with markers replaced by a unique placeholder
    # so alignment doesn't try to match "[REDACTED]" across documents
    PLACEHOLDER = "\x00"
    clean_base = base_text
    # Process positions in reverse to maintain indices
    offset_map = []  # [(original_pos, placeholder_pos, marker_len)]
    temp = base_text
    for pos, marker in reversed(positions):
        temp = temp[:pos] + PLACEHOLDER + temp[pos + len(marker):]

    # Align cleaned base with donor
    sm = SequenceMatcher(None, temp, donor_text, autojunk=False)
    opcodes = sm.get_opcodes()

    candidates = {}
    for pos, marker in positions:
        # Find where this position's placeholder ended up in temp
        # Calculate offset: each prior replacement changed length
        adjusted_pos = pos
        for orig_pos, _, m_len in offset_map:
            if orig_pos < pos:
                adjusted_pos -= (m_len - len(PLACEHOLDER))

        # Find the opcode that covers this position in temp
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'replace' and i1 <= adjusted_pos < i2:
                # The aligned region in donor_text is j1:j2
                candidate = donor_text[j1:j2].strip()
                if candidate and _is_real_recovery(candidate, redaction_markers):
                    candidates[pos] = candidate
                break
            elif tag == 'equal' and i1 <= adjusted_pos < i2:
                # Placeholder landed in an 'equal' block — alignment thinks
                # the marker text exists in the donor too (shouldn't happen
                # with our placeholder approach, but skip if it does)
                break

    return candidates
```

- [ ] **Step 4: Write anchor confirmation wrapper**

Add a confirmation function that validates alignment candidates against anchors:

```python
def _confirm_alignment_candidate(
    candidate: str,
    donor_text: str,
    base_text: str,
    pos: int,
    marker_len: int,
    redaction_markers: list[str],
) -> bool:
    """Confirm an alignment-proposed candidate using anchor context.

    Checks that the candidate text actually appears in the donor surrounded
    by context that matches the base document's context around the redaction.
    """
    # Get context around the redaction in base
    context_len = 30
    left_ctx = base_text[max(0, pos - context_len):pos].strip()
    right_ctx = base_text[pos + marker_len:pos + marker_len + context_len].strip()

    # Strip redaction markers from context
    if redaction_markers:
        for m in redaction_markers:
            idx = left_ctx.rfind(m)
            if idx >= 0:
                left_ctx = left_ctx[idx + len(m):].strip()
            idx = right_ctx.find(m)
            if idx >= 0:
                right_ctx = right_ctx[:idx].strip()

    # Check: does the candidate appear in donor_text near the expected context?
    cand_pos = donor_text.find(candidate)
    if cand_pos == -1:
        return False

    # Verify at least one anchor appears near the candidate in the donor
    search_window = 200
    donor_region = donor_text[max(0, cand_pos - search_window):cand_pos + len(candidate) + search_window]

    left_ok = not left_ctx or left_ctx in donor_region or _normalize_for_anchor(left_ctx) in _normalize_for_anchor(donor_region)
    right_ok = not right_ctx or right_ctx in donor_region or _normalize_for_anchor(right_ctx) in _normalize_for_anchor(donor_region)

    return left_ok or right_ok
```

- [ ] **Step 5: Integrate alignment merge into merge_group as a fallback**

After the anchor-based loop completes, check for any remaining unrecovered redactions. For those, attempt alignment-based recovery with anchor confirmation. Add this block after the `for pos, marker in reversed(positions):` loop and before the confidence flagging:

```python
    # --- Alignment fallback for remaining redactions ---
    # Re-scan merged text for any remaining redaction markers
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
                    candidate = alignment_candidates[pos]
                    if _confirm_alignment_candidate(
                        candidate, donor_text, merged, pos, len(marker),
                        redaction_markers
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
            # Update remaining positions (remove recovered ones)
            remaining_positions = [(p, m) for p, m in remaining_positions
                                   if p not in newly_recovered]
```

- [ ] **Step 6: Run tests**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 7: Write test that alignment does NOT false-recover**

```python
def test_alignment_merge_does_not_false_recover_unrelated_content(conn):
    """Alignment should not recover content from structurally different regions."""
    redacted_text = (
        "Section A: The investigation found [REDACTED] at the scene.\n"
        "Section B: The report concluded with no further action.\n"
    )
    # Donor has different content in Section A's region
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
    # "surplus funds" should NOT be recovered — context doesn't match
    assert "surplus funds" not in result["merged_text"]
    assert "[REDACTED]" in result["merged_text"]
```

- [ ] **Step 8: Run test**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_alignment_merge_does_not_false_recover_unrelated_content -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: sequence alignment merge with anchor confirmation fallback"
```

---

### Task 4: Two-Pass Merge

After the first pass (anchor + alignment), re-scan the merged text for remaining redactions. The first pass may have recovered text adjacent to still-redacted markers, creating better anchor context. Run the anchor matching loop again on the improved text.

**Files:**
- Modify: `stages/merger.py:256-333` (merge_group function)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for two-pass recovery**

```python
def test_two_pass_merge_recovers_adjacent_redactions(conn):
    """Second pass uses first-pass recoveries as anchor context for neighbors."""
    # Three consecutive redactions — middle one has good anchors,
    # outer ones depend on the middle one's content for their anchors
    base_text = (
        "The meeting was attended by [REDACTED], [REDACTED], and [REDACTED] at the resort."
    )
    donor_text = (
        "The meeting was attended by Bill Clinton, Prince Andrew, and Alan Dershowitz at the resort."
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # All three names should be recovered (possibly across two passes)
    assert result["recovered_count"] == 3
    assert "Bill Clinton" in result["merged_text"]
    assert "Prince Andrew" in result["merged_text"]
    assert "Alan Dershowitz" in result["merged_text"]
```

- [ ] **Step 2: Run test to verify it fails (or check how many recover in pass 1)**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_two_pass_merge_recovers_adjacent_redactions -v`
Expected: May partially recover — note how many in the error message

- [ ] **Step 3: Implement two-pass merge**

In `merge_group`, after the first anchor-based loop and the alignment fallback, add a second pass. Wrap the existing recovery logic in a `for pass_num in range(2):` loop. On pass 2, use the updated `merged` text (which now contains pass-1 recoveries) as the base for anchor extraction.

Add this structure to `merge_group`, replacing the single-pass anchor loop:

```python
    # Multi-pass merge: pass 2 uses pass-1 recoveries as improved anchor context
    for pass_num in range(2):
        current_positions = find_redaction_positions(merged, redaction_markers)
        if not current_positions:
            break  # No more redactions to recover

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
                        recovered_segments.append({
                            "text": recovered,
                            "source_doc_id": donor_id,
                            "stage": "merge",
                            "confidence": "high",
                            "anchor_alpha_len": len(alpha_content),
                        })
                        if donor_id not in source_doc_ids:
                            source_doc_ids.append(donor_id)
                        recovered_this = True
                        break

        # Alignment fallback for remaining redactions (same as Task 3)
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
                        candidate = alignment_candidates[pos]
                        if _confirm_alignment_candidate(
                            candidate, donor_text, merged, pos, len(marker),
                            redaction_markers
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
                remaining_positions = [(p, m) for p, m in remaining_positions
                                       if p not in newly_recovered]
```

Note: The anchor extraction in pass 2 uses `merged` (not `base_text`), which now contains pass-1 recoveries as context. This is the key difference — recovered text from pass 1 provides better anchors for adjacent redactions.

- [ ] **Step 4: Run all tests**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: two-pass merge — use pass-1 recoveries as anchor context for pass 2"
```

---

### Task 5: Anchor Chain Walking

Enhance the two-pass approach by explicitly using recovered text from adjacent redactions to extend anchor context. When a redaction is between two other redactions that were already recovered, use the recovered text as part of the anchor.

This is effectively achieved by Task 4's approach of using `merged` (which contains recoveries) as the anchor extraction source. However, we can further improve by extending the anchor window to explicitly reach across the position of a recently-recovered adjacent redaction.

**Files:**
- Modify: `stages/merger.py` (extract_anchors enhancement)
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write failing test for chain walking**

```python
def test_anchor_chain_walking_through_dense_redactions(conn):
    """Dense redaction blocks where only first/last have good anchors should chain-recover."""
    base_text = (
        "From: sender@example.com\n"
        "Subject: Guest List\n\n"
        "Confirmed guests:\n"
        "1. [REDACTED]\n"
        "2. [REDACTED]\n"
        "3. [REDACTED]\n"
        "4. [REDACTED]\n"
        "5. [REDACTED]\n"
        "\nPlease confirm attendance by Friday.\n"
    )
    donor_text = (
        "From: sender@example.com\n"
        "Subject: Guest List\n\n"
        "Confirmed guests:\n"
        "1. Bill Clinton\n"
        "2. Prince Andrew\n"
        "3. Alan Dershowitz\n"
        "4. Ghislaine Maxwell\n"
        "5. Jean-Luc Brunel\n"
        "\nPlease confirm attendance by Friday.\n"
    )
    seed_doc(conn, 1, base_text)
    seed_doc(conn, 2, donor_text)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # With chain walking + two passes, all 5 should eventually recover
    assert result["recovered_count"] >= 4, f"Only recovered {result['recovered_count']} of 5"
    assert "Bill Clinton" in result["merged_text"]
    assert "Jean-Luc Brunel" in result["merged_text"]
```

- [ ] **Step 2: Run test to see current recovery count**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_anchor_chain_walking_through_dense_redactions -v`
Expected: Note how many recover — the two-pass approach from Task 4 may already handle some

- [ ] **Step 3: Increase pass count if needed**

If the test fails because 2 passes aren't enough for 5 dense redactions, increase the pass limit. In the `for pass_num in range(2):` loop, change to `range(3)` to allow a third pass. Three passes is sufficient: pass 1 gets the easy ones (first/last with good outer anchors), pass 2 gets their neighbors (using recovered text as context), pass 3 gets the middle ones.

```python
    # Multi-pass merge: each pass uses prior recoveries as improved anchor context
    max_passes = 3
    for pass_num in range(max_passes):
        current_positions = find_redaction_positions(merged, redaction_markers)
        if not current_positions:
            break  # No more redactions to recover

        pre_pass_count = recovered_count
        # ... (rest of loop unchanged) ...

        # Early exit if no progress this pass
        if recovered_count == pre_pass_count:
            break
```

Add the `pre_pass_count` check and early exit to avoid wasted passes. Also bump `max_passes` to 3.

- [ ] **Step 4: Run all tests**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add stages/merger.py tests/stages/test_merger.py
git commit -m "feat: anchor chain walking via multi-pass merge with early exit"
```

---

### Task 6: Update PIPELINE.md

Update the Phase 4 documentation to reflect the new merge strategies.

**Files:**
- Modify: `PIPELINE.md:91-121`

- [ ] **Step 1: Update Phase 4 documentation**

Replace the Phase 4 section in `PIPELINE.md` with:

```markdown
## Phase 4 — Merging (Stage 3)

    FOR EACH match group with 2 or more members:

      1. Pick the "base" document:
         - IF any member has 0 redaction markers: use the least-redacted member
           as base (reverse merge — the unredacted doc is the best foundation).
         - OTHERWISE: use the member with the most redaction markers.

      2. Multi-pass anchor matching (up to 3 passes):
         FOR EACH redaction marker remaining in the merged text:
           a. Extract anchors (left + right context around the marker).
           b. Try progressively wider anchors: 50, 100, 150, 200 chars.
           c. FOR EACH donor member, search for the anchor pair.
              IF found uniquely: extract text between anchors as recovery.
           d. Each pass uses the updated merged text (including prior recoveries)
              as context, so recovered text improves anchors for adjacent redactions.
           e. Stop early if a pass makes no progress.

      3. Alignment fallback (after each anchor pass):
         FOR EACH remaining unrecovered redaction:
           a. Use difflib.SequenceMatcher to align base and donor texts.
           b. Read the aligned region from the donor as a candidate.
           c. Confirm the candidate via anchor context validation.
           d. Only accept if surrounding context matches.

      4. Build merged_text and store in merge_results.
```

- [ ] **Step 2: Commit**

```bash
git add PIPELINE.md
git commit -m "docs: update Phase 4 documentation with enhanced merge strategies"
```

---

### Task 7: Full Integration Test and Regression Check

Run the complete test suite and verify no regressions. Add an integration test that exercises all merge strategies together.

**Files:**
- Test: `tests/stages/test_merger.py`

- [ ] **Step 1: Write integration test combining all strategies**

```python
def test_enhanced_merge_full_integration(conn):
    """Exercise reverse merge + adaptive anchors + alignment + two-pass together."""
    # Complex scenario: group with 3 members of varying redaction levels
    heavily_redacted = (
        "MEMORANDUM\nFrom: DOJ Office\nRe: Case Update\n\n"
        "1. Subject [REDACTED] was seen at [REDACTED] on March 10.\n"
        "2. Witness [REDACTED] confirmed [REDACTED] was present.\n"
        "3. [REDACTED] transported [REDACTED] via private aircraft.\n"
        "4. Destination: [REDACTED]\n"
    )
    partially_redacted = (
        "MEMORANDUM\nFrom: DOJ Office\nRe: Case Update\n\n"
        "1. Subject John Smith was seen at [REDACTED] on March 10.\n"
        "2. Witness [REDACTED] confirmed John Smith was present.\n"
        "3. [REDACTED] transported John Smith via private aircraft.\n"
        "4. Destination: Palm Beach International\n"
    )
    unredacted = (
        "MEMORANDUM\nFrom: DOJ Office\nDate: March 15, 2005\nRe: Case Update\n\n"
        "1. Subject John Smith was seen at Mar-a-Lago on March 10.\n"
        "2. Witness Jane Doe confirmed John Smith was present.\n"
        "3. Captain James Lee transported John Smith via private aircraft.\n"
        "4. Destination: Palm Beach International\n"
    )
    seed_doc(conn, 1, heavily_redacted)
    seed_doc(conn, 2, partially_redacted)
    seed_doc(conn, 3, unredacted)
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    add_group_member(conn, g, 3, 0.85)
    conn.commit()

    result = merge_group(conn, g, REDACTION_MARKERS, anchor_length=50)
    # Reverse merge: unredacted (0 markers) becomes base
    # Result should have 0 remaining redactions since base has none
    assert "[REDACTED]" not in result["merged_text"]
    assert "John Smith" in result["merged_text"]
    assert "Mar-a-Lago" in result["merged_text"]
    assert "Jane Doe" in result["merged_text"]
    assert "Captain James Lee" in result["merged_text"]
    assert "Palm Beach International" in result["merged_text"]
```

- [ ] **Step 2: Run the integration test**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/stages/test_merger.py::test_enhanced_merge_full_integration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `/root/Unobfuscator/.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/stages/test_merger.py
git commit -m "test: add full integration test for enhanced merge strategies"
```
