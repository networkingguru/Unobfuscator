# Anchor Uniqueness Fix — Eliminate False Recoveries

**Date:** 2026-03-27
**Problem:** Consecutive redactions produce identical degenerate anchors, causing the merger to "recover" the same wrong text dozens of times.

## Root Cause

When redactions are consecutive (separated only by quote markers like `>>>>>>` or table pipes `|`), `extract_anchors()` truncates at neighboring redaction markers, leaving both anchors as identical short punctuation strings. `find_text_between_anchors()` then matches the first occurrence in the donor — the same wrong text every time.

**Impact:** ~177 false recoveries across 35 groups out of 2,649 total recoveries (~7%).

## Design

### Change 1: Anchor Quality Floor

In `merge_group()`, after calling `extract_anchors()`, check combined alphanumeric content:

```python
alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
if len(alpha_content) < 8:
    continue  # skip this redaction — anchors too weak
```

**Why 8:** Shorter than any meaningful phrase. Catches `>>>>>>` + `>>>>>>` (0 alphanumeric), `| East Coast |` (9 — passes), etc.

### Change 2: Uniqueness Check in `_find_between_exact`

After finding the first match, search for a second occurrence. If found, return `None`:

```python
# After finding left_pos and right_pos for the first match:
second_left = text.find(left_anchor, left_pos + 1)
if second_left != -1:
    second_right = text.find(right_anchor, second_left + len(left_anchor))
    if second_right != -1:
        return None  # ambiguous — anchor pair matches multiple positions
```

This protects all three matching tiers (exact, normalized, progressive shortening) since they all go through `_find_between_exact`.

**Special case:** When `left_anchor` is empty (redaction at start of text), only check right anchor uniqueness.

### Change 3: Data Remediation

One-time re-merge of all groups:

```sql
UPDATE match_groups SET merged = 0;
DELETE FROM merge_results;
```

Then run the merger. ~10 minutes for 44K groups at ~14ms/group.

## What This Does NOT Change

- `extract_anchors()` logic — the truncation at neighboring markers is correct behavior
- `_is_real_recovery()` — content validation is a separate concern
- No block-aware recovery — that's a future enhancement (recover consecutive blocks as a unit)

## Scope of Impact

- **False recoveries eliminated:** ~177 across 35 groups
- **True recoveries preserved:** The vast majority of the 2,649 recoveries use unique anchors and will be unaffected
- **Possible true recoveries lost:** Cases where a legitimate anchor pair happens to occur twice in the donor. This is the correct trade-off — if we can't tell which match is right, we shouldn't guess.

## GitHub Issue

File as Issue #4 on networkingguru/Unobfuscator alongside the 3 existing issues pending filing:
- Title: "False recoveries from degenerate anchors on consecutive redactions"
- Labels: bug
- Reference this spec and the fix commits
