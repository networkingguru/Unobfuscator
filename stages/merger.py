"""Stage 3: Merger — fill redaction gaps using anchor-phrase matching across match groups.

Logic reference: PIPELINE.md — Phase 4 (Merging)
"""

import bisect
import logging
import re
from typing import Optional
from core.db import upsert_merge_result

logger = logging.getLogger(__name__)


def _normalize_for_anchor(text: str) -> str:
    """Normalize text for anchor matching by stripping formatting differences.

    Different OCR/extraction passes may produce Markdown tables, headers, etc.
    while others produce plain text. This strips those differences so anchors
    can match across formatting variants.
    """
    # Strip Markdown table separators and pipes
    text = re.sub(r'\|[-—–]+\|', ' ', text)
    text = re.sub(r'[-—–]{3,}', ' ', text)
    text = text.replace('|', ' ')
    # Strip Markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strip bold/italic markers
    text = text.replace('**', '').replace('__', '').replace('*', '')
    # Strip list markers
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_redaction_positions(text: str, redaction_markers: list[str]) -> list[tuple[int, str]]:
    """Return list of (position, marker) for every redaction marker in text."""
    positions = []
    for marker in redaction_markers:
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos == -1:
                break
            positions.append((pos, marker))
            start = pos + len(marker)
    return sorted(positions, key=lambda x: x[0])


def extract_anchors(
    text: str, pos: int, marker_len: int, length: int = 50,
    redaction_markers: list[str] = None
) -> tuple[str, str]:
    """Return (left_anchor, right_anchor) of up to `length` chars each around a redaction.

    left_anchor: up to `length` chars immediately before pos.
    right_anchor: up to `length` chars immediately after pos + marker_len.

    If redaction_markers is provided, anchors are truncated at any embedded
    redaction marker (since those won't appear in donor text that has real content).
    """
    left_anchor = text[max(0, pos - length):pos].strip()
    right_anchor = text[pos + marker_len:pos + marker_len + length].strip()

    if redaction_markers:
        # Truncate left anchor: keep only text AFTER the last redaction marker
        for m in redaction_markers:
            idx = left_anchor.rfind(m)
            if idx >= 0:
                left_anchor = left_anchor[idx + len(m):].strip()

        # Truncate right anchor: keep only text BEFORE the first redaction marker
        for m in redaction_markers:
            idx = right_anchor.find(m)
            if idx >= 0:
                right_anchor = right_anchor[:idx].strip()

    return left_anchor, right_anchor


def find_text_between_anchors(
    text: str, left_anchor: str, right_anchor: str
) -> Optional[str]:
    """Search text for left_anchor, then find right_anchor after it.

    Returns the text between them if both anchors are found, else None.
    First tries exact matching, then normalized matching, then progressively
    shorter anchors for cross-format compatibility.
    """
    result = _find_between_exact(text, left_anchor, right_anchor)
    if result is not None:
        return result

    # Fallback: normalize both text and anchors to handle formatting differences
    norm_text = _normalize_for_anchor(text)
    norm_left = _normalize_for_anchor(left_anchor)
    norm_right = _normalize_for_anchor(right_anchor)
    result = _find_between_exact(norm_text, norm_left, norm_right)
    if result is not None:
        return result

    # Progressive shortening: drop words from outer edges of each anchor
    # until we find a match, but keep at least 3 words per anchor
    left_words = norm_left.split()
    right_words = norm_right.split()
    min_words = 3

    for drop in range(1, max(len(left_words), len(right_words)) - min_words + 1):
        shortened_left = " ".join(left_words[drop:]) if len(left_words) > min_words + drop - 1 else norm_left
        shortened_right = " ".join(right_words[:-drop]) if len(right_words) > min_words + drop - 1 else norm_right
        if shortened_left == norm_left and shortened_right == norm_right:
            continue
        result = _find_between_exact(norm_text, shortened_left, shortened_right)
        if result is not None and len(result) < 2000:  # sanity check on recovered length
            return result

    return None


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
        # Stricter than the non-empty case: with no left anchor to pin the search,
        # any repeated right anchor makes the match ambiguous
        if text.find(right_anchor, right_pos + 1) != -1:
            return None
        return text[:right_pos].strip()

    left_pos = text.find(left_anchor)
    if left_pos == -1:
        return None

    search_from = left_pos + len(left_anchor)

    # Symmetric to the empty-left case: no right anchor means return text from
    # after left_anchor to end of string, but only if the left anchor is unique.
    if not right_anchor:
        if text.find(left_anchor, left_pos + 1) != -1:
            return None  # ambiguous
        return text[search_from:].strip()

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


def _is_real_recovery(text: str, redaction_markers: list[str]) -> bool:
    """Return True only if the recovered text contains genuinely new content.

    Rejects:
    - Other redaction markers or case variants
    - Block redaction characters (█, X runs, ■ runs)
    - Pure formatting artifacts (table separators, markdown headers, HTML tags)
    - Generic placeholder text (Redacted, Unspecified, etc.)
    - Very short or whitespace-only text
    """
    stripped = text.strip()
    if len(stripped) < 2:
        return False

    # Check against known redaction markers (case-insensitive)
    lower = stripped.lower()
    for m in redaction_markers:
        if m.lower() in lower:
            return False

    # Escaped redaction markers like \[Redacted\]
    if "\\[" in stripped and "redact" in lower:
        return False

    # Block characters and X-runs — strip them and check if anything real remains
    block_count = sum(1 for c in stripped if c in '█■')
    if block_count > 0 and block_count / len(stripped) > 0.10:
        return False
    clean = re.sub(r'[█■]+', '', stripped).strip()
    # Also strip punctuation wrappers: <>, (), [], "", etc.
    clean_alpha = re.sub(r'[<>()\[\]"\'\s,;:+\-]+', '', clean).strip()
    if len(clean_alpha) < 4:
        return False

    # Any text that contains "redact" in any form
    if "redact" in lower:
        return False

    # Generic placeholders
    placeholders = {
        "unspecified", "recipient name", "sender name",
        "name redacted", "n/a", "unknown", "withheld",
    }
    if lower in placeholders:
        return False

    # Pure formatting artifacts
    formatting_only = re.sub(r'[-—–|#*_:\s\\]+', '', stripped)
    if not formatting_only:
        return False

    # Lone HTML tags
    if re.match(r'^</?[a-z]+/?>$', stripped, re.IGNORECASE):
        return False

    # Table column headers / generic labels that aren't real content
    if lower in {"item", "description", "field", "value", "date", "details",
                 "name", "notes", "type", "status", "number", "e-ticket number",
                 "category", "[blank]", "[blank]\n-", "blank", "(usanys)"}:
        return False

    # Redaction descriptions — not actual recovered content
    redaction_descriptions = {
        "blacked out", "blackened out", "blackened box",
        "blanked out", "blocked out", "covered", "obscured",
        "whited out", "crossed out",
    }
    # Strip brackets/angle brackets for matching
    bracket_stripped = re.sub(r'[\[\]<>]', '', lower).strip()
    if bracket_stripped in redaction_descriptions:
        return False

    # X-only runs (XXXXXXXX, etc.)
    if re.match(r'^[Xx]+$', stripped):
        return False

    # Very short text that's just a year or generic word
    if re.match(r'^\d{4}$', stripped):
        return False

    # AI image descriptions — OCR artifacts from redacted pages, not real content
    ai_description_prefixes = (
        "the image ", "this image ", "this page ",
        "the image is ", "this image is ", "this is an image",
        "this appears to be",
    )
    if any(lower.startswith(p) for p in ai_description_prefixes):
        return False

    return True


def _alignment_recover(
    base_text: str,
    donor_text: str,
    positions: list[tuple[int, str]],
    redaction_markers: list[str],
) -> dict[int, tuple[str, int]]:
    """Use line-level SequenceMatcher to propose recoveries for redaction positions.

    Returns dict mapping redaction position -> (candidate text, donor_line_char_offset).
    The donor_line_char_offset enables positional confirmation instead of global search.
    Works on lines with O(P log L) position mapping via bisect.
    """
    from difflib import SequenceMatcher

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

    # Map each redaction position to its line index (O(P log L) via bisect)
    pos_to_line = {}
    for pos, marker in positions:
        line_idx = bisect.bisect_right(base_offsets, pos) - 1
        if 0 <= line_idx < len(base_lines) and pos < base_offsets[line_idx] + len(base_lines[line_idx]):
            pos_to_line[pos] = line_idx

    # Align lines
    sm = SequenceMatcher(None, base_lines, donor_lines)
    opcodes = sm.get_opcodes()

    # Build line mapping: base_line_index -> donor_line_index for 'replace' blocks
    base_to_donor_line = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'replace':
            for bi in range(i1, i2):
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

        base_line = base_lines[line_idx]
        donor_line = donor_lines[donor_line_idx]
        line_offset = base_offsets[line_idx]
        pos_in_line = pos - line_offset

        # Char-level SequenceMatcher on the individual line pair
        line_sm = SequenceMatcher(None, base_line, donor_line)
        line_ops = line_sm.get_opcodes()

        # Only use candidate if the replace region contains exactly one marker
        for tag, li1, li2, lj1, lj2 in line_ops:
            if tag == 'replace' and li1 <= pos_in_line < li2:
                if pos_in_line >= li1 and pos_in_line + len(marker) <= li2:
                    markers_in_region = sum(
                        1 for p, m in positions
                        if pos_to_line.get(p) == line_idx
                        and li1 <= (p - line_offset) < li2
                    )
                    if markers_in_region == 1:
                        candidate = donor_line[lj1:lj2].strip()
                        if candidate and _is_real_recovery(candidate, redaction_markers):
                            candidates[pos] = (candidate, donor_offsets[donor_line_idx] + lj1)
                break

    return candidates


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

    # Quality floor: < 4 alphanumeric chars after truncation = can't confirm
    ctx_alpha = re.sub(r'[^a-zA-Z0-9]', '', left_ctx + right_ctx)
    if len(ctx_alpha) < 4:
        return False

    # Both non-empty anchors must match (AND, not OR)
    # Exact or normalized substring matching only — no fuzzy fallback (spec compliance)
    norm_left = _normalize_for_anchor(left_ctx) if left_ctx else ""
    norm_right = _normalize_for_anchor(right_ctx) if right_ctx else ""

    def _ctx_matches(ctx, norm_ctx, region, norm_reg):
        if not ctx:
            return True
        return ctx in region or norm_ctx in norm_reg

    left_ok = _ctx_matches(left_ctx, norm_left, donor_region, norm_region)
    right_ok = _ctx_matches(right_ctx, norm_right, donor_region, norm_region)

    return left_ok and right_ok


def merge_group(
    conn, group_id: int, redaction_markers: list[str], anchor_length: int = 50
) -> dict:
    """Merge all documents in a match group to produce the best reconstruction.

    Returns dict with merged_text, recovered_count, total_redacted, source_doc_ids.
    Logic reference: PIPELINE.md — Phase 4
    """
    rows = conn.execute("""
        SELECT d.id, d.extracted_text
        FROM match_group_members m
        JOIN documents d ON d.id = m.doc_id
        WHERE m.group_id = ?
    """, (group_id,)).fetchall()

    # Sort in Python: base = member with most redaction markers (the doc we want to fill)
    # donors = members with fewer redactions that can provide content
    def redaction_count(row):
        t = row["extracted_text"] or ""
        return sum(t.count(marker) for marker in redaction_markers)

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

    positions = find_redaction_positions(base_text, redaction_markers)
    if total_redacted is None:
        total_redacted = len(positions)
    recovered_count = 0
    source_doc_ids = []
    recovered_segments = []
    merged = base_text

    max_passes = 3
    for pass_num in range(max_passes):
        current_positions = find_redaction_positions(merged, redaction_markers)
        if not current_positions:
            break

        pre_pass_count = recovered_count

        # --- Anchor matching with adaptive widths ---
        # Process in reverse order so string positions remain valid after substitution
        for pos, marker in reversed(current_positions):
            left_anchor, right_anchor = extract_anchors(merged, pos, len(marker), anchor_length, redaction_markers)

            # Anchor quality floor: skip if combined anchors lack alphanumeric content
            alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
            if len(alpha_content) < 8:
                logger.debug("Skipping redaction at pos %d (pass %d): anchor quality too low (%d alphanumeric chars)", pos, pass_num, len(alpha_content))
                continue

            # Try progressively wider anchors if the default width fails
            anchor_widths = sorted(set([anchor_length, 100, 150, 200]))
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
                # Re-scan for remaining after this donor (positions shifted)
                remaining_positions = find_redaction_positions(merged, redaction_markers)

        # Early exit if no progress this pass
        if recovered_count == pre_pass_count:
            break

    # Flag questionable recoveries: same text recovered 3+ times is suspicious
    from collections import Counter
    text_counts = Counter(seg["text"] for seg in recovered_segments)
    for seg in recovered_segments:
        if text_counts[seg["text"]] >= 3:
            seg["confidence"] = "questionable"

    return {
        "merged_text": merged,
        "recovered_count": recovered_count,
        "total_redacted": total_redacted,
        "source_doc_ids": [base_id] + source_doc_ids,
        "recovered_segments": recovered_segments,
    }


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
