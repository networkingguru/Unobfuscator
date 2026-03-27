"""Stage 3: Merger — fill redaction gaps using anchor-phrase matching across match groups.

Logic reference: PIPELINE.md — Phase 4 (Merging)
"""

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
        return {"merged_text": "", "recovered_count": 0, "total_redacted": 0, "source_doc_ids": [], "recovered_segments": []}

    # Pick base: most redaction markers (first after sort); donors fill its gaps
    base_id = members[0]["id"]
    base_text = members[0]["extracted_text"] or ""
    donors = [(row["id"], row["extracted_text"] or "") for row in members[1:]]

    positions = find_redaction_positions(base_text, redaction_markers)
    total_redacted = len(positions)
    recovered_count = 0
    source_doc_ids = []
    recovered_segments = []
    merged = base_text

    # Process in reverse order so string positions remain valid after substitution
    for pos, marker in reversed(positions):
        left_anchor, right_anchor = extract_anchors(base_text, pos, len(marker), anchor_length, redaction_markers)

        # Anchor quality floor: skip if combined anchors lack alphanumeric content
        alpha_content = re.sub(r'[^a-zA-Z0-9]', '', left_anchor + right_anchor)
        if len(alpha_content) < 8:
            logger.debug("Skipping redaction at pos %d: anchor quality too low (%d alphanumeric chars)", pos, len(alpha_content))
            continue

        for donor_id, donor_text in donors:
            recovered = find_text_between_anchors(donor_text, left_anchor, right_anchor)
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
                break  # Found recovery for this gap — move to next

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
