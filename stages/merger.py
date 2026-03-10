"""Stage 3: Merger — fill redaction gaps using anchor-phrase matching across match groups.

Logic reference: PIPELINE.md — Phase 4 (Merging)
"""

from typing import Optional
from core.db import upsert_merge_result


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


def extract_anchors(text: str, pos: int, length: int = 50) -> tuple[str, str]:
    """Extract left and right anchor phrases around a position in text.

    Returns the full text before pos as left and full text after pos as right.
    The length parameter controls how many chars to use as anchor for matching.
    """
    left = text[:pos].strip()
    right = text[pos:].strip()
    return left, right


def find_text_between_anchors(
    text: str, left_anchor: str, right_anchor: str
) -> Optional[str]:
    """Search text for left_anchor, then find right_anchor after it.

    Returns the text between them if both anchors are found, else None.
    """
    if not left_anchor:
        # No left anchor — look for right anchor from start
        right_pos = text.find(right_anchor)
        if right_pos == -1:
            return None
        return text[:right_pos].strip()

    left_pos = text.find(left_anchor)
    if left_pos == -1:
        return None

    search_from = left_pos + len(left_anchor)
    right_pos = text.find(right_anchor, search_from)
    if right_pos == -1:
        return None

    return text[search_from:right_pos].strip()


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
        full_left, _ = extract_anchors(base_text, pos, anchor_length)
        # Use last anchor_length chars of left context for matching
        left_anchor = full_left[-anchor_length:].strip() if full_left else ""
        # Right anchor: text after the marker, limited to anchor_length chars
        right_start = pos + len(marker)
        right_anchor = base_text[right_start:right_start + anchor_length].strip()

        for donor_id, donor_text in donors:
            recovered = find_text_between_anchors(donor_text, left_anchor, right_anchor)
            if recovered and recovered not in redaction_markers and len(recovered) > 0:
                merged = merged[:pos] + recovered + merged[pos + len(marker):]
                recovered_count += 1
                recovered_segments.append({
                    "text": recovered,
                    "source_doc_id": donor_id,
                    "stage": "merge"
                })
                if donor_id not in source_doc_ids:
                    source_doc_ids.append(donor_id)
                break  # Found recovery for this gap — move to next

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
            result["source_doc_ids"]
        )
        conn.execute(
            "UPDATE match_groups SET merged = 1 WHERE group_id = ?", (group_id,)
        )
        conn.commit()
        count += 1

    return count
