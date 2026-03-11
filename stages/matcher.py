"""Stage 2: Matcher — find document groups that share overlapping content.

Logic reference: PIPELINE.md — Phases 0, 2, and 3
(Phase 1 fingerprinting is done by the Indexer in Stage 1.)
"""

import re
import numpy as np
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from core.db import (
    get_connection, create_match_group, add_group_member,
    get_doc_group, merge_groups
)

# Email header patterns to extract for Phase 0 fast-path
_HEADER_PATTERNS = [
    re.compile(r"^From:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^To:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Date:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Subject:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
]

# Minimum common-text length (chars) required to confirm a match when no
# complementary redactions are present (secondary confirmation signal).
_SECONDARY_OVERLAP_THRESHOLD = 500


def extract_email_headers(text: str) -> list[str]:
    """Return a list of normalized header values found in the text."""
    headers = []
    for pattern in _HEADER_PATTERNS:
        for match in pattern.finditer(text):
            headers.append(match.group(1).strip())
    return headers


def run_phase0_email_fastpath(conn, min_header_matches: int = 2) -> set[int]:
    """Group documents that share min_header_matches or more identical headers.

    Returns the set of doc_ids that were matched and grouped.
    """
    rows = conn.execute(
        "SELECT id, extracted_text FROM documents WHERE extracted_text IS NOT NULL"
    ).fetchall()

    # Build index: header_value → [doc_ids]
    header_index: dict[str, list[int]] = defaultdict(list)
    doc_headers: dict[int, list[str]] = {}
    for row in rows:
        headers = extract_email_headers(row["extracted_text"] or "")
        doc_headers[row["id"]] = headers
        for h in headers:
            header_index[h.lower()].append(row["id"])

    # Find pairs sharing >= min_header_matches headers
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for header_val, doc_ids in header_index.items():
        if len(doc_ids) < 2:
            continue
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                pair = (min(doc_ids[i], doc_ids[j]), max(doc_ids[i], doc_ids[j]))
                pair_counts[pair] += 1

    matched: set[int] = set()
    for (doc_a, doc_b), count in pair_counts.items():
        if count >= min_header_matches:
            _assign_to_group(conn, doc_a, doc_b, similarity=1.0)
            matched.add(doc_a)
            matched.add(doc_b)

    return matched


def load_fingerprints(conn, num_perm: int = 128) -> dict[int, MinHash]:
    """Load all stored fingerprints from DB and reconstruct MinHash objects."""
    rows = conn.execute(
        "SELECT doc_id, minhash_sig FROM document_fingerprints"
    ).fetchall()
    result = {}
    for row in rows:
        hashvalues = np.frombuffer(row["minhash_sig"], dtype=np.uint64)
        m = MinHash(num_perm=num_perm)
        m.hashvalues = hashvalues.copy()
        result[row["doc_id"]] = m
    return result


def run_phase2_lsh_candidates(
    conn, threshold: float = 0.70, num_perm: int = 128
) -> list[tuple[int, int]]:
    """Use LSH banding to find candidate pairs likely to be the same document.

    Excludes documents already grouped by Phase 0 — they need no further matching.
    Returns list of (doc_id_a, doc_id_b) candidate pairs for Phase 3 verification.
    """
    fingerprints = load_fingerprints(conn, num_perm=num_perm)
    if len(fingerprints) < 2:
        return []

    # Exclude docs already assigned to a group by Phase 0
    already_grouped = {
        row["doc_id"]
        for row in conn.execute(
            "SELECT doc_id FROM match_group_members"
        ).fetchall()
    }
    fingerprints = {
        doc_id: mh for doc_id, mh in fingerprints.items()
        if doc_id not in already_grouped
    }
    if len(fingerprints) < 2:
        return []

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for doc_id, minhash in fingerprints.items():
        lsh.insert(str(doc_id), minhash)

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for doc_id, minhash in fingerprints.items():
        neighbors = lsh.query(minhash)
        for neighbor_id_str in neighbors:
            neighbor_id = int(neighbor_id_str)
            if neighbor_id == doc_id:
                continue
            pair = (min(doc_id, neighbor_id), max(doc_id, neighbor_id))
            if pair not in seen:
                seen.add(pair)
                candidates.append(pair)

    return candidates


def find_longest_common_substring(
    text_a: str, text_b: str, redaction_markers: list[str]
) -> str:
    """Find common text between two documents, ignoring redaction markers.

    Strips redaction markers from both texts, then collects all common substrings
    of length >= 10 chars and returns them concatenated. This allows the result to
    span regions separated by redaction markers in the original texts.
    """
    def strip_markers(t):
        for m in redaction_markers:
            t = t.replace(m, " ")
        return re.sub(r"\s+", " ", t).strip()

    a = strip_markers(text_a)
    b = strip_markers(text_b)

    if not a or not b:
        return ""

    # Collect all common substrings of length >= min_seg using DP
    min_seg = 10
    max_chars = 2000
    if len(a) > max_chars or len(b) > max_chars:
        # For very long texts, just find the single longest common substring
        return _lcs_single(a, b, min_seg)

    return _collect_common_segments(a, b, min_seg)


def _collect_common_segments(a: str, b: str, min_seg: int) -> str:
    """Collect all common substrings >= min_seg chars using DP and concatenate them."""
    m, n = len(a), len(b)
    # DP table for common substring lengths
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    # Collect all (start_in_a, length) pairs for common substrings >= min_seg
    segments: list[tuple[int, int]] = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                run_len = curr[j]
                if run_len >= min_seg:
                    # Record the end of this segment; we'll deduplicate later
                    seg_start = i - run_len
                    segments.append((seg_start, run_len))
            else:
                curr[j] = 0
        prev, curr = curr, [0] * (n + 1)

    if not segments:
        return ""

    # Keep only maximal segments (remove subsegments)
    # Sort by start position, then by length descending
    segments.sort(key=lambda x: (x[0], -x[1]))
    # Merge overlapping/nested segments and collect unique text
    merged: list[tuple[int, int]] = []
    for start, length in segments:
        end = start + length
        if merged and start < merged[-1][0] + merged[-1][1]:
            # Overlaps or is nested — extend if needed
            prev_start, prev_len = merged[-1]
            prev_end = prev_start + prev_len
            if end > prev_end:
                merged[-1] = (prev_start, end - prev_start)
        else:
            merged.append((start, length))

    return " ".join(a[s:s + l] for s, l in merged)


def _lcs_single(a: str, b: str, min_len: int) -> str:
    """Find the single longest common substring for large texts."""
    best = ""
    max_len = min(len(a), len(b), 500)
    for length in range(max_len, min_len - 1, -1):
        found = False
        for start in range(0, len(a) - length + 1):
            substr = a[start:start + length]
            if substr in b:
                best = substr
                found = True
                break
        if found:
            break
    return best


def _has_complementary_redactions(
    text_a: str, text_b: str, redaction_markers: list[str]
) -> bool:
    """Return True if one doc has text where the other has a redaction marker."""
    for marker in redaction_markers:
        if marker in text_a and marker not in text_b:
            return True
        if marker in text_b and marker not in text_a:
            return True
    return False


def run_phase3_verify_and_group(
    conn, candidates: list[tuple[int, int]],
    redaction_markers: list[str],
    min_overlap_chars: int = 200
) -> None:
    """Verify candidate pairs and group confirmed matches.

    Logic reference: PIPELINE.md — Phase 3

    Primary confirmation: complementary redactions (one has text where other is redacted).
    Secondary: long common text (>500 chars) even without complementary redactions.
    Rejection: common text shorter than min_overlap_chars.
    """
    texts = {
        row["id"]: row["extracted_text"]
        for row in conn.execute(
            "SELECT id, extracted_text FROM documents WHERE extracted_text IS NOT NULL"
        ).fetchall()
    }

    for doc_a, doc_b in candidates:
        text_a = texts.get(doc_a, "")
        text_b = texts.get(doc_b, "")

        common = find_longest_common_substring(text_a, text_b, redaction_markers)
        if len(common) < min_overlap_chars:
            continue  # Insufficient overlap — reject

        has_complementary = _has_complementary_redactions(text_a, text_b, redaction_markers)
        if not has_complementary and len(common) <= _SECONDARY_OVERLAP_THRESHOLD:
            continue  # Weak evidence — not enough to confirm match

        # At least one confirmation signal met — assign to group
        _assign_to_group(conn, doc_a, doc_b,
                         similarity=len(common) / max(len(text_a), len(text_b), 1))


def _assign_to_group(conn, doc_a: int, doc_b: int, similarity: float) -> None:
    """Assign two documents to a shared match group, merging if needed."""
    group_a = get_doc_group(conn, doc_a)
    group_b = get_doc_group(conn, doc_b)

    if group_a is not None and group_b is not None:
        if group_a != group_b:
            merge_groups(conn, group_a, group_b)
    elif group_a is not None:
        add_group_member(conn, group_a, doc_b, similarity)
    elif group_b is not None:
        add_group_member(conn, group_b, doc_a, similarity)
    else:
        new_group = create_match_group(conn)
        add_group_member(conn, new_group, doc_a, 1.0)
        add_group_member(conn, new_group, doc_b, similarity)
