"""Stage 2: Matcher — find document groups that share overlapping content.

Logic reference: PIPELINE.md — Phases 0, 2, and 3
(Phase 1 fingerprinting is done by the Indexer in Stage 1.)
"""

import array
import ctypes
import ctypes.util
import logging
import os
import re
import sys
import time
from itertools import combinations
import numpy as np
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from core.db import (
    get_connection, create_match_group, add_group_member,
    get_doc_group, merge_groups, get_config, set_config
)

logger = logging.getLogger(__name__)

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

# Maximum group size allowed before blocking a merge/add. Groups at or above
# this limit will have the pair recorded as a verified_pair instead of merging.
_MERGE_SIZE_LIMIT = 500

# --- Rolling-hash constants ---
_RH_BASE1, _RH_MOD1 = 131, (1 << 61) - 1   # Mersenne prime
_RH_BASE2, _RH_MOD2 = 137, (1 << 59) - 55  # large prime, different size
_MAX_BUCKET = 256  # discard high-frequency seeds to bound worst case
_MAX_SEEDS = 500_000  # cap total seeds to bound sort + extension work
_LCS_TIMEOUT = 30  # seconds — safety circuit breaker

_DEFAULT_RAM_BYTES = 16 * 1024 ** 3  # 16 GB fallback when detection fails
_total_ram_cache: int | None = None  # cached — total RAM never changes


def _get_total_ram_bytes() -> int:
    """Return total physical RAM in bytes (cached after first call)."""
    global _total_ram_cache
    if _total_ram_cache is not None:
        return _total_ram_cache

    if sys.platform == "darwin":
        import subprocess
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                           text=True).strip()
            _total_ram_cache = int(out)
            return _total_ram_cache
        except Exception:
            _total_ram_cache = _DEFAULT_RAM_BYTES
            return _total_ram_cache
    try:
        _total_ram_cache = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        _total_ram_cache = _DEFAULT_RAM_BYTES
    return _total_ram_cache


def _get_rss_bytes() -> int:
    """Return current process RSS in bytes.

    Uses mach_task_basic_info on macOS (ctypes) and /proc/self/statm
    on Linux. Returns 0 if neither method works (disables the guard).
    """
    if sys.platform == "darwin":
        try:
            libsys = ctypes.CDLL(ctypes.util.find_library("System"), use_errno=True)

            class _TaskBasicInfo(ctypes.Structure):
                _fields_ = [
                    ("suspend_count", ctypes.c_uint32),
                    ("virtual_size", ctypes.c_uint64),
                    ("resident_size", ctypes.c_uint64),
                    ("user_time_secs", ctypes.c_uint32),
                    ("user_time_usecs", ctypes.c_uint32),
                    ("system_time_secs", ctypes.c_uint32),
                    ("system_time_usecs", ctypes.c_uint32),
                    ("policy", ctypes.c_int32),
                ]

            MACH_TASK_BASIC_INFO = 20
            info = _TaskBasicInfo()
            count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
            task = libsys.mach_task_self()
            kr = libsys.task_info(
                task, MACH_TASK_BASIC_INFO, ctypes.byref(info), ctypes.byref(count)
            )
            if kr == 0:
                return info.resident_size
        except Exception:
            pass
        return 0

    # Linux: read /proc/self/statm
    try:
        with open("/proc/self/statm") as f:
            fields = f.read().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


class MemoryLimitExceeded(Exception):
    """Raised when RSS exceeds the configured memory limit."""
    def __init__(self, rss_mb: int, limit_mb: int, limit_pct: int, total_mb: int):
        self.rss_mb = rss_mb
        self.limit_mb = limit_mb
        self.limit_pct = limit_pct
        self.total_mb = total_mb
        super().__init__(
            f"RSS {rss_mb:,} MB exceeds limit {limit_mb:,} MB "
            f"({limit_pct}% of {total_mb:,} MB)"
        )


def _check_memory(limit_pct: int) -> None:
    """Raise MemoryLimitExceeded if current RSS exceeds limit_pct of total RAM.

    If RSS cannot be determined (returns 0), the guard is silently disabled.
    """
    rss = _get_rss_bytes()
    if rss == 0:
        return  # guard disabled — can't measure RSS on this platform
    total = _get_total_ram_bytes()
    limit = total * limit_pct // 100
    if rss > limit:
        mb = 1024 * 1024
        raise MemoryLimitExceeded(
            rss_mb=rss // mb,
            limit_mb=limit // mb,
            limit_pct=limit_pct,
            total_mb=total // mb,
        )


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
    Processes in batches to avoid loading all document text into memory.
    """

    # Only process docs not already in a group
    already_grouped = {
        row["doc_id"]
        for row in conn.execute("SELECT doc_id FROM match_group_members").fetchall()
    }

    # Build header index in batches to limit memory.
    # Only scan docs that look like emails (contain "From:" or "To:").
    header_index: dict[str, list[str]] = defaultdict(list)
    batch_size = 50000
    offset = 0
    total_scanned = 0

    while True:
        rows = conn.execute(
            "SELECT id, extracted_text FROM documents "
            "WHERE extracted_text IS NOT NULL "
            "AND (extracted_text LIKE '%From:%' OR extracted_text LIKE '%To:%') "
            "ORDER BY id LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        if not rows:
            break
        for row in rows:
            if row["id"] in already_grouped:
                continue
            headers = extract_email_headers(row["extracted_text"] or "")
            for h in headers:
                header_index[h.lower()].append(row["id"])
        total_scanned += len(rows)
        offset += batch_size

    logger.info("Phase 0: scanned %d docs, %d unique headers", total_scanned, len(header_index))

    # Find pairs sharing >= min_header_matches headers using combination hashing.
    #
    # Instead of building all O(n²) pairs per header and counting overlaps,
    # generate all C(h, min_header_matches) combinations of each document's
    # unique header values and group documents sharing any combination.
    #
    # Most docs have 3-4 headers → C(4,2) = 6 combos each.
    # With ~1M email docs this is ~6M hash lookups — completes in seconds.
    # Step 1: Build per-document header sets from the inverted index.
    doc_headers: dict[str, set[str]] = defaultdict(set)
    for header_val, doc_ids in header_index.items():
        if len(doc_ids) > 1000:
            continue  # skip noise headers (common dates, generic addresses)
        for doc_id in doc_ids:
            doc_headers[doc_id].add(header_val)

    logger.info("Phase 0: %d docs with eligible headers, building combination index...",
                len(doc_headers))

    # Step 2: For each doc, hash all C(h, k) header combinations → bucket.
    # Documents in the same bucket share >= k identical headers.
    combo_buckets: dict[tuple, list[str]] = defaultdict(list)
    for doc_id, headers in doc_headers.items():
        if len(headers) < min_header_matches:
            continue
        for combo in combinations(sorted(headers), min_header_matches):
            combo_buckets[combo].append(doc_id)
    del doc_headers  # free memory

    # Step 3: Group all documents in each bucket.
    matched: set[str] = set()
    groups_created = 0
    for combo, doc_ids in combo_buckets.items():
        if len(doc_ids) < 2:
            continue
        # Use first doc as anchor; assign all others to same group.
        anchor = doc_ids[0]
        for other in doc_ids[1:]:
            _assign_to_group(conn, anchor, other, similarity=1.0)
            matched.add(anchor)
            matched.add(other)
            groups_created += 1
            if groups_created % 50_000 == 0:
                logger.info("Phase 0: processed %d group assignments, %d docs matched so far",
                            groups_created, len(matched))
                conn.commit()
    del combo_buckets  # free memory

    logger.info("Phase 0: %d documents grouped via email headers (%d group assignments)",
                len(matched), groups_created)
    conn.commit()
    return matched


def load_fingerprints(conn, num_perm: int = 128,
                      exclude: set = None,
                      memory_limit_pct: int = 0) -> dict[str, MinHash]:
    """Load stored fingerprints from DB and reconstruct MinHash objects.

    Args:
        exclude: set of doc_ids to skip (e.g. already-grouped docs).
                 Saves memory by not creating MinHash objects we'd discard.
    """
    total = conn.execute("SELECT COUNT(*) FROM document_fingerprints").fetchone()[0]
    logger.info("Loading %d fingerprints from DB...", total)

    result = {}
    batch_size = 100000
    offset = 0
    while True:
        rows = conn.execute(
            "SELECT doc_id, minhash_sig FROM document_fingerprints "
            "ORDER BY doc_id LIMIT ? OFFSET ?",
            (batch_size, offset)
        ).fetchall()
        if not rows:
            break
        for row in rows:
            if exclude and row["doc_id"] in exclude:
                continue
            hashvalues = np.frombuffer(row["minhash_sig"], dtype=np.uint64)
            m = MinHash(num_perm=num_perm)
            m.hashvalues = hashvalues.copy()
            result[row["doc_id"]] = m
        offset += batch_size
        logger.info("Loaded %d / %d fingerprints (skipped grouped)",
                     len(result), total)
        if memory_limit_pct > 0:
            _check_memory(memory_limit_pct)

    return result


def run_phase2_lsh_candidates(
    conn, threshold: float = 0.70, num_perm: int = 128,
    redaction_markers: list[str] = None,
    memory_limit_pct: int = 0,
) -> list[tuple[str, str]]:
    """Use LSH banding to find candidate pairs likely to be the same document.

    Only queries docs that contain redaction markers (since pairs of clean docs
    can't produce any recoveries). This dramatically reduces candidate count.
    Returns list of (doc_id_a, doc_id_b) candidate pairs for Phase 3 verification.

    Skips the expensive LSH rebuild when the fingerprint count and group count
    haven't changed since the last run (nothing new to match).
    """

    # Skip if nothing has changed since the last LSH run
    fp_count = conn.execute("SELECT COUNT(*) FROM document_fingerprints").fetchone()[0]
    group_count = conn.execute("SELECT COUNT(*) FROM match_group_members").fetchone()[0]
    last_fp_count = get_config(conn, "lsh_last_fp_count", default=0)
    last_group_count = get_config(conn, "lsh_last_group_count", default=0)
    if fp_count == last_fp_count and group_count == last_group_count and fp_count > 0:
        logger.info("Skipping LSH rebuild — no new fingerprints or groups since last run "
                     "(%d fingerprints, %d group members)", fp_count, group_count)
        return []

    # Exclude docs already assigned to a group — skip them during loading
    # so we never allocate MinHash objects for them.
    already_grouped = {
        row["doc_id"]
        for row in conn.execute(
            "SELECT doc_id FROM match_group_members"
        ).fetchall()
    }

    try:
        fingerprints = load_fingerprints(conn, num_perm=num_perm,
                                          exclude=already_grouped,
                                          memory_limit_pct=memory_limit_pct)
    except MemoryLimitExceeded as e:
        logger.warning(
            "LSH aborted — memory limit exceeded "
            "(RSS: %s MB, limit: %s MB / %d%% of %s MB).\n"
            "Root cause: MinHash fingerprint loading exceeded available memory budget.\n"
            "Tips:\n"
            "  - Increase memory.limit_percent in config.yaml (current: %d)\n"
            "  - Run on a machine with more RAM\n"
            "  - Reduce fingerprint count by processing fewer batches",
            f"{e.rss_mb:,}", f"{e.limit_mb:,}", e.limit_pct,
            f"{e.total_mb:,}", e.limit_pct
        )
        set_config(conn, "lsh_memory_warning",
                   f"Skipped: memory limit exceeded "
                   f"({e.rss_mb:,} MB / {e.limit_pct}%)")
        conn.commit()
        return []
    del already_grouped  # free the set
    if len(fingerprints) < 2:
        return []

    logger.info("Building LSH index with %d fingerprints...", len(fingerprints))
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for doc_id, minhash in fingerprints.items():
        lsh.insert(str(doc_id), minhash)

    # Only query docs that have redaction markers — no point matching
    # two clean docs since no recovery is possible.
    redacted_ids = set()
    if redaction_markers:
        conditions = " OR ".join(
            "extracted_text LIKE ?" for _ in redaction_markers
        )
        params = [f"%{m}%" for m in redaction_markers]
        rows = conn.execute(
            f"SELECT id FROM documents WHERE {conditions}", params
        ).fetchall()
        redacted_ids = {r["id"] for r in rows} & set(fingerprints.keys())
        logger.info("Querying LSH for %d redacted docs (of %d total)",
                     len(redacted_ids), len(fingerprints))
    else:
        redacted_ids = set(fingerprints.keys())

    candidates: list[tuple] = []
    seen: set[tuple] = set()
    for doc_id in redacted_ids:
        if doc_id not in fingerprints:
            continue
        neighbors = lsh.query(fingerprints[doc_id])
        for neighbor_id in neighbors:
            if neighbor_id == str(doc_id):
                continue
            pair = tuple(sorted([str(doc_id), neighbor_id]))
            if pair not in seen:
                seen.add(pair)
                candidates.append(pair)

    logger.info("LSH produced %d candidate pairs", len(candidates))

    # Remember current counts so we can skip the rebuild next cycle if nothing changed
    set_config(conn, "lsh_last_fp_count", fp_count)
    set_config(conn, "lsh_last_group_count", group_count)
    set_config(conn, "lsh_memory_warning", "")
    conn.commit()

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
        # Collapse Unicode full-block redaction runs (█, U+2588) from OCR output
        t = re.sub(r"\u2588+", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    a = strip_markers(text_a)
    b = strip_markers(text_b)

    if not a or not b:
        return ""

    # Collect all common substrings of length >= min_seg.
    # Scale min_seg with document size to bound rolling-hash seed counts.
    # Cap at 50 — well below the 200-char Phase 3 rejection threshold.
    shorter = min(len(a), len(b))
    min_seg = max(10, min(50, shorter // 2000))
    max_chars = 2000
    if len(a) > max_chars or len(b) > max_chars:
        # Scale bucket cap inversely with doc size to bound total seeds.
        # Target: total seeds < _MAX_SEEDS even if every window matches.
        windows = max(1, shorter - min_seg)
        max_bucket = max(4, min(_MAX_BUCKET, _MAX_SEEDS // windows))
        return _collect_common_segments_rolling_hash(a, b, min_seg, max_bucket)

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


def _collect_common_segments_rolling_hash(a: str, b: str, min_seg: int,
                                          max_bucket: int = _MAX_BUCKET) -> str:
    """Find all common substrings >= min_seg chars using rolling-hash seed-and-extend.

    Same semantics as _collect_common_segments (DP), but O(n+m+S) instead of O(n*m).
    Uses double-hashing to avoid false positives and a bucket cap to prevent
    degenerate O(n*m) behavior on highly repetitive text.
    """
    if len(a) < min_seg or len(b) < min_seg:
        return ""

    t_start = time.monotonic()

    # Precompute base^(min_seg-1) mod prime for rolling removal
    pow1 = pow(_RH_BASE1, min_seg - 1, _RH_MOD1)
    pow2 = pow(_RH_BASE2, min_seg - 1, _RH_MOD2)

    # Step 1: Hash all min_seg-length windows in `a` → dict of hash → positions
    index: dict[tuple[int, int], array.array] = {}
    h1 = h2 = 0
    for i, ch in enumerate(a):
        c = ord(ch)
        h1 = (h1 * _RH_BASE1 + c) % _RH_MOD1
        h2 = (h2 * _RH_BASE2 + c) % _RH_MOD2
        if i >= min_seg - 1:
            key = (h1, h2)
            bucket = index.get(key)
            if bucket is None:
                bucket = array.array("L")
                index[key] = bucket
            if len(bucket) < max_bucket:
                bucket.append(i - min_seg + 1)
            # Remove outgoing character
            out = ord(a[i - min_seg + 1])
            h1 = (h1 - out * pow1) % _RH_MOD1
            h2 = (h2 - out * pow2) % _RH_MOD2

    # Step 2: Scan `b`, collect seed pairs (pos_a, pos_b)
    seeds: list[tuple[int, int]] = []
    h1 = h2 = 0
    seed_overflow = False
    for j, ch in enumerate(b):
        c = ord(ch)
        h1 = (h1 * _RH_BASE1 + c) % _RH_MOD1
        h2 = (h2 * _RH_BASE2 + c) % _RH_MOD2
        if j >= min_seg - 1:
            key = (h1, h2)
            bucket = index.get(key)
            if bucket is not None:
                pos_b = j - min_seg + 1
                for pos_a in bucket:
                    seeds.append((pos_a, pos_b))
                if len(seeds) >= _MAX_SEEDS:
                    seed_overflow = True
                    break
            out = ord(b[j - min_seg + 1])
            h1 = (h1 - out * pow1) % _RH_MOD1
            h2 = (h2 - out * pow2) % _RH_MOD2

    if seed_overflow:
        logger.warning("Rolling-hash seed cap reached (%d seeds) on texts "
                       "len(a)=%d len(b)=%d min_seg=%d — results may be partial",
                       len(seeds), len(a), len(b), min_seg)

    if not seeds:
        return ""

    # Step 3: Sort by pos_a so we can skip already-covered regions
    seeds.sort()
    segments: list[tuple[int, int]] = []  # (start_in_a, length)
    covered_end_a = -1  # tracks how far we've extended in `a`

    for pos_a, pos_b in seeds:
        # Timeout guard
        if time.monotonic() - t_start > _LCS_TIMEOUT:
            logger.warning("Rolling-hash LCS timed out after %ds on texts "
                           "len(a)=%d len(b)=%d with %d seeds",
                           _LCS_TIMEOUT, len(a), len(b), len(seeds))
            break

        if pos_a < covered_end_a:
            continue  # already covered by a previous extension

        # Verify seed is a true match (not a hash collision)
        if a[pos_a:pos_a + min_seg] != b[pos_b:pos_b + min_seg]:
            continue

        # Extend forward
        end_a, end_b = pos_a + min_seg, pos_b + min_seg
        while end_a < len(a) and end_b < len(b) and a[end_a] == b[end_b]:
            end_a += 1
            end_b += 1

        # Extend backward
        start_a, start_b = pos_a, pos_b
        while start_a > 0 and start_b > 0 and a[start_a - 1] == b[start_b - 1]:
            start_a -= 1
            start_b -= 1

        seg_len = end_a - start_a
        if seg_len >= min_seg:
            segments.append((start_a, seg_len))
            covered_end_a = end_a

    if not segments:
        return ""

    # Step 4: Merge overlapping/adjacent segments (same logic as DP path)
    segments.sort(key=lambda x: (x[0], -x[1]))
    merged: list[tuple[int, int]] = []
    for start, length in segments:
        end = start + length
        if merged and start < merged[-1][0] + merged[-1][1]:
            prev_start, prev_len = merged[-1]
            prev_end = prev_start + prev_len
            if end > prev_end:
                merged[-1] = (prev_start, end - prev_start)
        else:
            merged.append((start, length))

    return " ".join(a[s:s + l] for s, l in merged)


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


def _get_text(conn, doc_id: str, _cache: dict = {}) -> str:
    """Fetch document text with simple LRU cache to avoid repeated DB reads."""
    doc_id = str(doc_id)
    if doc_id in _cache:
        return _cache[doc_id]
    row = conn.execute(
        "SELECT extracted_text FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    text = row["extracted_text"] if row and row["extracted_text"] else ""
    # Keep cache bounded
    if len(_cache) > 10000:
        _cache.clear()
    _cache[doc_id] = text
    return text


def run_phase3_verify_and_group(
    conn, candidates: list[tuple],
    redaction_markers: list[str],
    min_overlap_chars: int = 200,
    progress_callback=None,
    shutdown_check=None,
) -> None:
    """Verify candidate pairs and group confirmed matches.

    Logic reference: PIPELINE.md — Phase 3

    Primary confirmation: complementary redactions (one has text where other is redacted).
    Secondary: long common text (>500 chars) even without complementary redactions.
    Rejection: common text shorter than min_overlap_chars.

    progress_callback, if provided, is called periodically with
    (candidates_checked, total_candidates, confirmed_matches).
    """

    # Clear text cache for fresh run
    _get_text.__defaults__[0].clear()

    total = len(candidates)
    confirmed = 0
    rejected = 0
    t_start = time.monotonic()

    # Persist total so status can read it even between log intervals
    set_config(conn, "phase3_total", total)
    set_config(conn, "phase3_checked", 0)
    set_config(conn, "phase3_confirmed", 0)
    conn.commit()

    log_interval = 1_000  # log every N candidates

    for i, (doc_a, doc_b) in enumerate(candidates, 1):
        if shutdown_check and shutdown_check():
            logger.info("Phase 3 interrupted by shutdown after %d/%d candidates", i - 1, total)
            break
        text_a = _get_text(conn, doc_a)
        text_b = _get_text(conn, doc_b)

        common = find_longest_common_substring(text_a, text_b, redaction_markers)
        if len(common) < min_overlap_chars:
            rejected += 1
            if i % log_interval == 0:
                _phase3_progress(conn, i, total, confirmed, rejected,
                                 t_start, progress_callback)
            continue  # Insufficient overlap — reject

        has_complementary = _has_complementary_redactions(text_a, text_b, redaction_markers)
        if not has_complementary and len(common) <= _SECONDARY_OVERLAP_THRESHOLD:
            rejected += 1
            if i % log_interval == 0:
                _phase3_progress(conn, i, total, confirmed, rejected,
                                 t_start, progress_callback)
            continue  # Weak evidence — not enough to confirm match

        # At least one confirmation signal met — assign to group
        _assign_to_group(conn, doc_a, doc_b,
                         similarity=len(common) / max(len(text_a), len(text_b), 1))
        confirmed += 1
        if confirmed % 100 == 0:
            conn.commit()

        if i % log_interval == 0:
            _phase3_progress(conn, i, total, confirmed, rejected,
                             t_start, progress_callback)

    elapsed = time.monotonic() - t_start
    logger.info("Phase 3 complete: verified %d candidates → %d confirmed matches "
                "(%d rejected) in %.1f min",
                total, confirmed, rejected, elapsed / 60)

    # Clear progress keys now that phase is done
    set_config(conn, "phase3_total", 0)
    set_config(conn, "phase3_checked", 0)
    set_config(conn, "phase3_confirmed", 0)
    conn.commit()


def _phase3_progress(conn, checked, total, confirmed, rejected,
                     t_start, callback):
    """Log and persist Phase 3 progress."""
    elapsed = time.monotonic() - t_start
    pct = checked / total * 100 if total else 0
    rate = checked / elapsed if elapsed > 0 else 0
    remaining = (total - checked) / rate if rate > 0 else 0

    logger.info("Phase 3: %d / %d (%.1f%%) — %d confirmed, %d rejected — "
                "%.0f pairs/sec — ~%.0f min remaining",
                checked, total, pct, confirmed, rejected,
                rate, remaining / 60)

    set_config(conn, "phase3_checked", checked)
    set_config(conn, "phase3_confirmed", confirmed)
    conn.commit()

    if callback:
        callback(checked, total, confirmed)


def _assign_to_group(conn, doc_a: str, doc_b: str, similarity: float) -> None:
    """Assign two documents to a shared match group.

    When both docs are already in different groups and either group exceeds
    _MERGE_SIZE_LIMIT, record a verified pair instead of merging the groups.
    When one doc is ungrouped but the other's group exceeds the limit,
    leave the ungrouped doc ungrouped and record the pair.
    """
    from core.db import insert_verified_pair, get_group_member_count

    group_a = get_doc_group(conn, doc_a)
    group_b = get_doc_group(conn, doc_b)

    if group_a is not None and group_b is not None:
        if group_a != group_b:
            size_a = get_group_member_count(conn, group_a)
            size_b = get_group_member_count(conn, group_b)
            if size_a >= _MERGE_SIZE_LIMIT or size_b >= _MERGE_SIZE_LIMIT:
                insert_verified_pair(conn, doc_a, doc_b, similarity, phase="match")
            else:
                merge_groups(conn, group_a, group_b)
    elif group_a is not None:
        size_a = get_group_member_count(conn, group_a)
        if size_a >= _MERGE_SIZE_LIMIT:
            insert_verified_pair(conn, doc_a, doc_b, similarity, phase="match")
        else:
            add_group_member(conn, group_a, doc_b, similarity)
    elif group_b is not None:
        size_b = get_group_member_count(conn, group_b)
        if size_b >= _MERGE_SIZE_LIMIT:
            insert_verified_pair(conn, doc_a, doc_b, similarity, phase="match")
        else:
            add_group_member(conn, group_b, doc_a, similarity)
    else:
        new_group = create_match_group(conn)
        add_group_member(conn, new_group, doc_a, 1.0)
        add_group_member(conn, new_group, doc_b, similarity)
