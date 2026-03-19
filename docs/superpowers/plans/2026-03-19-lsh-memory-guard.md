# LSH Memory Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a memory guard that aborts the LSH matcher stage if RSS exceeds a configurable percentage of system RAM, with diagnostic logging and status display.

**Architecture:** Three private helper functions in `stages/matcher.py` (`_get_total_ram_bytes`, `_get_rss_bytes`, `_check_memory`) check current RSS at each batch boundary during fingerprint loading. A `MemoryLimitExceeded` exception propagates to `run_phase2_lsh_candidates` which logs a diagnostic warning and writes a DB flag surfaced by the `status` command.

**Tech Stack:** Python stdlib (`ctypes`, `os`, `sys`, `resource`), SQLite (existing `set_config`/`get_config`)

**Spec:** `docs/superpowers/specs/2026-03-19-lsh-memory-guard-design.md`

---

### Task 1: Memory Measurement Helpers

**Files:**
- Modify: `stages/matcher.py:1-17` (imports and module-level code)
- Test: `tests/stages/test_matcher.py`

- [ ] **Step 1: Write failing tests for RSS and RAM helpers**

Add to `tests/stages/test_matcher.py`:

```python
from stages.matcher import _get_total_ram_bytes, _get_rss_bytes


def test_get_total_ram_bytes_returns_positive_int():
    total = _get_total_ram_bytes()
    assert isinstance(total, int)
    assert total > 0
    # Sanity: at least 512 MB, at most 1 TB
    assert total >= 512 * 1024 * 1024
    assert total <= 1024 * 1024 * 1024 * 1024


def test_get_rss_bytes_returns_positive_int():
    rss = _get_rss_bytes()
    assert isinstance(rss, int)
    assert rss > 0
    # Current process should be at least 10 MB (Python interpreter)
    assert rss >= 10 * 1024 * 1024


def test_rss_is_less_than_total_ram():
    rss = _get_rss_bytes()
    total = _get_total_ram_bytes()
    assert rss < total
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/stages/test_matcher.py -v -k "ram_bytes or rss_bytes or rss_is_less" 2>&1 | tail -10`
Expected: ImportError for `_get_total_ram_bytes`

- [ ] **Step 3: Implement `_get_total_ram_bytes` and `_get_rss_bytes`**

Add imports at the top of `stages/matcher.py` (after existing imports):

```python
import ctypes
import ctypes.util
import os
import sys
```

Add after the `_SECONDARY_OVERLAP_THRESHOLD` constant (after line 29):

```python
def _get_total_ram_bytes() -> int:
    """Return total physical RAM in bytes."""
    if sys.platform == "darwin":
        import subprocess
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                           text=True).strip()
            return int(out)
        except Exception:
            return 16 * 1024 * 1024 * 1024  # 16 GB fallback
    # Linux: os.sysconf is reliable
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 16 * 1024 * 1024 * 1024  # 16 GB fallback


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/stages/test_matcher.py -v -k "ram_bytes or rss_bytes or rss_is_less" 2>&1 | tail -10`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "feat: add _get_total_ram_bytes and _get_rss_bytes helpers"
```

---

### Task 2: Memory Check and Exception

**Files:**
- Modify: `stages/matcher.py:30-70` (after the RAM helpers)
- Test: `tests/stages/test_matcher.py`

- [ ] **Step 1: Write failing tests for `_check_memory` and `MemoryLimitExceeded`**

Add to `tests/stages/test_matcher.py`:

```python
from unittest.mock import patch
from stages.matcher import _check_memory, MemoryLimitExceeded


def test_check_memory_passes_when_under_limit():
    """No exception when RSS is well under the limit."""
    with patch("stages.matcher._get_rss_bytes", return_value=1_000_000_000), \
         patch("stages.matcher._get_total_ram_bytes", return_value=16_000_000_000):
        # 1 GB / 16 GB = 6.25%, limit is 70% — should pass
        _check_memory(70)  # no exception


def test_check_memory_raises_when_over_limit():
    """MemoryLimitExceeded when RSS exceeds limit_pct of total RAM."""
    with patch("stages.matcher._get_rss_bytes", return_value=12_000_000_000), \
         patch("stages.matcher._get_total_ram_bytes", return_value=16_000_000_000):
        # 12 GB / 16 GB = 75%, limit is 70% — should raise
        with pytest.raises(MemoryLimitExceeded) as exc_info:
            _check_memory(70)
        assert exc_info.value.rss_mb == 12_000_000_000 // (1024 * 1024)
        assert exc_info.value.limit_mb == (16_000_000_000 * 70 // 100) // (1024 * 1024)
        assert exc_info.value.limit_pct == 70
        assert exc_info.value.total_mb == 16_000_000_000 // (1024 * 1024)


def test_check_memory_skips_when_rss_unavailable():
    """If _get_rss_bytes returns 0, guard is disabled — no exception."""
    with patch("stages.matcher._get_rss_bytes", return_value=0), \
         patch("stages.matcher._get_total_ram_bytes", return_value=16_000_000_000):
        _check_memory(70)  # no exception


def test_memory_limit_exceeded_has_correct_attributes():
    exc = MemoryLimitExceeded(rss_mb=8547, limit_mb=8192, limit_pct=50, total_mb=16384)
    assert exc.rss_mb == 8547
    assert exc.limit_mb == 8192
    assert exc.limit_pct == 50
    assert exc.total_mb == 16384
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/stages/test_matcher.py -v -k "check_memory or memory_limit_exceeded" 2>&1 | tail -10`
Expected: ImportError for `_check_memory`

- [ ] **Step 3: Implement `MemoryLimitExceeded` and `_check_memory`**

Add to `stages/matcher.py` after `_get_rss_bytes`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/stages/test_matcher.py -v -k "check_memory or memory_limit_exceeded" 2>&1 | tail -10`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "feat: add MemoryLimitExceeded exception and _check_memory guard"
```

---

### Task 3: Wire Guard into load_fingerprints and run_phase2_lsh_candidates

**Files:**
- Modify: `stages/matcher.py:104-222` (load_fingerprints and run_phase2_lsh_candidates)
- Test: `tests/stages/test_matcher.py`

- [ ] **Step 1: Write failing tests for guard integration**

Add to `tests/stages/test_matcher.py`:

```python
from core.db import get_config


def test_load_fingerprints_raises_when_memory_exceeded(conn):
    """load_fingerprints should raise MemoryLimitExceeded when over limit."""
    # Insert enough fingerprints to trigger a batch boundary
    for i in range(5):
        sig = build_fingerprint(clean_text(f"unique text number {i} " * 50, []))
        upsert_fingerprint(conn, f"doc_{i}", sig, 100)
    conn.commit()

    with patch("stages.matcher._check_memory",
               side_effect=MemoryLimitExceeded(8547, 8192, 70, 16384)):
        with pytest.raises(MemoryLimitExceeded):
            load_fingerprints(conn, memory_limit_pct=70)


def test_phase2_catches_memory_exceeded_returns_empty(conn):
    """run_phase2_lsh_candidates should return [] and write DB flag on memory exceeded."""
    for i in range(5):
        sig = build_fingerprint(clean_text(f"unique text number {i} " * 50, []))
        upsert_fingerprint(conn, f"doc_{i}", sig, 100)
    conn.commit()

    with patch("stages.matcher._check_memory",
               side_effect=MemoryLimitExceeded(8547, 8192, 70, 16384)):
        result = run_phase2_lsh_candidates(conn, memory_limit_pct=70)

    assert result == []
    warning = get_config(conn, "lsh_memory_warning", default="")
    assert "memory limit" in warning.lower()


def test_phase2_clears_warning_on_success(conn):
    """A successful LSH run should clear any previous memory warning."""
    from core.db import set_config
    set_config(conn, "lsh_memory_warning", "previous warning")
    conn.commit()

    # Insert 2 docs so LSH actually runs (needs >= 2 fingerprints)
    for i in range(2):
        doc = {
            "id": f"clear_test_{i}", "source": "test",
            "release_batch": "TEST", "original_filename": f"t{i}.pdf",
            "page_count": 1, "size_bytes": 100,
            "description": "test", "extracted_text": f"text {i} " * 50,
        }
        upsert_document(conn, doc)
        sig = build_fingerprint(clean_text(doc["extracted_text"], []))
        upsert_fingerprint(conn, doc["id"], sig, 100)
    conn.commit()

    run_phase2_lsh_candidates(conn, memory_limit_pct=70)

    warning = get_config(conn, "lsh_memory_warning", default="")
    assert warning == ""
```

Note: The `conn` fixture already exists in the test file (check existing fixtures). If not, add:

```python
@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/stages/test_matcher.py -v -k "memory_exceeded or clears_warning" 2>&1 | tail -10`
Expected: TypeError — `memory_limit_pct` not accepted

- [ ] **Step 3: Add `memory_limit_pct` parameter to `load_fingerprints`**

Update the signature and add the check inside the batch loop:

```python
def load_fingerprints(conn, num_perm: int = 128,
                      exclude: set = None,
                      memory_limit_pct: int = 0) -> dict[str, MinHash]:
```

Add docstring line: `memory_limit_pct: if > 0, check RSS after each batch and raise MemoryLimitExceeded if over.`

Inside the `while True` loop, after the `logger.info("Loaded ...")` line and before continuing to the next batch, add:

```python
        if memory_limit_pct > 0:
            _check_memory(memory_limit_pct)
```

- [ ] **Step 4: Add `memory_limit_pct` parameter to `run_phase2_lsh_candidates`**

Update the signature:

```python
def run_phase2_lsh_candidates(
    conn, threshold: float = 0.70, num_perm: int = 128,
    redaction_markers: list[str] = None,
    memory_limit_pct: int = 0,
) -> list[tuple[str, str]]:
```

Wrap the fingerprint loading in a try/except. Replace the existing `fingerprints = load_fingerprints(...)` call and the `del already_grouped` line with:

```python
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
```

At the end of the function, after the existing `conn.commit()` and before `return candidates`, clear any previous warning:

```python
    set_config(conn, "lsh_memory_warning", "")
    conn.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/stages/test_matcher.py -v 2>&1 | tail -20`
Expected: All PASS (including existing tests — ensure no regressions)

- [ ] **Step 6: Commit**

```bash
git add stages/matcher.py tests/stages/test_matcher.py
git commit -m "feat: wire memory guard into load_fingerprints and LSH candidates"
```

---

### Task 4: Config and Daemon Integration

**Files:**
- Modify: `config.yaml`
- Modify: `unobfuscator.py:112-116` (LSH call site)
- Modify: `unobfuscator.py:424-440` (status command)

- [ ] **Step 1: Add memory config section**

In `config.yaml`, add after the `ocr` section:

```yaml

memory:
  limit_percent: 70
```

- [ ] **Step 2: Pass `memory_limit_pct` from daemon to LSH**

In `unobfuscator.py`, update the `run_phase2_lsh_candidates` call (line 114-115):

```python
    mem_limit = cfg_get(cfg, "memory.limit_percent", default=70)
    candidates = run_phase2_lsh_candidates(conn, threshold=threshold,
                                            redaction_markers=markers,
                                            memory_limit_pct=mem_limit)
```

- [ ] **Step 3: Add LSH warning row to status command**

In `unobfuscator.py`, in the `status` function, after `output_dir = ...` (line 422) and before `t = Table(...)` (line 424), add:

```python
    lsh_warning = get_config(conn, "lsh_memory_warning", default="")
```

Then after `t.add_row("Stage 2 Matcher", ...)` (line 431), add:

```python
    if lsh_warning:
        t.add_row("LSH Warning", f"[yellow]{lsh_warning}[/yellow]")
```

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/ -v 2>&1 | tail -20`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add config.yaml unobfuscator.py
git commit -m "feat: add memory.limit_percent config, wire into daemon and status"
```

---

### Task 5: Full Integration Verification

- [ ] **Step 1: Verify the guard fires with a low limit**

Run (will trigger the guard since LSH needs ~10 GB but limit is ~1.6 GB):

```bash
cd /Users/brianhill/Scripts/Unobfuscator && python3 -c "
from core.db import init_db, get_connection, get_config
from stages.matcher import run_phase2_lsh_candidates
init_db('./data/unobfuscator.db')
conn = get_connection('./data/unobfuscator.db')
result = run_phase2_lsh_candidates(conn, memory_limit_pct=10)
print(f'Result: {len(result)} candidates')
warning = get_config(conn, 'lsh_memory_warning', default='')
print(f'Warning: {warning}')
conn.close()
"
```

Expected: `Result: 0 candidates` and a non-empty warning string.

- [ ] **Step 2: Verify status shows the warning**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 unobfuscator.py status`

Expected: Should show a yellow "LSH Warning" row with the memory exceeded message.

- [ ] **Step 3: Clear the warning (verify clearing works)**

Run:

```bash
cd /Users/brianhill/Scripts/Unobfuscator && python3 -c "
from core.db import get_connection, set_config
conn = get_connection('./data/unobfuscator.db')
set_config(conn, 'lsh_memory_warning', '')
conn.commit()
conn.close()
print('Warning cleared')
"
```

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 unobfuscator.py status`

Expected: No "LSH Warning" row in status table.

- [ ] **Step 4: Run full test suite one final time**

Run: `cd /Users/brianhill/Scripts/Unobfuscator && python3 -m pytest tests/ -v 2>&1 | tail -20`
Expected: All PASS
