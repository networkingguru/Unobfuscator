# LSH Memory Guard — Design Spec

**Date:** 2026-03-19
**Status:** Approved

## Problem

The LSH matcher stage loads 1.3M+ MinHash objects into memory (~3-4 GB) then builds a MinHashLSH index (another ~3-4 GB), peaking at ~10 GB. On machines with limited RAM this can cause the system to swap heavily or OOM-kill the daemon. There is no protection or warning — the daemon silently consumes all available memory.

## Solution: Inline RSS Guard

Add a memory check at natural batch boundaries during fingerprint loading. If RSS exceeds a configurable percentage of system RAM, abort the LSH stage gracefully, log a diagnostic message with root cause and tips, write a flag to the DB so the `status` command surfaces the warning, and continue with lighter stages.

### Config

New section in `config.yaml`:

```yaml
memory:
  limit_percent: 70  # max % of system RAM before aborting LSH
```

Default 70% if absent. Acceptable range: 10–95. Values outside this range are clamped with a warning.

### Getting System RAM

Use `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')` on Linux. On macOS, `os.sysconf('SC_PHYS_PAGES')` may not be available — fall back to `subprocess.check_output(['sysctl', '-n', 'hw.memsize'])`. No new dependencies required.

### Getting Current RSS

`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` — returns bytes on macOS, KB on Linux. Detect platform via `sys.platform` and normalize to bytes.

### Check Points

Two check points, both in `stages/matcher.py`:

1. **`load_fingerprints()`** — after each 100k-row batch. Currently 13 checks over 1.3M rows (~one every 30 seconds). If over limit, raise `MemoryLimitExceeded`.

2. **`run_phase2_lsh_candidates()`** — catches `MemoryLimitExceeded` from `load_fingerprints()`. Logs warning, writes DB flag, returns `[]`.

No check during LSH insertion or querying. If loading completes within budget, the index build adds ~30-50% more — if that's a problem, the user should lower `limit_percent`.

### MemoryLimitExceeded Exception

New exception in `stages/matcher.py`:

```python
class MemoryLimitExceeded(Exception):
    """Raised when RSS exceeds the configured memory limit."""
    def __init__(self, rss_mb: int, limit_mb: int, limit_pct: int, total_mb: int):
        self.rss_mb = rss_mb
        self.limit_mb = limit_mb
        self.limit_pct = limit_pct
        self.total_mb = total_mb
```

### Helper Functions

Add to `stages/matcher.py`:

```python
def _get_total_ram_bytes() -> int:
    """Return total physical RAM in bytes."""

def _get_rss_bytes() -> int:
    """Return current process RSS in bytes."""

def _check_memory(limit_pct: int) -> None:
    """Raise MemoryLimitExceeded if RSS exceeds limit_pct of total RAM."""
```

These are private to the module. No new files needed.

### Abort Behavior

When `MemoryLimitExceeded` is raised:

1. `run_phase2_lsh_candidates()` catches it
2. Logs a WARNING with this format:
   ```
   LSH aborted — memory limit exceeded (RSS: 8,547 MB, limit: 8,192 MB / 50% of 16,384 MB).
   Root cause: 1.2M MinHash objects + LSH index exceed available memory budget.
   Tips:
     - Increase memory.limit_percent in config.yaml (current: 50)
     - Run on a machine with more RAM (need ~10 GB for 1.3M fingerprints)
     - Reduce fingerprint count by processing fewer batches
   ```
3. Writes `set_config(conn, "lsh_memory_warning", "<message>")` with a short summary
4. Does NOT update `lsh_last_fp_count` / `lsh_last_group_count` — so next cycle with new fingerprints will retry
5. Returns `[]`

### Clearing the Warning

On a successful LSH completion (after `lsh.query()` finishes), delete the flag:
```python
set_config(conn, "lsh_memory_warning", "")
```

### Status Display

`unobfuscator.py status` checks `get_config(conn, "lsh_memory_warning")`. If non-empty, adds a row:

```
│ LSH Warning         │ Skipped: memory limit exceeded (8,547 MB / 70%)  │
```

### What This Does NOT Do

- No background thread or polling — inline checks only
- No protection for other stages (they stay under 500MB)
- No kill/restart — graceful skip only
- No new Python dependencies

## File Changes

| File | Change |
|------|--------|
| `stages/matcher.py` | Add `MemoryLimitExceeded`, `_get_total_ram_bytes()`, `_get_rss_bytes()`, `_check_memory()`. Add check in `load_fingerprints()` loop. Catch exception in `run_phase2_lsh_candidates()`. Clear flag on success. |
| `unobfuscator.py` | Pass `memory_limit_pct` from config to `run_phase2_lsh_candidates()`. Add LSH warning row to `status` command. |
| `config.yaml` | Add `memory.limit_percent: 70`. |

## Testing

- Unit test `_check_memory()` with mocked `_get_total_ram_bytes` and `_get_rss_bytes`
- Unit test that `load_fingerprints()` raises `MemoryLimitExceeded` when limit is exceeded mid-load
- Unit test that `run_phase2_lsh_candidates()` catches the exception, returns `[]`, and writes the DB flag
- Unit test that a successful LSH run clears the flag
