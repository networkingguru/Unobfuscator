# Unobfuscator — Design Spec
**Date:** 2026-03-10
**Status:** Approved
**Data Source:** [Jmail Data API](https://jmail.world/docs/introduction)

---

## Overview

Unobfuscator is a CLI tool that cross-references the Jmail document corpus (1.41M documents from the Jeffrey Epstein archive) to recover redacted text wherever possible. It does this by:

1. Finding documents that are the same underlying document but with different portions redacted
2. Merging the unredacted portions across all versions to produce the most complete reconstruction
3. Stripping soft/overlay PDF redactions where the original text is still present in the file stream
4. Outputting highlighted PDFs with footnotes linking all source documents

The tool runs as a background daemon processing the full corpus, and also supports targeted manual searches that feed results into the shared dataset.

---

## Architecture

### File Structure

```
unobfuscator/
├── PIPELINE.md              ← Plain-English pseudocode. Source of truth for logic.
├── config.yaml              ← User configuration (output dir, thresholds, workers)
├── unobfuscator.py          ← CLI entry point
│
├── stages/
│   ├── indexer.py           ← Stage 1: Fetch docs from Jmail, build content index
│   ├── matcher.py           ← Stage 2: Find document groups sharing overlapping text
│   ├── merger.py            ← Stage 3: Merge groups, fill redactions from other versions
│   ├── pdf_processor.py     ← Stage 4: Download & strip soft PDF redactions (slow)
│   └── output_generator.py  ← Stage 5: Write highlighted output PDFs with footnotes
│
├── core/
│   ├── db.py                ← All SQLite reads/writes (one place, no SQL elsewhere)
│   ├── queue.py             ← Job queue: priority, status, manual vs background
│   └── api.py               ← Jmail API wrapper (caching, retries)
│
└── data/                    ← Created at runtime, gitignored
    ├── unobfuscator.db      ← SQLite: all state, progress, index, results
    └── cache/               ← Cached Parquet files from Jmail
```

### Stage Pipeline

```
[Jmail API]
    |
[Stage 1: Indexer]
    Fetches document metadata + extracted text → stores in DB
    Builds MinHash fingerprint per document → stored in DB
    |
[Stage 2: Matcher]
    Phase 0: Email chain fast-path (exact header matching)
    Phase 1: MinHash fingerprinting for remaining docs
    Phase 2: LSH banding to find candidate pairs
    Phase 3: Verify candidates, group confirmed matches
    |
[Stage 3: Merger]
    For each match group, aligns all versions
    Fills redaction gaps using anchor-phrase matching
    Records which source filled each gap
    |
[Stage 4: PDF Processor]  ← runs in parallel, slow background track
    Downloads original PDFs
    Detects and removes soft/overlay redactions
    Feeds recovered text back into merger
    |
[Stage 5: Output Generator]
    Only generates output if at least one redaction was recovered
    Builds PDF with recovered text highlighted
    Appends footnote page with source links
    Writes to configured output directory
```

### Two Modes, One Shared Dataset

- **Background mode** (`unobfuscator start`): processes all 1.41M docs as a daemon, low priority, saves progress at every step
- **Manual mode** (`unobfuscator search "Donald Trump"`): inserts high-priority (+100) jobs into the queue, results persist into the shared dataset so background mode skips them

---

## Data Model

All state lives in `unobfuscator.db`. All reads/writes go through `core/db.py`.

### Tables

```
documents
  id              INTEGER   Jmail document ID (primary key)
  source          TEXT      "doj" or "house_oversight"
  release_batch   TEXT      e.g. "VOL00008", "DataSet11"
  original_filename TEXT
  page_count      INTEGER
  size_bytes      INTEGER
  description     TEXT      AI-generated description from Jmail
  extracted_text  TEXT      Full text from Jmail API
  pdf_url         TEXT      URL to original PDF (discovered during pdf_processor stage)
  indexed_at      DATETIME
  text_processed  BOOLEAN   Has Stage 2 run on this doc?
  pdf_processed   BOOLEAN   Has Stage 4 run on this doc?

document_fingerprints
  doc_id          INTEGER   → documents.id
  minhash_sig     BLOB      128-hash MinHash signature (binary)
  shingle_count   INTEGER   How many shingles were generated
  created_at      DATETIME

match_groups
  group_id        INTEGER   Auto-increment primary key
  created_at      DATETIME
  merged          BOOLEAN   Has Stage 3 produced a result for this group?

match_group_members
  group_id        INTEGER   → match_groups.group_id
  doc_id          INTEGER   → documents.id
  similarity      REAL      Jaccard similarity score vs. group representative
  added_at        DATETIME

merge_results
  group_id                INTEGER   → match_groups.group_id (primary key)
  merged_text             TEXT      Best reconstruction of full document text
  recovered_count         INTEGER   How many redacted segments were filled
  previous_recovered_count INTEGER  Count before last update (for change tracking)
  total_redacted          INTEGER   Total redacted segments found across all versions
  source_doc_ids          TEXT      JSON array of doc IDs that contributed recovered text
  created_at              DATETIME
  updated_at              DATETIME
  output_generated        BOOLEAN

release_batches
  batch_id        TEXT      Primary key (e.g. "VOL00012")
  first_seen_at   DATETIME
  fully_indexed   BOOLEAN

jobs
  job_id          INTEGER   Auto-increment primary key
  stage           TEXT      "index" | "match" | "merge" | "pdf" | "output"
  payload         TEXT      JSON — e.g. {"doc_id": 12345} or {"query": "Trump"}
  priority        INTEGER   Higher = runs sooner. Manual jobs get +100 boost.
  status          TEXT      "pending" | "running" | "done" | "failed"
  error           TEXT      Error message if failed, null otherwise
  created_at      DATETIME
  updated_at      DATETIME

config
  key             TEXT      Primary key
  value           TEXT      JSON-encoded value
```

### Key Design Decisions

- `match_group_members` is a many-to-many join — one doc can only belong to one group, but groups accumulate members as more docs are indexed
- `jobs` drives all work across both modes — the only difference between background and manual is the `priority` value
- `config` table mirrors `config.yaml` but allows runtime overrides without editing the file
- All boolean progress flags (`text_processed`, `merged`, `output_generated`) are the resume mechanism — on restart, anything `False` gets re-queued

---

## Content Matching Algorithm

Defined fully in `PIPELINE.md`. Summary:

### Phase 0 — Email Chain Fast-Path (runs first)

```
FOR EACH document:
  Extract any email headers present (From, To, Date, Subject lines)

FOR EACH pair that shares 2+ identical headers:
  → Confirmed match. Group them immediately.
  → Mark both as "matched". Skip Phases 1–3 for this pair.
```

Most of the email corpus is handled here. Only unmatched documents proceed to Phase 1.

### Phase 1 — Fingerprinting

```
FOR EACH unmatched document:
  1. Clean the text — strip redaction markers, normalize whitespace, lowercase
  2. Slice into overlapping 8-word chunks ("shingles")
  3. Generate a 128-number MinHash signature from those chunks
  4. Store in document_fingerprints table
```

### Phase 2 — Candidate Finding (LSH Banding)

Rather than comparing every document to every other (500 billion comparisons), LSH Banding works as a shortcut:

```
Split each 128-number fingerprint into 16 groups of 8 numbers.
For each group, assign the document to a "bucket" based on those 8 numbers.
IF two documents land in the same bucket for ANY group:
  → Flag them as candidates worth inspecting closely.

Documents that share no buckets are very unlikely to be the same document
and are safely skipped. Result: only a small fraction of pairs are inspected.
```

### Phase 3 — Verification & Grouping

```
FOR EACH candidate pair (doc_A, doc_B):

  1. Find the longest common text segments (ignoring redaction markers)
  2. IF common text < 200 characters → Reject.
  3. Check for complementary redactions:
     IF doc_A has text where doc_B is redacted, OR vice versa:
       → Confirmed match. Group them.
  4. Assign to a match group:
     - IF doc_A already in a group → add doc_B to that group
     - IF doc_B already in a group → add doc_A to that group
     - IF both in different groups → merge the groups
     - IF neither → create new group with both
```

### Phase 4 — Merging (Stage 3)

```
FOR EACH match group with 2+ members:

  1. Pick the "base" document: the one with fewest redactions
  2. FOR EACH redacted segment in base document:
     a. Extract anchor phrases immediately before and after the redaction
     b. Search all other group members for those same anchor phrases
     c. IF found: extract the text between those anchors in the other doc
        → Recovered text. Record which source document provided it.
  3. Build merged_text: base document with recovered segments inserted
  4. Store in merge_results
```

### Redaction Markers (configurable)

```
[REDACTED]        [REDACTED TEXT]     [REDACTED PER]
[b(6)]            [b(7)(c)]           [b(7)(e)]
XXXXXXXXX         ■■■■■■■             (redacted)
*** REDACTED ***  <REDACTED>
```

---

## CLI Interface

### Background Mode

```
unobfuscator start     Start the background daemon (all 5 stages)
unobfuscator stop      Stop gracefully (finishes current job first)
unobfuscator status    Show progress across all stages and daemon state
```

Status output example:
```
┌─────────────────────────────────────────────────┐
│ Unobfuscator — Status                           │
│                                                 │
│ Daemon:     running (PID 4821)                  │
│                                                 │
│ Stage 1 Indexer:       841,204 / 1,410,000 docs │
│ Stage 2 Matcher:       612,000 / 841,204 indexed│
│ Stage 3 Merger:        18,443 groups merged      │
│ Stage 4 PDF Processor: 2,201 / 18,443 PDFs done │
│ Stage 5 Output:        1,876 files written       │
│                                                 │
│ Recovered redactions:  47,291 total              │
│ Output directory:      C:\Users\...\output\      │
└─────────────────────────────────────────────────┘
```

When all jobs complete, the daemon enters idle/polling state:
```
  Daemon: idle — polling for updates (next check in 47 min)
```

### Manual / Targeted Mode

```
unobfuscator search "Donald Trump"
unobfuscator search --person "Ghislaine Maxwell"
unobfuscator search --date 2000-01-01 2010-12-31
unobfuscator search --batch VOL00008
unobfuscator search --doc 12345

Flags:
  --wait        Block until results are ready, then print summary
  --output DIR  Override output directory for this run only
```

Manual search jobs get priority +100. Results persist into the shared dataset.

### Configuration Commands

```
unobfuscator config show
unobfuscator config set KEY VALUE
unobfuscator status --doc 12345    Show full DB record for a specific document
```

---

## Configuration (`config.yaml`)

```yaml
output_dir: ./output
cache_dir: ./data/cache
db_path: ./data/unobfuscator.db

workers:
  text: 4          # Parallel workers for stages 1-3 and 5
  pdf: 2           # Parallel workers for stage 4 (heavier, fewer)

matching:
  min_overlap_chars: 200
  similarity_threshold: 0.70
  email_header_min_matches: 2

polling:
  interval_minutes: 60

redaction_markers:
  - "[REDACTED]"
  - "[REDACTED TEXT]"
  - "[b(6)]"
  - "[b(7)(c)]"
  - "[b(7)(e)]"
  - "XXXXXXXXX"
  - "*** REDACTED ***"
  - "<REDACTED>"
```

---

## Output PDF Format

### Directory Structure

```
output/
└── doj/
    └── VOL00008/
        └── 12345_merged.pdf
└── house_oversight/
    └── DataSet11/
        └── 67890_merged.pdf
```

### PDF Generation

```
IF original PDF is available:
  → Use as template
  → Remove redaction overlay (if soft redaction, Stage 4)
    OR insert recovered text at redaction position (if cross-doc merge, Stage 3)
  → Apply highlight behind all recovered text

IF no original PDF available (text-only source):
  → Generate clean PDF from merged text
  → Apply highlights to recovered segments
  → Add note: "Reconstructed from extracted text — original layout not available"
```

### Highlight Colours

- **Yellow** — text recovered by cross-document merging (Stage 3)
- **Green** — text recovered by soft/overlay redaction removal (Stage 4)

### Update Behaviour

When a match group gains new members (new batch released, new doc indexed):
- Re-run Stage 3 for the group
- If `recovered_count` increases: regenerate and **overwrite** the PDF
- If `recovered_count` unchanged: skip, no new output
- DB records `previous_recovered_count` and `updated_at` for change history

### Footnote Page (always last page)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCES — Unobfuscator v1.0 — 2026-03-10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This document was reconstructed from the following sources:

[1] Document ID 12345 — DOJ VOL00008 — original_filename.pdf
    https://data.jmail.world/v1/documents/12345

[2] Document ID 14892 — DOJ VOL00009 — related_filename.pdf
    https://data.jmail.world/v1/documents/14892

Redactions recovered:       12
Soft redactions removed:     3
Recovery method: cross-document merge + soft redaction removal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Key Dependencies

- **Python 3.11+**
- `duckdb` — query Jmail Parquet files
- `datasketch` — MinHash and LSH
- `pymupdf` (fitz) — PDF reading, soft redaction removal, output generation
- `sqlite3` — built-in, no install needed
- `pyyaml` — config file parsing
- `httpx` — async HTTP for API calls and PDF downloads
- `rich` — CLI status display
