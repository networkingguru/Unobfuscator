# Text Recovery Pipeline — Design Spec

**Date:** 2026-03-18
**Status:** Approved

## Problem

95,663 documents in the database have no `extracted_text`. Three root causes:

1. **Jmail shard mismatch (~37k docs):** The code constructs `documents-full/{batch_id}.parquet` per batch, but Jmail shards text across 5 files: `VOL00008.parquet`, `VOL00009.parquet`, `VOL00010.parquet`, `DataSet11.parquet`, and `other.parquet`. Batches like VOL00001, DOJ-COURT, all court case batches, VOL00008-2, and VOL00008-OFFICIAL-DOJ-LATEST have text in `other.parquet` or `VOL00008.parquet` that the code never queries. The 404 is silently swallowed.

2. **No OCR fallback (~6.3k docs with PDF URLs, ~57k after VOL00009 download):** Documents with downloadable PDFs but no text in Jmail — likely scanned or image-based PDFs. No extraction pipeline exists beyond soft-redaction detection.

3. **Image-only documents (~8.7k house_oversight JPGs):** Raw image files with no PDF URL and no text. Need OCR directly on the images.

Additionally, ~57k VOL00009 docs have no text, no PDF URL, and no local cache. These require downloading the dataset from archive.org before OCR is possible.

## Solution: Unified Text Recovery Stage (Phase 5.5)

A single new pipeline stage between PDF processing (Phase 5) and output generation (Phase 6) that handles all three problems.

### Step 1: Shard-Aware Jmail Backfill

#### Shard Mapping

Add to `core/api.py`:

```python
SHARD_MAP = {
    "VOL00008": "VOL00008",
    "VOL00008-2": "VOL00008",
    "VOL00008-OFFICIAL-DOJ-LATEST": "VOL00008",
    "VOL00009": "VOL00009",
    "VOL00010": "VOL00010",
    "DataSet11": "DataSet11",
}
DEFAULT_SHARD = "other"
```

`fetch_documents_text_batch()` looks up `SHARD_MAP.get(batch_id, DEFAULT_SHARD)` to determine the correct shard file, then filters by the original `batch_id` within that shard.

#### Backfill Logic

On daemon startup (and via `unobfuscator.py backfill` CLI command), detect documents where `text_processed=1` but `extracted_text` is empty and the batch exists in a known shard. Re-fetch text for those documents.

**Sequence of operations** (critical — must not go through `run_indexer_batch` which would re-fetch from the wrong shard):

1. Query the correct shard parquet for the batch's text
2. Call `update_extracted_text(doc_id, text, text_source='jmail')` to write text directly
3. Build the MinHash fingerprint immediately (call `build_fingerprint` from `indexer.py`)
4. Call `upsert_fingerprint()` to store it
5. Leave `text_processed=1` (fingerprint is already built — no need for Phase 1 to redo it)

This bypasses the indexer's `run_indexer_batch` entirely, avoiding the bug where re-indexing would re-fetch from the broken URL.

Also update `fetch_document_text()` (singular, line 51 of `api.py`) with the same shard resolution, or remove it if unused.

### Step 2: PyMuPDF Text Layer Extraction

For documents that still have no text after the Jmail fix — those with a PDF available (local cache at `pdf_cache/{batch_id}/{filename}` or downloadable via `pdf_url`):

1. Open PDF with PyMuPDF
2. Extract text via `page.get_text()` for each page
3. Calculate average words per page
4. If avg ≥ 50 words/page, accept the text layer — mark `text_source='pdf_text_layer'`
5. If avg < 50 words/page, proceed to OCR (Step 3)

The 50 words/page threshold is configurable via `config.yaml` (`ocr.min_words_per_page`). This is a conservative default — most text-based legal documents have 200+ words/page, while scanned PDFs with no text layer return 0-5.

### Step 3: OCR with Pixel Histogram Pre-Filter

For documents that failed the text layer threshold:

#### Page Classification (Pre-OCR)

Render each page to a grayscale image via PyMuPDF `page.get_pixmap(colorspace=fitz.csGRAY)`. Convert to numpy array via `numpy.frombuffer(pixmap.samples, dtype=numpy.uint8).reshape(pixmap.h, pixmap.w)`. Analyze the pixel histogram:

```python
pixels = numpy.frombuffer(pixmap.samples, dtype=numpy.uint8).reshape(pixmap.h, pixmap.w)
white_pct = (pixels > 230).mean()   # background
black_pct = (pixels < 25).mean()    # ink or redaction
mid_pct = 1 - white_pct - black_pct # gradients, photos

if black_pct > 0.90:
    tag = "redacted"   # heavily redacted, skip OCR
elif mid_pct > 0.40:
    tag = "photo"      # photograph, skip OCR
elif white_pct > 0.98:
    tag = "blank"      # empty/separator page, skip OCR
else:
    tag = "text"       # text on white background, OCR this page
```

#### OCR Execution

Only pages tagged `"text"` are OCR'd via `pytesseract.image_to_string()`. Per-page results are concatenated with page separators. Mark `text_source='ocr'`.

#### JPG/Image OCR

For house_oversight and other image-only documents (non-PDF), apply the same pixel pre-filter and OCR directly on the image file via Tesseract. Same classification logic applies.

### Step 4: Re-Integration with Pipeline

After text recovery stores `extracted_text` (via OCR or text layer extraction), build the MinHash fingerprint immediately within the text recovery stage itself:

1. Call `clean_text()` and `build_fingerprint()` from `indexer.py`
2. Call `upsert_fingerprint()` to store the signature
3. Mark `text_processed=1` (fingerprint is current)

This avoids routing back through `run_indexer_batch`, which only processes whole batches (not individual docs) and would re-fetch from Jmail. The fingerprint is built inline so the document is ready for matching on the next daemon cycle:

- Phases 2-3 find new LSH matches using the new fingerprint
- Phase 4 merges any new match groups
- The recovered text participates in unredaction naturally

Note: Newly downloaded PDFs from Phase 5 that complete during the same cycle as Phase 5.5 will not be OCR'd until the next daemon cycle. This is a latency issue, not a correctness issue.

## Database Changes

### New Columns on `documents`

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `text_source` | TEXT | NULL | Where text came from: `'jmail'`, `'pdf_text_layer'`, `'ocr'`, NULL (no text) |
| `ocr_processed` | BOOLEAN | 0 | Whether OCR/text-recovery has been attempted |
| `page_tags` | TEXT | NULL | JSON dict of per-page classifications: `{"0": "text", "1": "photo", ...}` |

### Migration

`core/db.py` adds columns via `ALTER TABLE ... ADD COLUMN` with try/except for SQLite compatibility. Non-destructive — existing data is unaffected. Additionally, set `text_source = 'jmail'` for all existing documents that already have non-empty `extracted_text` (one-time migration) so `text_source` NULL reliably means "no text."

### New DB Helpers

- `get_docs_needing_text_recovery()` — returns docs where `ocr_processed=0` and `extracted_text` is empty
- `get_docs_needing_backfill()` — returns docs where `text_processed=1`, `extracted_text` is empty, and batch is in a known shard
- `update_extracted_text(doc_id, text, text_source, page_tags=None)` — stores recovered text and metadata
- `mark_ocr_processed(doc_id)` — sets `ocr_processed=1`

## File Changes

| File | Change |
|------|--------|
| `core/api.py` | Add `SHARD_MAP`, `DEFAULT_SHARD`. Update both `fetch_documents_text_batch()` and `fetch_document_text()` to resolve shard before constructing URL. |
| `core/db.py` | Add 3 new columns with migration. Add new helper functions. |
| `stages/text_recovery.py` | **New file.** Phase 5.5: Jmail backfill → text layer extraction → OCR with pre-filter. |
| `unobfuscator.py` | Insert Phase 5.5 call in `_run_one_cycle()`. Add `backfill` CLI command. Extend `status` to show text recovery stats. Update idle-loop sleep guard to account for pending text recovery work (`ocr_processed=0` with empty text) so the daemon doesn't sleep while OCR work remains. |
| `requirements.txt` | Add `pytesseract`, `Pillow`. |
| `config.yaml` | Add `ocr.min_words_per_page: 50` setting. Falls back to hardcoded default of 50 if absent. |

## Local File Resolution Order

When text recovery needs a file for OCR:

1. `pdf_cache/{batch_id}/{original_filename}` — local cache (archive.org downloads)
2. `pdf_url` from DB — download from justice.gov/mirrors to cache
3. Not found → mark `ocr_processed=1`, leave `extracted_text` empty, log it

## Download Strategy for Missing PDFs

### VOL00009

Add to download pipeline — fetch `DataSet 9` zip from archive.org, extract to `pdf_cache/VOL00009/`. Same pattern as existing datasets 3-8, 12. Filenames match `original_filename` in the DB.

### House Oversight JPGs

Filenames like `HOUSE_OVERSIGHT_003954.jpg` have no `source_url`. **Out of scope for this spec** — a follow-up investigation will check jmail.world/drive and oversight.house.gov as potential download sources. The text recovery stage will attempt to process any house_oversight files found in `pdf_cache/` but will not attempt to download them. Missing files are logged and marked `ocr_processed=1`.

## Resumability

- **Jmail backfill:** Processes in batches per `release_batch`. Each doc committed individually. Interrupted runs skip already-backfilled docs.
- **OCR:** Queries `WHERE ocr_processed = 0`. Processes in batches of 100, commits after each doc. Interrupted runs resume at next unprocessed doc.
- **Failed OCR:** Marks `ocr_processed=1` with empty text — does not block other docs or retry endlessly.

## Dependencies

- **Python:** `pytesseract`, `Pillow` (pip) — pytesseract requires PIL Image objects as input
- **System:** `tesseract` binary (`brew install tesseract` on macOS)
- **Existing:** `pymupdf`, `numpy` (already in requirements.txt)

## Error Handling

- Tesseract not installed → Phase 5.5 logs a warning, skips OCR steps, Jmail backfill still runs
- Network error during Jmail fetch → leaves doc untouched for retry next cycle
- Corrupt PDF → marks `ocr_processed=1`, logs error, continues
- All-black/photo/blank pages → classified and skipped, tagged in `page_tags` for future use

## Page Tag Values

| Tag | Meaning | OCR'd? |
|-----|---------|--------|
| `"text"` | Text on white background | Yes |
| `"photo"` | Photograph (high mid-tone %) | No — tagged for future processing |
| `"redacted"` | >90% black pixels | No |
| `"blank"` | >98% white pixels | No |

## Pipeline Position

```
Phase 5  (pdf_processor)    — downloads PDFs, extracts soft redactions
Phase 5.5 (text_recovery)   — NEW: Jmail backfill + text layer + OCR
Phase 6  (output_generator) — generates annotated PDFs
```

## Expected Impact

| Source | Docs recovered | Method |
|--------|---------------|--------|
| Jmail shard fix | ~37,000 | Correct parquet shard query |
| PDF text layer | Unknown (subset of 6.3k) | PyMuPDF extraction |
| OCR (PDFs) | Unknown (remainder of 6.3k + 57k VOL00009) | Tesseract |
| OCR (JPGs) | Up to 8,715 | Tesseract on images |

All recovered text feeds back into the indexer for fingerprinting and participates in unredaction matching.
