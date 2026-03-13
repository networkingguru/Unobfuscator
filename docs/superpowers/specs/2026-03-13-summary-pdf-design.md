# Summary PDF Report — Design Spec

## Overview

A single PDF generated from all recovered segments across output documents. Serves as an index into the corpus: what names, organizations, and identifiers were redacted, how often, and where to find them.

Generated to `{output_dir}/summary_report.pdf` (respects `output_dir` from `config.yaml`). Triggered via `./unob summary` CLI command. Not run automatically in the daemon cycle — on-demand only.

## Data Source

The `merge_results` table stores recovered segments as a JSON array in the `recovered_segments` column. Each element is an object:

```json
{"text": "SARAH KELLEN", "source_doc_id": "vol00009-efta00091556-pdf", "stage": "merge"}
```

**Query filter:** All rows where `recovered_count > 0` (regardless of `output_generated` flag). New `core/db.py` helper: `get_all_recovery_groups(conn)` returns `group_id, recovered_segments, source_doc_ids` for these rows.

**Link resolution:** For each group, the base document ID is `source_doc_ids[0]`. To resolve links:
- Local PDF path: `{output_dir}/{source}/{release_batch}/{doc_id}_merged.pdf` (via `build_output_path`)
- DOJ source URL: `documents.pdf_url` for the base doc ID

## Entity Categories

Heuristic regex extraction applied to each recovered segment's `text` field. No external dependencies. Extractors run in priority order — first match wins to avoid double-counting:

1. **Email Addresses** — Standard email regex. Deduplicate by lowercasing.
2. **Phone Numbers** — US formats: `(NNN) NNN-NNNN`, `NNN-NNN-NNNN`, `NNN.NNN.NNNN`, plus bare 10-digit sequences. Deduplicate by stripping non-digits.
3. **Case Numbers** — Patterns like `NN-XX-NNNNNN`, `NNNN-NNN`. Deduplicate exact.
4. **Organizations** — Phrases ending in known suffixes (LLC, Inc, Corp, Department, Agency, Bureau, Office, Court, FBI, DOJ). Case-insensitive suffix match. Deduplicate by lowercasing.
5. **People** — 2-4 consecutive capitalized words where: (a) at least 2 words, (b) not in a stopword/false-positive list (`The United`, `New York`, `This Report`, `United States`, common geographic/legal phrases), (c) not already captured by Organizations. Deduplicate case-insensitively, display the most common casing seen.
6. **Other** — Segments that yield no entities from categories 1-5. Displayed as truncated raw text (first 80 chars). Deduplicate exact.

A single segment may yield multiple entities across categories (e.g., an email containing both a name and an email address).

## PDF Structure

### Page 1 — Title & Stats

- "Unobfuscator Summary Report" + generation date
- Stats: total documents indexed (`COUNT(*) FROM documents`), match groups with recoveries, total recovered segments, recovery rate (`SUM(recovered_count) / SUM(total_redacted)` across all groups)

### Page 2 — Top Findings

- Top 20 entities across all categories, ranked by frequency
- Each row: entity name, category tag, count, first document reference
- Quick overview for someone who wants the headlines

### Pages 3+ — Category Sections

One section per non-empty category (in the order listed above):

- Section header with count of unique entities
- Table rows: entity, frequency count, document group IDs
- Each group ID displayed with clickable PDF link annotations (`fitz.Page.insert_link`) for both the local merged PDF path and the DOJ source URL
- Empty categories are omitted

## Data Flow

```
merge_results WHERE recovered_count > 0
        |
  Parse recovered_segments JSON -> list of {text, source_doc_id, group_id}
        |
  Strip block chars, skip junk (< 4 meaningful chars)
        |
  Run extractors in priority order -> { entity: {category, group_ids} }
        |
  Deduplicate per category (case/format normalization)
        |
  Sort by frequency within each category
        |
  Build PDF: title page -> top findings -> category sections
        |
  Save to: {output_dir}/summary_report.pdf
```

## Edge Cases

- Short junk recoveries (< 4 meaningful chars after stripping block chars): skip
- Duplicate entities with casing variation: normalize to the most common casing seen
- Segments with block characters (`█■`): strip before extraction
- Segments spanning multiple lines: split on newlines, extract from each line

## Implementation

- New file: `stages/summary_generator.py`
- New DB helper: `get_all_recovery_groups()` in `core/db.py`
- New CLI command: `./unob summary` in `unobfuscator.py`
- Uses PyMuPDF for PDF generation (existing dependency)
- No new dependencies

## Testing

New file: `tests/stages/test_summary_generator.py`

Key test scenarios:
- Empty corpus (no recoveries) — produces PDF with zero-count stats, no category sections
- Single recovery — appears in correct category
- Multi-category extraction from one segment
- Case-insensitive deduplication
- Phone/email normalization deduplication
- False-positive exclusion for People category (stopword list)
- Link resolution from group_id to file path and URL
