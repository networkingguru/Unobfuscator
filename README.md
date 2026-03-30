# Unobfuscator

Unobfuscator cross-references copies of the same document from the Jeffrey Epstein
archive to recover text that was redacted in some copies but left visible in others.
The DOJ released the archive with inconsistent redaction across document versions —
Unobfuscator exploits that inconsistency to automatically fill in the gaps.

It runs as a background daemon, continuously pulling documents from the
[Jmail API](https://jmail.world), fingerprinting them, grouping near-duplicate
pairs, and producing annotated PDFs that show what was recovered and where it
came from.

> **Note:** The underlying data is public record released by the U.S. Department
> of Justice. Unobfuscator does not distribute any documents — it only processes
> them locally.

---

## How It Works

Processing happens in six phases that run automatically each daemon cycle:

```
Phase 0 — Email fast-path     Match documents sharing identical From/To/Date/Subject headers
Phase 1 — Fingerprinting      Build 128-bit MinHash signatures from 8-word shingles
Phase 2 — LSH candidates      Use locality-sensitive hashing to find near-duplicate pairs (≥70% Jaccard)
Phase 3 — Verification        Confirm pairs share substantial common text and complementary redactions
Phase 4 — Merge & recover     Multi-strategy redaction recovery (see below)
Phase 5 — PDF processing      Extract text hidden under black overlay rectangles (soft redactions)
Phase 5.5 — Text recovery     Backfill missing text from Jmail, extract PDF text layers, OCR
Phase 6 — Output generation   Produce annotated PDFs with recovered text highlighted
```

### Phase 4 — Merge Strategies

The merger uses four complementary techniques to maximize recovery while
maintaining zero false positives:

1. **Reverse merge** — when a group member has zero redactions (a fully
   unredacted copy), use it as the merge base instead of patching into
   the redacted version. Eliminates the anchor-matching bottleneck for
   the highest-value cases.

2. **Adaptive anchor widening** — if the default 50-character anchors
   are ambiguous (match multiple positions), automatically retry with
   100, 150, and 200-character anchors for disambiguation.

3. **Sequence alignment fallback** — for redactions that anchors can't
   reach (different page breaks, shifted content), align the base and
   donor texts line-by-line using `difflib.SequenceMatcher`, then
   confirm candidates via anchor context validation.

4. **Multi-pass chain walking** — up to 3 passes, where each pass uses
   prior recoveries as anchor context for adjacent redactions. Recovers
   dense redaction blocks where middle entries only become matchable
   after their neighbors are filled in.

### Quality Safeguards

- **Anchor uniqueness check** — rejects ambiguous matches where the
  anchor pair appears at multiple positions in the donor
- **Anchor quality floor** — skips redactions where combined anchor
  context has fewer than 8 alphanumeric characters
- **Content validation** — rejects recovered text that contains other
  redaction markers, block characters, generic placeholders, or AI
  image descriptions
- **Confidence flagging** — marks recoveries of the same text 3+ times
  as "questionable" (displayed as orange highlights in output PDFs)
- **Alignment confirmation** — sequence alignment candidates must pass
  both-sides anchor context validation with a quality floor

See [PIPELINE.md](PIPELINE.md) for the full plain-English logic reference and
[docs/USER_GUIDE.md](docs/USER_GUIDE.md) for detailed usage instructions.

---

## Requirements

- Python 3.11+
- ~5 GB free disk space for the database and PDF cache (significantly more if
  downloading the full dataset archive — see below)
- Network access to [jmail.world](https://jmail.world) (for document metadata)
  and [justice.gov](https://justice.gov) (for original PDFs)

### Python dependencies

```
click
rich
datasketch
duckdb
pymupdf
httpx
pyyaml
numpy
pillow
pytesseract
pandas
```

---

## Installation

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# macOS
# brew install tesseract

git clone https://github.com/networkingguru/Unobfuscator.git
cd Unobfuscator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** Tesseract is required for OCR of scanned documents (~4.6% of the corpus).
> Without it, the pipeline runs but those documents will have no text and cannot be matched.

### Optional: download the local dataset archive

The daemon fetches document text directly from the Jmail API and PDFs from
justice.gov on demand. If you want to pre-cache the PDF archive locally, run:

```bash
python download_datasets.py
```

This downloads datasets 3–12 from [archive.org](https://archive.org) and
community mirrors ([geeken.dev](https://justice.geeken.dev)), verifies
checksums where available, and extracts files into `pdf_cache/`.
Dataset 9 is excluded (community-reconstructed, cannot be verified against
the DOJ original). Disk usage varies by dataset (~112 GB total for DS10 + DS11);
the script checks available space before each download and aborts if less
than 10% would remain free.

---

## Quickstart

```bash
# Start the background daemon
python unobfuscator.py start

# Check progress
python unobfuscator.py status

# Follow the log
python unobfuscator.py log --follow

# Stop gracefully
python unobfuscator.py stop
```

The daemon polls for new release batches every 60 minutes (configurable).
On first run it will index all known documents before beginning matching —
this takes several hours depending on hardware.

---

## Tracking Progress

Some phases take a long time — here's what to expect and how to monitor them:

| Phase | Typical Duration | What's Happening |
|---|---|---|
| Phase 0 — Email fast-path | ~17 min | Scanning 1.4M docs and grouping by email headers |
| Phase 1 — Fingerprinting | varies | Building MinHash signatures (skipped if already in DB) |
| Phase 2 — LSH candidates | ~30 min | Loading 1.3M fingerprints and building the LSH index |
| **Phase 3 — Verification** | **2–3 days** | **The bottleneck: rolling-hash LCS on ~2M candidate pairs at ~10 pairs/sec** |
| Phase 5 — PDF processing | ~1 hr 15 min | Fetching and analyzing ~77K PDFs from justice.gov / archive.org |
| Phase 6 — Output generation | ~6 min | Writing annotated PDFs |

**Check what the daemon is doing right now:**

```bash
./unob status
```

This shows the current phase, how many documents have been processed, and
the daemon's most recent activity line.

**Watch progress in real time:**

```bash
./unob log -f
```

Each phase logs a counter as it works (e.g., `Phase 0: grouped 12,400 / 53,000 pairs`),
so you can see it advancing. Use `./unob log -n 50` to see the last 50 lines
without following.

> **Tip:** The first full run is the slowest. Subsequent daemon cycles only
> process new or changed documents and complete much faster.

---

## Commands

| Command | Description |
|---|---|
| `start [--foreground]` | Start the daemon (background by default) |
| `stop [--timeout N]` | Graceful stop, force-kill after N seconds |
| `status [--doc ID]` | Show pipeline progress; optionally inspect one document |
| `search QUERY` | Queue a targeted search at priority +100 |
| `search --person NAME` | Look up a person via Jmail and queue their documents |
| `search --batch ID` | Target a specific release batch |
| `search --wait` | Block until the daemon processes the queued job |
| `log [-n N] [-f]` | Show last N log lines; -f to follow |
| `config show` | Print current configuration |
| `config set KEY VALUE` | Set a configuration value (dot notation) |
| `summary` | Generate a summary PDF of all recovered entities |

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for full details and examples.

---

## Output

Results are written to `./output/` (configurable via `config.yaml`):

```
output/
  doj/
    DataSet11/
      <doc_id>_merged.pdf   ← annotated PDF with recovered text
  VOL00008/
    ...
```

Each output PDF contains:
- **Green highlights** — text recovered from another document version
- **Yellow highlights** — the corresponding source passage in the donor document
- **Orange highlights** — questionable recoveries (same text recovered 3+ times from different sources)
- A metadata page listing source documents, recovery counts, and links to originals

A cross-dataset entity summary (people, organizations, email addresses, phone
numbers, case numbers) can be generated with `python unobfuscator.py summary`.

---

## Configuration

Edit `config.yaml` to adjust paths, thresholds, and redaction marker patterns:

```yaml
output_dir: ./output
cache_dir: ./pdf_cache
db_path: ./data/unobfuscator.db

workers:
  pdf: 10                       # Parallel PDF processors
  pdf_batch_size: 1000

matching:
  similarity_threshold: 0.70    # Jaccard similarity cutoff for LSH candidates
  min_overlap_chars: 200        # Minimum common text to confirm a match
  email_header_min_matches: 2   # Headers needed to group email chains

polling:
  interval_minutes: 60

ocr:
  min_words_per_page: 50        # OCR quality filter

memory:
  limit_percent: 70             # RSS limit as % of total RAM
```

---

## TEREDACTA

[TEREDACTA](https://github.com/networkingguru/TEREDACTA) is a companion web UI
for browsing and searching Unobfuscator results. It provides a document viewer,
entity browser, and recovery timeline.

---

## Data Sources

**Primary:** Document text and metadata are fetched read-only from the
[Jmail API](https://jmail.world), which hosts the Epstein archive as Parquet
files queryable via DuckDB. Original PDFs are fetched from justice.gov.

**Fallback mirrors:** For datasets no longer hosted on justice.gov (bulk ZIP
downloads were removed in Feb 2026), Unobfuscator uses
[archive.org](https://archive.org) mirrors and
[geeken.dev](https://justice.geeken.dev) community-hosted copies.
All non-DOJ sources are tracked in `pdf_cache/provenance.json` with
checksums and download timestamps.

**Disaster recovery:** If the Jmail API is unavailable, the pipeline can
rebuild from local PDFs using text layer extraction and OCR (Phase 5.5).

---

## License

[MIT](LICENSE)
