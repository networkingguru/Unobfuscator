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
Phase 4 — Merge & recover     Replace redaction markers with text recovered from the matching document
Phase 5 — PDF processing      Extract text hidden under black overlay rectangles (soft redactions)
Phase 6 — Output generation   Produce annotated PDFs with recovered text highlighted
```

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
```

---

## Installation

```bash
git clone https://github.com/networkingguru/Unobfuscator.git
cd Unobfuscator
python3 -m venv venv
source venv/bin/activate
pip install click rich datasketch duckdb pymupdf httpx pyyaml numpy
```

### Optional: download the local dataset archive

The daemon fetches document text directly from the Jmail API and PDFs from
justice.gov on demand. If you want to pre-cache the full PDF archive locally
(datasets 3–9 and 12, which are not on justice.gov), run:

```bash
python download_datasets.py
```

This downloads from [archive.org](https://archive.org) and a community mirror,
verifies checksums where available, and extracts files into `pdf_cache/`.
Disk usage varies by dataset; the script checks available space before each
download and aborts if less than 10% would remain free.

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
- A metadata page listing source documents, recovery counts, and links to originals

A cross-dataset entity summary (people, organizations, email addresses, phone
numbers, case numbers) can be generated with `python unobfuscator.py summary`.

---

## Configuration

Edit `config.yaml` to adjust paths, thresholds, and redaction marker patterns:

```yaml
output_dir: ./output
db_path: ./data/unobfuscator.db

matching:
  similarity_threshold: 0.70   # Jaccard similarity cutoff for LSH candidates
  min_overlap_chars: 200       # Minimum common text to confirm a match

polling:
  interval_minutes: 60
```

---

## TEREDACTA

A web UI for browsing and searching Unobfuscator results is in active development
as a companion project at `../TEREDACTA` (a sibling directory to this repo).
TEREDACTA provides a document viewer, entity browser, and recovery timeline.
It is not yet released publicly.

---

## Data Source

Document text and metadata are fetched read-only from the
[Jmail API](https://jmail.world), which hosts the Epstein archive as Parquet
files queryable via DuckDB. Original PDFs are fetched from justice.gov.

For datasets not hosted on justice.gov, Unobfuscator falls back to
[archive.org](https://archive.org) mirrors and community-hosted copies.
All non-DOJ sources are tracked in `pdf_cache/provenance.json` with
checksums and download timestamps.

---

## License

[MIT](LICENSE)
