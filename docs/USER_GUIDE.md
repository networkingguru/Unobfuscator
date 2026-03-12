# Unobfuscator — User Guide

## What it does

Unobfuscator cross-references documents from the Jmail/Epstein archive to recover
redacted text. It works by finding multiple versions of the same document where
one version has text that another has redacted, then merging them. It also detects
"soft redactions" — PDFs where text was visually hidden by an annotation rectangle
but never actually removed from the file.

Results are written as PDFs with recovered text highlighted.

---

## Running the daemon

The daemon runs all processing automatically in the background.

```bash
# Start
python unobfuscator.py start

# Check progress
python unobfuscator.py status

# Stop gracefully (finishes current stage before exiting)
python unobfuscator.py stop
```

You can close your terminal — the daemon keeps running as a background process.
When you come back, `status` will show current progress.

### What the status output means

| Field | What it counts |
|---|---|
| Stage 1 Indexer | Docs fetched and fingerprinted from Jmail |
| Stage 2 Matcher | MinHash fingerprints built (should match indexed) |
| Stage 3 Merger | Match groups where redaction recovery was attempted |
| Stage 4 PDF Processor | PDFs scanned for soft (annotation-only) redactions |
| Stage 5 Output | Output PDFs written to disk |
| Recovered redactions | Total redaction gaps filled across all documents |

---

## Targeted searches

The daemon processes everything automatically, but `search` lets you jump the queue
with higher priority. The daemon must be running to process queued searches.

```bash
# Search by person name (queries email participant fields)
python unobfuscator.py search --person "Ghislaine Maxwell"

# Search by keyword in document description
python unobfuscator.py search "flight logs"

# Target a specific release batch
python unobfuscator.py search --batch "DOJ-COURT"

# Process a specific document by ID
python unobfuscator.py search --doc "doj-24c5e0dc807d3b63.pdf"

# Block until results are ready (5-minute timeout), then show status
python unobfuscator.py search --person "Prince Andrew" --wait

# Write results to a custom directory
python unobfuscator.py search --person "Jeffrey Epstein" --output ./my-results
```

The `--person` flag searches the Jmail emails dataset for anyone appearing in any
email header field (From, To, CC, etc.).

---

## Where to find results

Output files are written to `./output/` by default, organized as:

```
output/{source}/{batch}/{doc_id}_merged.pdf
```

For example:
```
output/doj/DOJ-COURT/doj-abc123_merged.pdf
```

Each output PDF contains:
- The merged document text
- **Yellow highlights** — redactions recovered by cross-referencing another version
- **Green highlights** — soft redactions (text hidden by annotation but still in the file)
- A footnote page listing which source documents contributed recovered text,
  the recovered/total redaction counts, and the date generated

**A file is only written if at least one redaction was recovered.** Documents
that matched but had no complementary gaps are not written.

---

## Configuration

```bash
# View current config
python unobfuscator.py config show

# Change a value (dot notation supported)
python unobfuscator.py config set output_dir ./my-output
python unobfuscator.py config set matching.similarity_threshold 0.80
```

Key settings in `config.yaml`:

| Key | Default | Description |
|---|---|---|
| `output_dir` | `./output` | Where result PDFs are written |
| `matching.similarity_threshold` | `0.70` | Minimum Jaccard similarity to consider two docs a match |
| `matching.min_overlap_chars` | `200` | Minimum shared text length to confirm a match |
| `matching.email_header_min_matches` | `2` | Headers required to fast-path match email chains |
| `polling.interval_minutes` | `60` | How often to check Jmail for new release batches |
| `workers.pdf` | `2` | PDFs processed per cycle |

---

## Checking a specific document

```bash
python unobfuscator.py status --doc <document-id>
```

This prints the full database record for that document, including whether it has
been indexed, fingerprinted, PDF-processed, and which match group it belongs to.
