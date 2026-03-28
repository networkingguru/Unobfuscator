# Unobfuscator Pipeline — Plain English Logic

This file is the source of truth for what the tool does.
The Python code in `stages/` implements these steps exactly.

---

## Phase 0 — Email Chain Fast-Path

Run this BEFORE fingerprinting. Handles most of the email corpus instantly.

    FOR EACH document:
      Extract any email headers present (From:, To:, Date:, Subject: lines)

    BUILD an index: header_value → [list of doc_ids that contain it]

    FOR EACH header_value that appears in 2 or more documents:
      All those documents are confirmed matches.
      Group them immediately.
      Mark them as "phase0_matched" so they skip Phases 1–3.

---

## Phase 1 — Fingerprinting (for documents NOT matched in Phase 0)

    FOR EACH unmatched document:
      1. Clean the text:
         - Remove all redaction markers (from config.yaml redaction_markers list)
         - Normalize whitespace (collapse multiple spaces/newlines to single space)
         - Lowercase everything

      2. Shingling — slice into overlapping 8-word windows:
         e.g. "the meeting was held at mar a lago" produces:
           ["the meeting was held at mar a",
            "meeting was held at mar a lago"]

      3. MinHash — generate a 128-number signature:
         Use datasketch MinHashLSH with num_perm=128
         Each number represents the minimum hash of all shingles
         under one of 128 different hash functions.
         Documents with similar content will have similar signatures.

      4. Store the signature in document_fingerprints table.

---

## Phase 2 — Candidate Finding via LSH Banding

The similarity_threshold in config.yaml (default: 0.70) is the target
Jaccard similarity. The LSH parameters (bands/rows) are set to guarantee
that documents with ≥70% overlap will share at least one bucket with
>99% probability.

    USING datasketch MinHashLSH:
      Insert all fingerprints into the LSH index.

      FOR EACH document:
        Query the LSH index for its nearest neighbors.
        These are "candidates" — documents likely to overlap ≥70%.

    Candidates proceed to Phase 3 for verification.
    Non-candidates are skipped — assumed not the same document.

---

## Phase 3 — Verification and Grouping

    FOR EACH candidate pair (doc_A, doc_B):

      1. Strip redaction markers from both texts.
         Find the longest common substring between them.
         IF that substring is shorter than min_overlap_chars (from config):
           → REJECT this pair. Not the same document.

      2. Check for complementary redactions:
         Find all redaction marker positions in doc_A.
         Find all redaction marker positions in doc_B.
         IF doc_A has text where doc_B is redacted, OR vice versa:
           → CONFIRMED MATCH. Proceed to grouping.
         ELSE IF common text is long (>500 chars) but no complementary redactions:
           → Still group them (more versions = better coverage later).

      3. Assign to a match group:
         - IF doc_A already in a group → add doc_B to that group
         - IF doc_B already in a group → add doc_A to that group
         - IF both in different groups → merge the two groups (keep lower group_id)
         - IF neither in a group → create new group, add both

---

## Phase 4 — Merging (Stage 3)

    FOR EACH match group with 2 or more members:

      1. Pick the "base" document:
         - IF any member has 0 redaction markers AND similar text length
           (within 3x): use the least-redacted member as base (reverse merge).
           total_redacted is taken from the most-redacted member for reporting.
         - OTHERWISE: use the member with the most redaction markers.

      2. Multi-pass anchor matching (up to 3 passes, early exit on no progress):
         FOR EACH redaction marker remaining in the merged text:
           a. Extract anchors (left + right context around the marker).
           b. Try progressively wider anchors: 50, 100, 150, 200 chars.
           c. FOR EACH donor member, search for the anchor pair.
              IF found uniquely: extract text between anchors as recovery.
           d. Each pass uses the updated merged text (including prior recoveries)
              as context, enabling chain walking through dense redaction blocks.

      3. Alignment fallback (after each anchor pass):
         FOR EACH remaining unrecovered redaction:
           a. Use difflib.SequenceMatcher on lines to align base and donor.
           b. Read the aligned region from the donor as a candidate.
           c. Confirm via anchor context (both left AND right must match).
           d. Only accept if surrounding context validates the candidate.

      4. Build merged_text and store in merge_results.

---

## Phase 5 — PDF Soft Redaction Removal (Stage 4)

    FOR EACH document that has a PDF available:

      1. Open the PDF with PyMuPDF (fitz).

      2. FOR EACH page:
         Get all annotation objects on the page.
         FOR EACH annotation of type "Redact":
           Extract the text hidden under the annotation rectangle
           using page.get_textbox(annotation.rect).
           IF text found:
             → Soft redaction detected. Text was never truly removed.
             → Record the position and recovered text.

      3. IF any soft redactions found:
         Insert a new "merge" job for this document's match group
         so Stage 3 re-runs with the additional recovered text.

---

## Phase 6 — Output Generation (Stage 5)

    FOR EACH merge_result where recovered_count > 0
    AND output_generated = False (or recovered_count increased):

      1. Determine PDF source:
         IF original PDF available:
           Open it with PyMuPDF as the base.
         ELSE:
           Create a new blank PDF with fpdf2 or PyMuPDF.

      2. For each recovered segment:
         IF using original PDF:
           Find the text position on the page.
           Draw a yellow rectangle highlight behind it (cross-doc merge)
           OR green rectangle (soft redaction removal).
         ELSE (reconstructed PDF):
           Insert the merged_text as plain text.
           Apply yellow highlight to recovered segments.

      3. Add a final footnote page:
         List all source documents (ID, source, batch, filename, URL).
         Show recovered_count, soft redactions removed, date generated.

      4. Save to: output/{source}/{batch}/{doc_id}_merged.pdf
         Mark output_generated = True in merge_results.
         IF overwriting: update updated_at.
