"""Summary report generator — extract entities from recovered segments and build a PDF index.

Logic reference: docs/superpowers/specs/2026-03-13-summary-pdf-design.md
"""

import json
import re
from datetime import date
from pathlib import Path

import fitz

from core.db import get_all_recovery_groups, get_documents_by_ids

# Stopwords for people extraction — common legal/geographic/email phrases
_PEOPLE_STOPWORDS = {
    # Geographic
    "united states", "new york", "palm beach", "west palm beach",
    "south florida", "north florida", "virgin islands", "salt lake city",
    "fort lauderdale", "phoenix arizona", "san francisco",
    "lexington avenue", "park avenue", "wilson blvd", "wall st",
    "putnam avenue", "foley square", "federal plaza", "plaza new",
    "mountain view", "little st", "boca raton",
    # Legal/court
    "district court", "southern district", "northern district",
    "eastern district", "western district", "judicial circuit",
    "this report", "the united", "the honorable",
    "certificate of service", "trust agreement", "dispositive provisions",
    "related touhy requests", "deposition attachment",
    "grand jury", "grand jury subpoena", "fifth amendment",
    "special master", "in trust", "the trustees", "the grantor",
    "the government", "the jeffrey", "case no", "case name",
    "counsel list", "composite exhibit", "incident report",
    "plaintiff bradley edwards", "discovery issues", "motion for reconsideration",
    "epstein depo", "epstein evidence", "defendant bradley",
    "defendant epstein", "financial condition", "financial condition report",
    "materials reviewed", "independent accountants", "special housing unit",
    "mandatory copies to",
    # Travel/business
    "air france", "royal air", "american express", "american express travel",
    "flight information", "other information", "hotel information",
    "travel service", "centurion travel service", "privacy statement",
    "entry and exit", "economy class", "estimated time",
    "ticket number", "detail sheet", "airline record locator",
    "best regards", "original message",
    "agent details", "travel arrangements", "travel details",
    "key information", "ticket amount", "ticket base fare", "ticket date",
    "ticket information", "toll free", "departure terminal",
    "arrival terminal", "check digit", "airline name",
    "airline imposed fees", "airline baggage fee", "airline baggage",
    "record locator", "exit information", "service provider",
    "video conference", "when overseas call collect",
    "digital itinerary", "total charged",
    # Email date headers (On Mon, On Tue, etc.)
    "on jan", "on feb", "on mar", "on apr", "on may", "on jun",
    "on jul", "on aug", "on sep", "on oct", "on nov", "on dec",
    "on mon", "on tue", "on wed", "on thu", "on fri", "on sat", "on sun",
    # Time zones / email artifacts
    "pm edt", "am edt", "pm est", "am est", "pm pst", "am pst",
    # Email metadata patterns
    "external email", "this email", "via email", "do not reply to",
    "do not forward", "good morning",
    # Non-name phrases / generic terms
    "of the", "sec ig", "pwm aml", "sexual abuse", "third request",
    "internal revenue code", "children by epstein", "cares act",
    "global lock", "product team", "little red", "origins project",
    "cosmology initiative", "space exploration",
    "limited liability corporations", "civil litigation alert",
    "approved by", "drafted by", "prepared by",
    "balance due", "sales tax",
    # Technical/data artifacts
    "incl ticketno", "inquiry regarding", "amlcompliance inquiries",
    "sec owb", "knight capital", "danya friedman on behalf",
    "for palm beach county", "judicial circuit in and",
    "behrend dr ste", "australian ave",
    # Common non-person multi-word phrases
    "task force", "pony express", "gen atlantic",
    "no meal", "in august", "not august", "and august",
    "pro hac", "chair clayton",
    "hong kong", "las vegas", "rhode island", "la guardia",
    "coast guard", "bear stearns", "home depot", "white house",
    "first class", "kind regards",
    "sent date", "sent tue", "sent wed", "sent thu", "sent fri",
    "sent mon", "sent sat", "sent sun",
    "of course", "in april", "in july",
    "mobile number", "direct phone",
    "virginia beach", "new year", "west side", "ft lauderdale",
    "fifth avenue", "one st", "sullivan cromwell",
    "the ambassador", "the sept", "the times",
    "the dancing", "to boston", "for epstein", "as david",
    "my debts", "or ambaji",
    "of amlcompliance", "amex centurian", "rim waterfront",
    "phonetic sp", "loi informatique", "nurik one",
    "little dick", "attorney marie", "krauss director",
    "regulation cmte", "legat canberra",
    "dr ste", "dr. ste", "berger montague",
    "howland evangelista", "kohlenberg burnett",
    "tawaraya ryokan",
}

_ORG_SUFFIXES = re.compile(
    r'\b(?:LLC|Inc|Corp|Department|Agency|Bureau|Office|Court|FBI|DOJ|OIG|CIA'
    r'|Police|Division|Commission|Authority|Association|Bank|Trust|Capital'
    r'|Securities|Group|Partners|Services|Foundation)\b',
    re.IGNORECASE,
)

# Email metadata suffixes — "Lesley Groff Sent", "Name Cc", etc.
_EMAIL_META_SUFFIXES = {"sent", "cc", "from", "to", "subject", "bcc"}

# Greeting prefixes — "Dear Jeffrey", "Hi Lesley", etc.
_GREETING_PREFIXES = {"dear", "hi", "hello", "hey"}

# Words that disqualify a candidate from being a person's name.
# If ANY word (lowercased) in the candidate appears here, it's not a person.
_NON_NAME_WORDS = {
    # Transport
    "airlines", "airways", "airline", "airport",
    # Business/org (not caught by _ORG_SUFFIXES)
    "corporation", "university", "college", "company", "laboratory",
    "warehouse", "maritime", "manager", "relationship", "centurion",
    # Objects / technical
    "battery", "filter", "assy", "harness", "plate", "deck", "fuel",
    "cycle", "jerky", "leak", "roof",
    # Document / legal / formal
    "statement", "disclosure", "report", "exhibit", "document",
    "amendment", "certificate", "provision", "motion", "dissenting",
    "liability", "appearance", "obligation", "address", "memorandum",
    "investigation", "reconsideration", "obstruction", "justice",
    "burden", "reducing", "command", "rider", "attached",
    "offices", "invoice",
    # Email artifacts
    "forwardedmessage", "message", "messages", "censored", "transmission",
    # Actions / descriptors
    "imposed", "carry", "dated", "prior", "provide", "electronic",
    "format", "possible", "additional", "blacked", "sweep",
    "exploitation", "trafficking", "reporting", "visiting",
    # Transport / travel
    "air", "lines", "shuttle", "flights", "domestic", "baggage",
    "tickets", "meal", "customer", "terminal", "class",
    # Business titles / roles
    "president", "vice", "agent", "warden", "commissioner",
    "counsel", "associate", "assistant", "reviewer", "preparer",
    "special", "inspector", "general",
    # Greetings / closings
    "thank", "thanks", "regards", "welcome",
    # Common English words unlikely in names
    "acrobat", "reader", "parkway", "plaza", "grantor", "otcm",
    "international", "groupe", "building", "wednesday",
    "reg", "hac", "junk", "hell", "entry", "cheese", "chip",
    "tartar", "train", "depot", "project", "management",
    "capital", "holdings", "stearns", "asset", "banking",
    "specific", "rights", "reserved", "transfers", "total",
    "fare", "wealth", "unknown", "female", "male",
    "separation", "settlement", "kingdom", "republic",
    "complaints", "correspondence", "complaint", "draft",
    "section", "title", "description", "purpose", "context",
    "recipient", "sender", "number", "amount", "received",
    "distribution", "limited", "prohibited", "strictly",
    "passport", "valid",
    # Places / geography
    "avenue", "street", "blvd", "boulevard", "road", "drive",
    "beach", "centre", "center", "correctional", "metropolitan",
    "harbor", "switzerland", "intl",
    # Business / finance
    "broker", "dealer", "estate", "trust", "trusts", "nationals",
    "eagle", "bureau", "compagnie", "banque", "industries",
    "industrie", "graphics", "cloudcommerce",
    # Misc non-name words
    "crimes", "children", "debts", "documentation", "derivative",
    "depo", "notice", "confidentiality", "unit", "housing",
    "mirror", "nickel", "russian", "castle", "begin",
    "forwarded", "messenger", "roaming", "voyager",
    "arriving", "departing", "arrival", "mins",
    "ambassador", "correctional", "sales", "research",
}

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
_PHONE_RE = re.compile(
    r'(?:\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})'
    r'|\b\d{10}\b'
)
_CASE_RE = re.compile(r'\b\d{2,4}[-–][A-Z]{1,4}[-–]\d{4,}')
_NAME_RE = re.compile(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b')

# Title + name pattern: Mr./Ms./Mrs./Dr./Senator/Judge/etc. followed by a name
_TITLE_NAME_RE = re.compile(
    r'\b((?:Mr|Mrs|Ms|Dr|Prof|Judge|Senator|Rep|Col|Sgt|Det|Atty|Amb)'
    r'\.?\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b'
)

# Common short English words that appear in ALL CAPS — not names
_COMMON_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "or",
    "and", "not", "do", "is", "it", "be", "as", "if", "so", "no", "my",
    "we", "us", "up", "all", "but", "are", "was", "has", "had", "may",
    "can", "new", "old", "out", "how", "its", "our", "any", "per",
    "see", "mr", "ms", "st", "vs", "re",
}

# Words that can't start a person name (common adjectives, prepositions, etc.)
_NON_FIRST_WORDS = {
    "after", "again", "all", "another", "any", "approved", "as",
    "background", "bad", "because", "before", "best", "both",
    "call", "california", "campaign", "chase", "co", "columbia",
    "commercial", "commissioners", "compact", "confidential", "confirmed", "current",
    "destination", "deutsche", "direct", "during",
    "enforcement", "estimated", "every",
    "fairmont", "final", "first", "flight", "following", "for",
    "former", "four", "from",
    "global", "government", "grand", "great", "green", "gross",
    "hotel", "how",
    "in", "including", "institutional", "integrity", "internal",
    "island",
    "let", "limited",
    "mandatory", "medical", "metropolitan", "mins", "mobile", "more",
    "mutual",
    "net", "newark", "noble", "normal", "north", "not",
    "on", "one", "operated", "origin", "other", "our", "ownership",
    "passed", "plaintiff", "place", "please", "priority", "pvt",
    "rated", "regional", "related", "renaissance", "requested",
    "requests", "reviewed",
    "salt", "scan", "seats", "second", "see", "silver", "south",
    "states", "structure", "subpoena",
    "technology", "the", "this", "thx", "ticket", "time", "to",
    "travel", "treasury", "tuesday",
    "ultimate", "urban",
    "washington", "west", "when", "where", "would", "york", "young",
}

# Words that can't end a person name
_NON_LAST_WORDS = {
    "on", "do", "let", "if", "the", "by", "of", "in", "at",
    "this", "that", "new", "all", "pro", "ex",
    "case", "date", "dir", "dr", "st", "jr", "sr",
    "state", "record", "base", "palm", "lake", "place",
    "collect", "begin", "significant", "would",
    "attn", "tech", "bank", "crom",
    "school", "hotel", "dec", "bay", "amb",
    "boston", "jet", "behalf",
    "national", "magna", "nasional", "director",
    "cmte", "handler", "informatique", "centurian",
    "waterfront", "sp", "de", "le", "la",
}


def extract_entities(text: str) -> list[dict]:
    """Extract categorized entities from a recovered text segment.

    Returns list of {"text": str, "category": str} dicts.
    Extractors run in priority order; matched spans are excluded from later extractors.
    Multi-line segments are split and each line is processed independently.
    """
    # Strip block chars
    text = text.replace("█", " ").replace("■", " ")

    # Split on newlines and process each line to avoid cross-line false matches
    all_entities = []
    for line in text.split("\n"):
        line = re.sub(r'\s+', ' ', line).strip()
        if len(line) < 4:
            continue
        all_entities.extend(_extract_from_line(line))
    return all_entities


def _extract_from_line(text: str) -> list[dict]:
    """Extract entities from a single line of text."""
    entities = []
    consumed: set[tuple[int, int]] = set()

    def _is_consumed(start: int, end: int) -> bool:
        for cs, ce in consumed:
            if start < ce and end > cs:
                return True
        return False

    # 1. Emails
    for m in _EMAIL_RE.finditer(text):
        if not _is_consumed(m.start(), m.end()):
            entities.append({"text": m.group(), "category": "email"})
            consumed.add((m.start(), m.end()))

    # 2. Phones
    for m in _PHONE_RE.finditer(text):
        if not _is_consumed(m.start(), m.end()):
            entities.append({"text": m.group(), "category": "phone"})
            consumed.add((m.start(), m.end()))

    # 3. Case numbers
    for m in _CASE_RE.finditer(text):
        if not _is_consumed(m.start(), m.end()):
            entities.append({"text": m.group(), "category": "case_number"})
            consumed.add((m.start(), m.end()))

    # 4. Organizations (suffix match)
    for m in _ORG_SUFFIXES.finditer(text):
        # Build org name: look backward for preceding words and forward for "of X" phrases
        prefix_start = max(0, m.start() - 60)
        prefix = text[prefix_start:m.end()]
        # Find capitalized words leading into the suffix
        org_match = re.search(
            r'(?:(?:the|of|for|and|in)\s+)*(?:[A-Z][a-zA-Z]*\s+)*' + re.escape(m.group()),
            prefix, re.IGNORECASE
        )
        if not org_match:
            continue
        org_name = org_match.group().strip()
        abs_start = prefix_start + org_match.start()
        abs_end = prefix_start + org_match.end()
        # Extend forward: grab trailing "of X", "for X" phrases
        tail = text[abs_end:]
        tail_match = re.match(r'(?:\s+(?:of|for|and|in|the)\s+[A-Z][a-zA-Z]*)+', tail)
        if tail_match:
            org_name += tail_match.group()
            abs_end += tail_match.end()
        # Strip leading articles
        org_name = re.sub(r'^(?:the|of|for)\s+', '', org_name, flags=re.IGNORECASE).strip()
        if len(org_name) > 3 and not _is_consumed(abs_start, abs_end):
            entities.append({"text": org_name, "category": "organization"})
            consumed.add((abs_start, abs_end))

    # 5a. Title-prefixed names (Mr., Ms., Dr., Senator, etc.)
    for m in _TITLE_NAME_RE.finditer(text):
        name = m.group(1)
        if not name or _is_consumed(m.start(), m.end()):
            continue
        # Check stopwords on the name part (after stripping title)
        name_part = _TITLE_STRIP_RE.sub('', name).strip()
        if not name_part:
            continue
        name_lower = name.lower()
        if name_lower in _PEOPLE_STOPWORDS:
            continue
        if any(sw in name_lower for sw in _PEOPLE_STOPWORDS if len(sw) > 5):
            continue
        entities.append({"text": name, "category": "people"})
        consumed.add((m.start(), m.end()))

    # 5b. People names
    for m in _NAME_RE.finditer(text):
        name = m.group(1)
        if not name or _is_consumed(m.start(), m.end()):
            continue
        words = name.split()
        # Skip single-word matches (regex requires 2+, but verify)
        if len(words) < 2:
            continue
        name_lower = name.lower()
        words_lower = [w.lower() for w in words]
        # Check stopwords — exact match or any stopword is a substring
        if name_lower in _PEOPLE_STOPWORDS:
            continue
        if any(sw in name_lower for sw in _PEOPLE_STOPWORDS if len(sw) > 5):
            continue
        # Reject if last word is an email metadata suffix
        if words_lower[-1] in _EMAIL_META_SUFFIXES:
            continue
        # Reject if contains @ (email fragment)
        if "@" in name:
            continue
        # Reject if first word is a greeting or non-name word
        if words_lower[0] in _GREETING_PREFIXES:
            continue
        if words_lower[0] in _NON_FIRST_WORDS:
            continue
        # Reject if last word is a non-name word
        if words_lower[-1] in _NON_LAST_WORDS:
            continue
        # Reject if any word is a known non-name word
        if any(w in _NON_NAME_WORDS for w in words_lower):
            continue
        # Reject if any word is too short (1 char) — likely an initial fragment
        if any(len(w) < 2 for w in words):
            continue
        entities.append({"text": name, "category": "people"})
        consumed.add((m.start(), m.end()))

    return entities


_TITLE_STRIP_RE = re.compile(
    r'^(?:Mr|Mrs|Ms|Dr|Prof|Judge|Senator|Rep|Col|Sgt|Det|Atty|Amb)'
    r'\.?\s+', re.IGNORECASE
)


def _normalize_key(text: str, category: str) -> str:
    """Normalize entity text for deduplication."""
    if category == "email":
        return text.lower()
    if category == "phone":
        return re.sub(r'\D', '', text)
    if category == "case_number":
        return text  # exact match per spec
    if category == "people":
        # Strip title prefix for dedup so "Mr. Kamensky" merges with "Kamensky"
        text = _TITLE_STRIP_RE.sub('', text)
    return text.lower()


def aggregate_entities(raw_entities: list[dict]) -> list[dict]:
    """Deduplicate and count entities. Returns sorted list per category.

    Input: list of {"text": str, "category": str, "group_id": int}
    Output: list of {"text": str, "category": str, "count": int, "group_ids": list[int]}
    """
    buckets: dict[str, dict] = {}  # key -> {text, category, count, group_ids, casing_counts}

    for e in raw_entities:
        key = f"{e['category']}:{_normalize_key(e['text'], e['category'])}"
        if key not in buckets:
            buckets[key] = {
                "text": e["text"],
                "category": e["category"],
                "count": 0,
                "group_ids": [],
                "casing_counts": {},
            }
        bucket = buckets[key]
        bucket["count"] += 1
        if e["group_id"] not in bucket["group_ids"]:
            bucket["group_ids"].append(e["group_id"])
        # Track casing to pick the most common form
        bucket["casing_counts"][e["text"]] = bucket["casing_counts"].get(e["text"], 0) + 1

    result = []
    for bucket in buckets.values():
        # Use the most common casing
        best_text = max(bucket["casing_counts"], key=bucket["casing_counts"].get)
        result.append({
            "text": best_text,
            "category": bucket["category"],
            "count": bucket["count"],
            "group_ids": sorted(bucket["group_ids"]),
        })

    # Sort by count descending within each category
    result.sort(key=lambda e: (-e["count"], e["text"]))
    return result


# Category display order and labels
_CATEGORY_ORDER = [
    ("people", "People"),
    ("email", "Email Addresses"),
    ("phone", "Phone Numbers"),
    ("case_number", "Case Numbers"),
    ("organization", "Organizations"),
    ("other", "Other Recoveries"),
]


def _collect_raw_entities(conn) -> list[dict]:
    """Parse all recovered segments and extract entities with group context.

    Segments that yield no categorized entities are added as "other" category
    (truncated to 80 chars).
    """
    groups = get_all_recovery_groups(conn)
    raw = []
    for g in groups:
        segments = json.loads(g["recovered_segments"] or "[]")
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text or len(text) < 4:
                continue
            entities = extract_entities(text)
            if entities:
                for e in entities:
                    e["group_id"] = g["group_id"]
                    raw.append(e)
            else:
                # "Other" category — truncated raw text
                display = text.replace("\n", " ")[:80]
                raw.append({
                    "text": display,
                    "category": "other",
                    "group_id": g["group_id"],
                })
    return raw


def _write_title_page(pdf: fitz.Document, conn, entity_count: int,
                      segment_count: int) -> None:
    """Write page 1: title and high-level stats."""
    page = pdf.new_page()
    margin = 40
    y = margin
    fontsize = 16

    page.insert_text((margin, y + fontsize), "Unobfuscator Summary Report",
                     fontsize=fontsize)
    y += fontsize * 2

    fontsize = 10
    line_height = fontsize * 1.8
    today = date.today().isoformat()

    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    total_groups = conn.execute(
        "SELECT COUNT(*) FROM merge_results WHERE recovered_count > 0"
    ).fetchone()[0]
    recovered_sum = conn.execute(
        "SELECT COALESCE(SUM(recovered_count), 0) FROM merge_results"
    ).fetchone()[0]
    redacted_sum = conn.execute(
        "SELECT COALESCE(SUM(total_redacted), 0) FROM merge_results"
    ).fetchone()[0]
    rate = f"{recovered_sum/redacted_sum:.1%}" if redacted_sum > 0 else "N/A"

    stats = [
        f"Generated: {today}",
        f"Documents indexed: {total_docs:,}",
        f"Match groups with recoveries: {total_groups:,}",
        f"Total recovered segments: {segment_count:,}",
        f"Unique entities extracted: {entity_count:,}",
        f"Recovery rate: {rate} ({recovered_sum:,} / {redacted_sum:,} redactions)",
    ]
    page.insert_text((margin, y + fontsize), "=" * 60, fontsize=9)
    y += line_height
    for line in stats:
        page.insert_text((margin, y + fontsize), line, fontsize=fontsize)
        y += line_height
    page.insert_text((margin, y + fontsize), "=" * 60, fontsize=9)


def _write_top_findings(pdf: fitz.Document, entities: list[dict]) -> None:
    """Write page 2: top 20 entities across all categories."""
    page = pdf.new_page()
    margin = 40
    fontsize = 12
    y = margin

    page.insert_text((margin, y + fontsize), "Top Findings", fontsize=fontsize)
    y += fontsize * 2

    fontsize = 10
    line_height = fontsize * 1.5
    page.insert_text((margin, y + fontsize), "=" * 60, fontsize=9)
    y += line_height

    top = entities[:20]
    for i, e in enumerate(top, 1):
        label = f"{i:>2}. [{e['category'].upper()}] {e['text']} — {e['count']}x"
        if y + line_height > page.rect.height - 60:
            page = pdf.new_page()
            y = margin
        page.insert_text((margin, y + fontsize), label, fontsize=fontsize)
        y += line_height


def _write_category_section(pdf: fitz.Document, label: str,
                            entities: list[dict], conn,
                            output_dir: str) -> None:
    """Write a category section with entity table and links."""
    page = pdf.new_page()
    margin = 40
    fontsize = 12
    y = margin

    page.insert_text((margin, y + fontsize),
                     f"{label} ({len(entities)} unique)", fontsize=fontsize)
    y += fontsize * 2

    fontsize = 9
    line_height = fontsize * 1.6

    for e in entities:
        # Entity line
        group_str = ", ".join(str(g) for g in e["group_ids"][:10])
        if len(e["group_ids"]) > 10:
            group_str += f" (+{len(e['group_ids']) - 10} more)"
        line = f"{e['text']}  —  {e['count']}x  —  groups: {group_str}"

        if y + line_height * 2 > page.rect.height - 60:
            page = pdf.new_page()
            y = margin

        page.insert_text((margin, y + fontsize), line[:100], fontsize=fontsize)
        y += line_height

        # Links for first 3 groups
        for gid in e["group_ids"][:3]:
            source_doc_ids = _get_source_doc_ids(conn, gid)
            if source_doc_ids:
                base_id = source_doc_ids[0]
                docs = get_documents_by_ids(conn, [base_id])
                if docs:
                    d = docs[0]
                    local_path = str(
                        Path(output_dir).resolve() / d.get("source", "unknown")
                        / d.get("release_batch", "unknown")
                        / f"{base_id}_merged.pdf"
                    )
                    url = d.get("pdf_url") or ""
                    if y + line_height * 2 > page.rect.height - 60:
                        page = pdf.new_page()
                        y = margin
                    # Local file link (use file:// URI for PDF viewer compat)
                    local_label = f"    Local: {local_path}"
                    page.insert_text((margin + 10, y + fontsize),
                                     local_label[:90], fontsize=fontsize)
                    local_width = fitz.get_text_length(local_label[:90], fontsize=fontsize)
                    file_uri = Path(local_path).as_uri()
                    page.insert_link({
                        "kind": fitz.LINK_URI,
                        "from": fitz.Rect(margin + 10, y,
                                          margin + 10 + local_width, y + fontsize + 2),
                        "uri": file_uri,
                    })
                    y += line_height
                    # DOJ source URL link
                    if url:
                        url_label = f"    Source: {url}"
                        page.insert_text((margin + 10, y + fontsize),
                                         url_label[:90], fontsize=fontsize)
                        url_width = fitz.get_text_length(url_label[:90], fontsize=fontsize)
                        page.insert_link({
                            "kind": fitz.LINK_URI,
                            "from": fitz.Rect(margin + 10, y,
                                              margin + 10 + url_width, y + fontsize + 2),
                            "uri": url,
                        })
                        y += line_height

        y += line_height * 0.5  # spacing between entities


def _get_source_doc_ids(conn, group_id: int) -> list[str]:
    """Get source_doc_ids for a group from merge_results."""
    row = conn.execute(
        "SELECT source_doc_ids FROM merge_results WHERE group_id = ?",
        (group_id,)
    ).fetchone()
    if row and row["source_doc_ids"]:
        return json.loads(row["source_doc_ids"])
    return []


def generate_summary_pdf(conn, output_dir: str) -> str:
    """Generate the summary report PDF. Returns the output path."""
    raw = _collect_raw_entities(conn)
    aggregated = aggregate_entities(raw)

    # Count total segments
    groups = get_all_recovery_groups(conn)
    segment_count = sum(
        len(json.loads(g["recovered_segments"] or "[]")) for g in groups
    )

    output_path = str(Path(output_dir) / "summary_report.pdf")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open()

    # Page 1: Title and stats
    _write_title_page(pdf, conn, len(aggregated), segment_count)

    # Page 2: Top findings
    _write_top_findings(pdf, aggregated)

    # Category sections
    for cat_key, cat_label in _CATEGORY_ORDER:
        cat_entities = [e for e in aggregated if e["category"] == cat_key]
        if cat_entities:
            _write_category_section(pdf, cat_label, cat_entities, conn, output_dir)

    pdf.save(output_path)
    pdf.close()
    return output_path
