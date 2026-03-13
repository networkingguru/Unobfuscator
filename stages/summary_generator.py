"""Summary report generator — extract entities from recovered segments and build a PDF index.

Logic reference: docs/superpowers/specs/2026-03-13-summary-pdf-design.md
"""

import re

# Stopwords for people extraction — common legal/geographic phrases
_PEOPLE_STOPWORDS = {
    "united states", "new york", "district court", "southern district",
    "northern district", "eastern district", "western district",
    "palm beach", "this report", "the united", "the honorable",
    "air france", "royal air", "american express", "flight information",
    "other information", "hotel information", "travel service",
    "privacy statement", "entry and exit",
}

_ORG_SUFFIXES = re.compile(
    r'\b(?:LLC|Inc|Corp|Department|Agency|Bureau|Office|Court|FBI|DOJ|OIG|CIA'
    r'|Police|Division|Commission|Authority|Association)\b',
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
_PHONE_RE = re.compile(
    r'(?:\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})'
    r'|\b\d{10}\b'
)
_CASE_RE = re.compile(r'\b\d{2,4}[-–][A-Z]{1,4}[-–]\d{4,}')
_NAME_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b|'
                       r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b')


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

    # 5. People names
    for m in _NAME_RE.finditer(text):
        name = m.group(1) or m.group(2)
        if not name or _is_consumed(m.start(), m.end()):
            continue
        # Check stopwords
        if name.lower() in _PEOPLE_STOPWORDS:
            continue
        # Skip single-word matches (regex requires 2+, but verify)
        if len(name.split()) < 2:
            continue
        entities.append({"text": name, "category": "people"})
        consumed.add((m.start(), m.end()))

    return entities


def _normalize_key(text: str, category: str) -> str:
    """Normalize entity text for deduplication."""
    if category == "email":
        return text.lower()
    if category == "phone":
        return re.sub(r'\D', '', text)
    if category == "case_number":
        return text  # exact match per spec
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
