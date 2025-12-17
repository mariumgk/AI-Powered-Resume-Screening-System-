import re
from typing import Pattern


# Minimal, embedding-friendly preprocessing:
# We avoid lemmatization/stopword removal/tokenization because transformer sentence embeddings
# already capture semantics; heavy preprocessing can remove useful signals (e.g., skill names,
# product names, acronyms) and harm semantic similarity.


_RE_HTML_TAGS: Pattern[str] = re.compile(r"<[^>]+>")
_RE_EMAIL: Pattern[str] = re.compile(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_RE_URL: Pattern[str] = re.compile(r"\b(?:https?://|www\.)\S+\b")
_RE_PHONE: Pattern[str] = re.compile(
    r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]?\d{3,4})"
)


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _remove_repeated_headers(text: str) -> str:
    """Remove common repeated resume headers that can appear multiple times due to formatting.

    This is intentionally heuristic and minimal: we only remove exact duplicate header lines.
    """

    lines = [ln.strip() for ln in text.splitlines()]
    seen = set()
    out = []
    for ln in lines:
        key = ln.lower()
        if len(key) <= 60 and key in {
            "resume",
            "curriculum vitae",
            "cv",
            "page",
        }:
            continue

        # Drop exact duplicates of short header-like lines.
        if 0 < len(key) <= 40 and key in seen:
            continue

        if 0 < len(key) <= 40:
            seen.add(key)

        out.append(ln)

    return "\n".join(out)


def clean_text(raw_text: str) -> str:
    if raw_text is None:
        return ""

    text = str(raw_text)

    # Remove HTML if present (dataset can contain mixed artifacts).
    text = _RE_HTML_TAGS.sub(" ", text)

    # Remove high-noise PII-like patterns.
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_PHONE.sub(" ", text)

    text = _remove_repeated_headers(text)
    text = _collapse_whitespace(text)

    # Requirement: lowercase.
    text = text.lower()

    return text
