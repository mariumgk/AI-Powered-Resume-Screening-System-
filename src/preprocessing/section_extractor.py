import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ResumeSections:
    summary: str
    skills: str
    experience: str
    full_text: str


_SECTION_HEADERS = {
    "summary": ["summary", "professional summary", "profile", "objective"],
    "skills": ["skills", "technical skills", "core skills", "key skills"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
    ],
}


def _find_header_positions(text: str) -> List[Tuple[int, str]]:
    """Return ordered list of (start_index, section_name) for detected headers."""

    positions: List[Tuple[int, str]] = []
    for section, variants in _SECTION_HEADERS.items():
        for v in variants:
            # Match header at line start, allowing punctuation like ':'
            pat = re.compile(rf"(^|\n)\s*{re.escape(v)}\s*[:\-]?\s*(\n|$)")
            for m in pat.finditer(text):
                positions.append((m.start(), section))

    positions.sort(key=lambda x: x[0])

    # Deduplicate close duplicates (same section found multiple times near each other)
    filtered: List[Tuple[int, str]] = []
    for pos, sec in positions:
        if filtered and filtered[-1][1] == sec and abs(filtered[-1][0] - pos) < 50:
            continue
        filtered.append((pos, sec))

    return filtered


def extract_sections(cleaned_text: str) -> ResumeSections:
    """Extract key sections.

    Section awareness often improves semantic signal by concentrating the most informative
    parts (skills/experience/summary) and reducing noise from boilerplate.

    Fallback rule (mandatory): if sections are not detected, return empty sections and keep full text.
    """

    text = cleaned_text or ""
    positions = _find_header_positions(text)

    if not positions:
        return ResumeSections(summary="", skills="", experience="", full_text=text)

    # Build spans between headers
    spans: Dict[str, str] = {"summary": "", "skills": "", "experience": ""}
    for i, (start, sec) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunk = text[start:end].strip()

        # Remove the header line itself
        chunk = re.sub(r"^\s*([a-z ]{3,40})\s*[:\-]?\s*\n", "", chunk)
        chunk = chunk.strip()

        # Keep the longest chunk if section repeats
        if len(chunk) > len(spans.get(sec, "")):
            spans[sec] = chunk

    return ResumeSections(
        summary=spans.get("summary", ""),
        skills=spans.get("skills", ""),
        experience=spans.get("experience", ""),
        full_text=text,
    )


def build_embedding_text(sections: ResumeSections) -> str:
    """Build a single document string used for embeddings.

    We prefer sections when present; otherwise we fall back to full text.
    """

    parts = []
    if sections.summary.strip():
        parts.append("summary: " + sections.summary.strip())
    if sections.skills.strip():
        parts.append("skills: " + sections.skills.strip())
    if sections.experience.strip():
        parts.append("experience: " + sections.experience.strip())

    combined = "\n".join(parts).strip()
    if combined:
        return combined

    return (sections.full_text or "").strip()
