from typing import Dict

from src.preprocessing.cleaner import clean_text
from src.preprocessing.section_extractor import ResumeSections


def _cap_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text[:max_chars].strip()


def build_chunks(sections: ResumeSections) -> Dict[str, str]:
    """Build cleaned, capped chunks for chunk-based embeddings.

    Rules:
        - summary/experience/skills are cleaned independently
        - skills are capped to avoid overpowering context
        - experience prioritizes recent content by taking the leading part
        - if all sections are empty, fall back to full_text as experience
    """

    summary = clean_text(sections.summary or "")
    skills = clean_text(sections.skills or "")
    experience = clean_text(sections.experience or "")

    # Cap noisy sections.
    skills = _cap_text(skills, max_chars=400)

    # Prioritize recent experience (typically top of resume). Keep more context than skills.
    experience = _cap_text(experience, max_chars=1200)

    summary = (summary or "").strip()

    if not summary and not skills and not experience:
        # Mandatory fallback: if no sections are detected, use full resume text as experience.
        experience = _cap_text(clean_text(sections.full_text or ""), max_chars=1200)

    return {
        "summary": summary,
        "experience": experience,
        "skills": skills,
    }
