import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.knowledge.domain_mapping import DomainEvidence, top_domains


@dataclass(frozen=True)
class DecisionOutput:
    primary_domain: str
    secondary_domains: List[Tuple[str, float]]
    confidence: float
    explanation_trace: List[str]


_RE_YEARS = re.compile(r"\b(\d{1,2})\s*\+?\s*years\b")


def estimate_years_experience(text: str) -> int:
    if not text:
        return 0
    matches = [int(m.group(1)) for m in _RE_YEARS.finditer(text.lower()) if m.group(1).isdigit()]
    return max(matches) if matches else 0


def apply_reasoning(
    ml_probs: Dict[str, float],
    skill_evidence: DomainEvidence,
    resume_text: str,
) -> DecisionOutput:
    """Rule-based reasoning/constraint satisfaction (no ML).

    Inputs:
        - ml_probs: probabilities from Logistic Regression
        - skill_evidence: normalized domain scores derived from ontology overlap

    Rules (defensible, deterministic):
        - If skill evidence strongly supports a different domain than ML top-1, reduce confidence.
        - If top-2 ML probs are close, return multi-domain (secondary domains).
        - If resume is very short or malformed, cap confidence.
        - Optional experience constraint: very low experience reduces confidence.
    """

    trace: List[str] = []

    if not ml_probs:
        return DecisionOutput(
            primary_domain="Unknown",
            secondary_domains=[],
            confidence=0.0,
            explanation_trace=["No ML probabilities available."],
        )

    ml_ranked = list(ml_probs.items())
    primary, primary_p = ml_ranked[0]
    trace.append(f"ML top-1 domain: {primary} (p={primary_p:.3f}).")

    # Multi-domain detection based on probability gap.
    secondary: List[Tuple[str, float]] = []
    if len(ml_ranked) >= 2:
        d2, p2 = ml_ranked[1]
        gap = primary_p - p2
        if gap < 0.10:
            secondary.append((d2, float(p2)))
            trace.append(f"Top-2 close (gap={gap:.3f}) => multi-domain candidate: {d2}.")

    # Skill graph evidence adjustment.
    skill_top = top_domains(skill_evidence.domain_scores, k=1)
    if skill_top:
        skill_dom, skill_score = skill_top[0]
        trace.append(f"Skill-evidence top domain: {skill_dom} (score={skill_score:.3f}).")

        if skill_score >= 0.75 and skill_dom != primary:
            trace.append(
                "High skill evidence conflicts with ML top-1 => reducing confidence and promoting secondary domain."
            )
            secondary.insert(0, (primary, float(primary_p)))
            primary = skill_dom
            primary_p = max(primary_p * 0.8, 0.05)

    # Experience constraint.
    years = estimate_years_experience(resume_text)
    if years == 0:
        trace.append("Experience years not detected; applying mild uncertainty.")
        primary_p *= 0.95
    elif years < 1:
        trace.append("<1 year experience detected; reducing confidence.")
        primary_p *= 0.85

    # Short resume constraint.
    n_chars = len(resume_text or "")
    if n_chars < 300:
        trace.append("Resume text is short; capping confidence.")
        primary_p = min(primary_p, 0.50)

    confidence = float(max(min(primary_p, 1.0), 0.0))

    # Deduplicate secondary domains, keep order.
    seen = {primary}
    uniq_secondary: List[Tuple[str, float]] = []
    for d, p in secondary:
        if d not in seen:
            uniq_secondary.append((d, float(p)))
            seen.add(d)

    return DecisionOutput(
        primary_domain=primary,
        secondary_domains=uniq_secondary,
        confidence=confidence,
        explanation_trace=trace,
    )
