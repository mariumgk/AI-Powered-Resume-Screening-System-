from dataclasses import dataclass
from typing import Dict, List, Tuple

from .skill_graph import SkillOntology


@dataclass(frozen=True)
class DomainEvidence:
    domain_scores: Dict[str, float]
    matched_skills_by_domain: Dict[str, List[str]]


def score_domains_from_skills(skills: List[str], ontology: SkillOntology) -> DomainEvidence:
    skills_set = {s.lower() for s in (skills or [])}

    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}

    for domain, domain_skills in ontology.domains.items():
        domain_skill_set = {s.lower() for s in domain_skills}
        overlap = sorted(list(skills_set.intersection(domain_skill_set)))
        matched[domain] = overlap
        scores[domain] = float(len(overlap))

    # Normalize to [0,1] if possible
    max_score = max(scores.values()) if scores else 0.0
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}

    return DomainEvidence(domain_scores=scores, matched_skills_by_domain=matched)


def top_domains(domain_scores: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:k]
