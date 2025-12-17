import json
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class SkillOntology:
    domains: Dict[str, List[str]]
    aliases: Dict[str, str]

    @staticmethod
    def load(path: str) -> "SkillOntology":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        domains_obj = obj.get("domains", {})
        domains = {k: list(v.get("skills", [])) for k, v in domains_obj.items()}
        aliases = dict(obj.get("skill_aliases", {}))
        return SkillOntology(domains=domains, aliases=aliases)

    def all_skills(self) -> Set[str]:
        s: Set[str] = set()
        for lst in self.domains.values():
            for item in lst:
                s.add(item.lower())
        return s


def _normalize_skill(skill: str, aliases: Dict[str, str]) -> str:
    s = (skill or "").strip().lower()
    return aliases.get(s, s)


def extract_skills(text: str, ontology: SkillOntology) -> List[str]:
    """Extract skills using deterministic substring matching.

    This is intentionally simple/defensible for an academic project:
    - No black-box extraction
    - No heavy NLP dependencies
    """

    if not text:
        return []

    hay = text.lower()

    found: Set[str] = set()

    # Match canonical skills
    for sk in ontology.all_skills():
        if sk and sk in hay:
            found.add(_normalize_skill(sk, ontology.aliases))

    # Match aliases
    for alias, canonical in ontology.aliases.items():
        a = (alias or "").lower().strip()
        if a and a in hay:
            found.add(_normalize_skill(canonical, ontology.aliases))

    return sorted(found)
