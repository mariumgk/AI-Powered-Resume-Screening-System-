from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge.skill_graph import SkillOntology, extract_skills


@dataclass(frozen=True)
class MatchResult:
    similarity: float
    matched_skills: List[str]
    missing_skills: List[str]
    explanation: str


def rank_embeddings_by_cosine_similarity(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: Sequence[str],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Rank a corpus by cosine similarity to a query embedding.

    Assumption:
        Embeddings are produced in the same MPNet embedding space.

    Note:
        If embeddings are L2-normalized (our default), cosine similarity reduces to dot product.
    """

    if query_embedding is None or corpus_embeddings is None or len(corpus_ids) == 0:
        return []

    if corpus_embeddings.shape[0] != len(corpus_ids):
        raise ValueError("corpus_embeddings rows must match corpus_ids length")

    q = query_embedding.reshape(1, -1)

    # Robust cosine computation (works even if not normalized).
    scores = cosine_similarity(q, corpus_embeddings)[0]
    order = np.argsort(-scores)[: max(0, int(top_n))]

    return [(str(corpus_ids[i]), float(scores[i])) for i in order]


def match_resume_to_jd(
    resume_text: str,
    jd_text: str,
    resume_embedding: np.ndarray,
    jd_embedding: np.ndarray,
    ontology: SkillOntology,
) -> MatchResult:
    if resume_embedding is None or jd_embedding is None:
        return MatchResult(
            similarity=0.0,
            matched_skills=[],
            missing_skills=[],
            explanation="Job matching unavailable due to missing embeddings.",
        )

    sim = float(cosine_similarity(resume_embedding.reshape(1, -1), jd_embedding.reshape(1, -1))[0][0])

    resume_skills = set(extract_skills(resume_text or "", ontology))
    jd_skills = set(extract_skills(jd_text or "", ontology))

    matched = sorted(list(resume_skills.intersection(jd_skills)))
    missing = sorted(list(jd_skills.difference(resume_skills)))

    explanation = (
        f"Cosine similarity in MPNet embedding space: {sim:.3f}. "
        f"Matched {len(matched)} skills; missing {len(missing)} skills from the JD." 
    )

    return MatchResult(similarity=sim, matched_skills=matched, missing_skills=missing, explanation=explanation)
