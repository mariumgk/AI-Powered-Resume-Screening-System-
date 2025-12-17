import json
import os
import sys
from typing import Any, Dict, Optional

import joblib

from src.embeddings.embedder import DEFAULT_MODEL_NAME, MPNetEmbedder
from src.knowledge.skill_graph import SkillOntology, extract_skills
from src.knowledge.domain_mapping import score_domains_from_skills
from src.models.classifier import load_classifier, predict_proba
from src.models.clustering import load_clustering
from src.matching.resume_jd_matcher import match_resume_to_jd
from src.preprocessing.cleaner import clean_text
from src.preprocessing.section_extractor import extract_sections
from src.preprocessing.chunk_builder import build_chunks
from src.reasoning.csp_engine import apply_reasoning


def _load_artifacts_only(base_dir: str) -> Dict[str, Any]:
    data_dir = os.path.join(base_dir, "data")
    artifacts_dir = os.path.join(data_dir, "processed", "artifacts")

    embedder_meta_path = os.path.join(artifacts_dir, "embedder.json")
    if os.path.exists(embedder_meta_path):
        with open(embedder_meta_path, "r", encoding="utf-8") as f:
            embedder_meta = json.load(f)
        model_name = str(embedder_meta.get("embedding_model") or DEFAULT_MODEL_NAME)
    else:
        model_name = DEFAULT_MODEL_NAME

    ontology_path = os.path.join(data_dir, "skill_ontology.json")

    ontology = SkillOntology.load(ontology_path)
    clf_artifacts = load_classifier(artifacts_dir)
    clustering = load_clustering(artifacts_dir)

    resume_index_path = os.path.join(artifacts_dir, "resume_index.joblib")
    resume_index: Optional[Dict[str, Any]] = None
    if os.path.exists(resume_index_path):
        resume_index = joblib.load(resume_index_path)

    embedder = MPNetEmbedder(model_name=model_name, cache=None)

    return {
        "artifacts_dir": artifacts_dir,
        "ontology": ontology,
        "clf": clf_artifacts,
        "clustering": clustering,
        "resume_index": resume_index,
        "embedder": embedder,
    }


def run_inference(resume_text: str, jd_text: str) -> Dict[str, Any]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ctx = _load_artifacts_only(base_dir)

    cleaned_resume = clean_text(resume_text or "")
    cleaned_jd = clean_text(jd_text or "")

    resume_sections = extract_sections(cleaned_resume)
    jd_sections = extract_sections(cleaned_jd)

    resume_chunks = build_chunks(resume_sections)
    jd_chunks = build_chunks(jd_sections)

    res_emb = ctx["embedder"].encode_weighted_chunks_one(resume_chunks)
    jd_emb = ctx["embedder"].encode_weighted_chunks_one(jd_chunks) if cleaned_jd else None

    ml_probs = predict_proba(ctx["clf"], res_emb)
    top_items = list(ml_probs.items())[:3]
    max_prob = float(top_items[0][1]) if top_items else 0.0

    resume_skills = extract_skills(cleaned_resume, ctx["ontology"])
    evidence = score_domains_from_skills(resume_skills, ctx["ontology"])

    decision = apply_reasoning(ml_probs=ml_probs, skill_evidence=evidence, resume_text=cleaned_resume)

    match = match_resume_to_jd(
        resume_text=cleaned_resume,
        jd_text=cleaned_jd,
        resume_embedding=res_emb,
        jd_embedding=jd_emb,
        ontology=ctx["ontology"],
    )

    alpha = float(os.environ.get("SCREEN_ALPHA", "0.7"))
    beta = float(os.environ.get("SCREEN_BETA", "0.3"))
    raw_sim = float(match.similarity)
    sim_component = float((raw_sim + 1.0) / 2.0)
    if not cleaned_jd:
        alpha = 0.0
    suitability = float(alpha * sim_component + beta * max_prob)

    if suitability >= 0.75:
        strength = "Strong Match"
    elif suitability >= 0.55:
        strength = "Partial Match"
    else:
        strength = "Weak Match"

    return {
        "screening": {
            "suitability_score": suitability,
            "match_strength": strength,
            "alpha": alpha,
            "beta": beta,
            "similarity_raw": raw_sim,
            "max_class_probability": max_prob,
            "top3": [{"domain": d, "prob": float(p)} for d, p in top_items],
        },
        "chunks": {
            "summary_len": len(resume_chunks.get("summary") or ""),
            "experience_len": len(resume_chunks.get("experience") or ""),
            "skills_len": len(resume_chunks.get("skills") or ""),
        },
        "decision_support": {
            "primary_domain": decision.primary_domain,
            "confidence": float(decision.confidence),
        },
        "match": {
            "matched_skills": match.matched_skills,
            "missing_skills": match.missing_skills,
            "explanation": match.explanation,
        },
    }


def main() -> None:
    resume_text = (os.environ.get("SMOKE_RESUME") or "").strip()
    jd_text = (os.environ.get("SMOKE_JD") or "").strip()

    if not resume_text:
        resume_text = "Experienced data analyst with Python, SQL, and dashboarding."
    if not jd_text:
        jd_text = "Looking for a data analyst with SQL, Python, and BI dashboard experience."

    out = run_inference(resume_text=resume_text, jd_text=jd_text)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
