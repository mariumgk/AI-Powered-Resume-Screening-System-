import json
import os
from typing import Any, Dict, Optional

import joblib
from flask import Flask, render_template, request

from src.embeddings.embedder import DEFAULT_MODEL_NAME, MPNetEmbedder
from src.ingestion.pdf_parser import extract_text_from_pdf_bytes
from src.knowledge.domain_mapping import score_domains_from_skills
from src.knowledge.skill_graph import SkillOntology, extract_skills
from src.matching.resume_jd_matcher import match_resume_to_jd, rank_embeddings_by_cosine_similarity
from src.models.classifier import load_classifier, predict_proba
from src.models.clustering import load_clustering
from src.preprocessing.cleaner import clean_text
from src.preprocessing.chunk_builder import build_chunks
from src.preprocessing.section_extractor import extract_sections
from src.reasoning.csp_engine import apply_reasoning


def create_app() -> Flask:
    app = Flask(__name__)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    ontology_path = os.path.join(data_dir, "skill_ontology.json")

    artifacts_dir = os.path.join(data_dir, "processed", "artifacts")

    # Inference must be artifact-only: do not read from data/raw or other data/processed paths.
    embedder_meta_path = os.path.join(artifacts_dir, "embedder.json")
    if os.path.exists(embedder_meta_path):
        with open(embedder_meta_path, "r", encoding="utf-8") as f:
            embedder_meta = json.load(f)
        model_name = str(embedder_meta.get("embedding_model") or DEFAULT_MODEL_NAME)
    else:
        model_name = DEFAULT_MODEL_NAME

    ontology = SkillOntology.load(ontology_path)
    clf_artifacts = load_classifier(artifacts_dir)

    # Clustering is optional for inference UI; loaded to ensure artifact consistency.
    clustering = load_clustering(artifacts_dir)

    resume_index_path = os.path.join(artifacts_dir, "resume_index.joblib")
    resume_index: Optional[Dict[str, Any]] = None
    if os.path.exists(resume_index_path):
        resume_index = joblib.load(resume_index_path)

    # IMPORTANT: cache is disabled in inference to guarantee no reads/writes outside artifacts.
    embedder = MPNetEmbedder(model_name=model_name, cache=None)

    @app.route("/", methods=["GET", "POST"])
    def index():
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None

        if request.method == "POST":
            try:
                resume_text = (request.form.get("resume_text") or "").strip()
                jd_text = (request.form.get("jd_text") or "").strip()

                resume_pdf = request.files.get("resume_pdf")
                if resume_pdf and resume_pdf.filename:
                    resume_text = extract_text_from_pdf_bytes(resume_pdf.read())

                if not (resume_text or "").strip():
                    raise ValueError("Resume text is required (paste text or upload a PDF).")

                cleaned_resume = clean_text(resume_text)
                cleaned_jd = clean_text(jd_text)

                resume_sections = extract_sections(cleaned_resume)
                jd_sections = extract_sections(cleaned_jd)

                resume_chunks = build_chunks(resume_sections)
                jd_chunks = build_chunks(jd_sections)

                res_emb = embedder.encode_weighted_chunks_one(resume_chunks)
                jd_emb = embedder.encode_weighted_chunks_one(jd_chunks) if cleaned_jd else None

                ml_probs = predict_proba(clf_artifacts, res_emb)
                top_items = list(ml_probs.items())[:3]
                top_domains = [
                    {
                        "domain": str(d),
                        "prob": float(p),
                        "pct": float(p) * 100.0,
                    }
                    for d, p in top_items
                ]
                max_prob = float(top_items[0][1]) if top_items else 0.0

                resume_skills = extract_skills(cleaned_resume, ontology)
                evidence = score_domains_from_skills(resume_skills, ontology)

                decision = apply_reasoning(ml_probs=ml_probs, skill_evidence=evidence, resume_text=cleaned_resume)

                match = match_resume_to_jd(
                    resume_text=cleaned_resume,
                    jd_text=cleaned_jd,
                    resume_embedding=res_emb,
                    jd_embedding=jd_emb,
                    ontology=ontology,
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

                # Optional decision-support add-ons based on offline dataset index.
                jd_ranked_resumes = []
                similar_resumes = []
                cluster_id: Optional[int] = None
                if resume_index is not None:
                    ids = resume_index.get("ids", [])
                    emb = resume_index.get("embeddings", None)
                    cl = resume_index.get("cluster_labels", [])

                    if emb is not None and len(ids) == emb.shape[0]:
                        jd_ranked_resumes = rank_embeddings_by_cosine_similarity(
                            query_embedding=jd_emb,
                            corpus_embeddings=emb,
                            corpus_ids=ids,
                            top_n=5,
                        )

                        similar_resumes = rank_embeddings_by_cosine_similarity(
                            query_embedding=res_emb,
                            corpus_embeddings=emb,
                            corpus_ids=ids,
                            top_n=5,
                        )

                    try:
                        cluster_id = int(clustering.kmeans.predict(res_emb.reshape(1, -1))[0])
                    except Exception:
                        cluster_id = None

                result = {
                    "screening": {
                        "suitability_score": suitability,
                        "match_strength": strength,
                        "top_domains": top_domains,
                        "alpha": alpha,
                        "beta": beta,
                        "similarity_raw": raw_sim,
                        "similarity_component": sim_component,
                        "max_class_probability": max_prob,
                    },
                    "decision": {
                        "primary_domain": decision.primary_domain,
                        "secondary_domains": decision.secondary_domains,
                        "confidence": decision.confidence,
                        "explanation_trace": decision.explanation_trace,
                        "ml_probs": ml_probs,
                        "skill_domain_scores": evidence.domain_scores,
                    },
                    "match": {
                        "similarity": match.similarity,
                        "matched_skills": match.matched_skills,
                        "missing_skills": match.missing_skills,
                        "explanation": match.explanation,
                    },
                    "resume_skills": resume_skills,
                    "cluster_id": cluster_id,
                    "jd_ranked_resumes": jd_ranked_resumes,
                    "similar_resumes": similar_resumes,
                }

            except Exception as e:
                error = str(e)

        return render_template("index.html", result=result, error=error)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
