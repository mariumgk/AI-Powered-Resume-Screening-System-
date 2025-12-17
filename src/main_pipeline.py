import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.embeddings.embedder import DEFAULT_MODEL_NAME, EmbeddingCache, MPNetEmbedder, save_embedder_metadata
from src.evaluation.metrics import classification_metrics
from src.models.classifier import load_classifier, save_classifier, train_classifier
from src.models.clustering import save_clustering, train_kmeans
from src.preprocessing.cleaner import clean_text
from src.preprocessing.chunk_builder import build_chunks
from src.preprocessing.section_extractor import extract_sections


def _paths() -> Tuple[str, str, str]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    raw_csv = os.path.join(data_dir, "raw", "updated_resume_dataset.csv")
    processed_dir = os.path.join(data_dir, "processed")
    artifacts_dir = os.path.join(processed_dir, "artifacts")
    return raw_csv, processed_dir, artifacts_dir


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place dataset CSV at data/raw/updated_resume_dataset.csv"
        )

    df = pd.read_csv(csv_path)

    cols = set(df.columns)
    if {"Resume", "Category"}.issubset(cols):
        df = df[["Resume", "Category"]].copy()
        df = df.rename(columns={"Resume": "resume_text", "Category": "label"})
        df["resume_text"] = df["resume_text"].fillna("")
        df["label"] = df["label"].fillna("Unknown")
        df.insert(0, "id", df.index.astype(str))
        return df

    if {"ID", "Resume_str", "Category"}.issubset(cols):
        df = df[["ID", "Resume_str", "Category"]].copy()
        df = df.rename(columns={"ID": "id", "Resume_str": "resume_text", "Category": "label"})
        df["resume_text"] = df["resume_text"].fillna("")
        df["label"] = df["label"].fillna("Unknown")
        df["id"] = df["id"].astype(str)
        return df

    raise ValueError(
        "Unrecognized dataset schema. Expected either columns: ['Resume','Category'] "
        "or ['ID','Resume_str','Category']."
    )


def preprocess_and_build_chunks(resume_texts: List[str]) -> List[dict]:
    out: List[dict] = []
    for raw in resume_texts:
        cleaned = clean_text(raw)
        sections = extract_sections(cleaned)
        out.append(build_chunks(sections))
    return out


def main() -> None:
    raw_csv, processed_dir, artifacts_dir = _paths()
    os.makedirs(os.path.join(processed_dir, "artifacts"), exist_ok=True)

    df = load_dataset(raw_csv)

    chunks_list = preprocess_and_build_chunks(df["resume_text"].tolist())
    labels = df["label"].astype(str).tolist()

    cache = EmbeddingCache(cache_path=os.path.join(processed_dir, "embedding_cache.joblib"))
    embedder = MPNetEmbedder(model_name=DEFAULT_MODEL_NAME, cache=cache)

    batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "16"))
    X = embedder.encode_weighted_chunks(
        chunks_list,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    save_embedder_metadata(os.path.join(artifacts_dir, "embedder.json"), DEFAULT_MODEL_NAME)

    # Offline training & evaluation (test metrics only).
    artifacts, clf_info = train_classifier(X=X, y=labels, random_state=42)
    save_classifier(artifacts, artifacts_dir)

    # Evaluate using explicit metrics (accuracy alone is insufficient).
    # Rebuild the same split to compute top-k metrics consistently.
    from sklearn.preprocessing import LabelEncoder

    le = artifacts.label_encoder
    y_enc = le.transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    proba = artifacts.model.predict_proba(X_test)
    y_pred = artifacts.model.predict(X_test)
    metrics = classification_metrics(y_true=y_test, y_pred=y_pred, proba=proba)

    # Clustering (unsupervised) on all embeddings.
    clustering_artifacts, clustering_info = train_kmeans(X=X, n_clusters=10, random_state=42)
    save_clustering(clustering_artifacts, artifacts_dir)

    # Save an inference-safe dataset index for ranking and recommendations.
    # IMPORTANT: Do NOT store ground-truth labels here. Flask must not access Category labels.
    dataset_ids = df["id"].astype(str).tolist()
    cluster_labels = clustering_artifacts.kmeans.predict(X).astype(int).tolist()
    resume_index_path = os.path.join(artifacts_dir, "resume_index.joblib")
    joblib.dump(
        {
            "ids": dataset_ids,
            "embeddings": X,
            "cluster_labels": cluster_labels,
        },
        resume_index_path,
    )

    report_path = os.path.join(artifacts_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": metrics,
                "classifier_report": clf_info["test_classification_report"],
                "clustering": clustering_info,
                "resume_index": {
                    "n_resumes": int(len(dataset_ids)),
                    "index_file": "resume_index.joblib",
                },
            },
            f,
            indent=2,
        )

    print("=== Offline Evaluation (TEST ONLY) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"silhouette_score: {clustering_info['silhouette_score']:.4f}")
    print(f"Saved artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
