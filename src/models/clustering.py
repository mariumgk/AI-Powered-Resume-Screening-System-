import os
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringArtifacts:
    kmeans: KMeans


def train_kmeans(
    X: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> Tuple[ClusteringArtifacts, Dict[str, float]]:
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)

    sil = -1.0
    if n_clusters >= 2 and X.shape[0] >= n_clusters:
        sil = float(silhouette_score(X, labels))

    return ClusteringArtifacts(kmeans=km), {"silhouette_score": sil, "n_clusters": float(n_clusters)}


def save_clustering(artifacts: ClusteringArtifacts, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifacts.kmeans, os.path.join(out_dir, "kmeans.joblib"))


def load_clustering(out_dir: str) -> ClusteringArtifacts:
    km = joblib.load(os.path.join(out_dir, "kmeans.joblib"))
    return ClusteringArtifacts(kmeans=km)
