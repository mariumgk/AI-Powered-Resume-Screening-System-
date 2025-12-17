from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    """Compute Top-K accuracy.

    Why accuracy alone is insufficient:
    - In multi-class, near-miss predictions are common.
    - Top-K reflects decision-support behavior (ranked suggestions).
    """

    topk = np.argsort(-proba, axis=1)[:, :k]
    hits = 0
    for i in range(y_true.shape[0]):
        if int(y_true[i]) in set(topk[i].tolist()):
            hits += 1
    return float(hits / max(1, y_true.shape[0]))


def mean_reciprocal_rank(y_true: np.ndarray, proba: np.ndarray) -> float:
    order = np.argsort(-proba, axis=1)
    rr_sum = 0.0
    n = int(y_true.shape[0])
    for i in range(n):
        true_cls = int(y_true[i])
        ranks = np.where(order[i] == true_cls)[0]
        if ranks.size == 0:
            continue
        rr_sum += 1.0 / float(int(ranks[0]) + 1)
    return float(rr_sum / max(1, n))


def precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    return float(top_k_accuracy(y_true, proba, k=k))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top3_accuracy": float(top_k_accuracy(y_true, proba, k=3)),
        "precision_at_3": float(precision_at_k(y_true, proba, k=3)),
        "mrr": float(mean_reciprocal_rank(y_true, proba)),
    }
