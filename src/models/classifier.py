import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class ClassifierArtifacts:
    model: LogisticRegression
    label_encoder: LabelEncoder


def train_classifier(
    X: np.ndarray,
    y: List[str],
    random_state: int = 42,
) -> Tuple[ClassifierArtifacts, Dict[str, object]]:
    """Train a class-balanced Logistic Regression classifier on embeddings."""

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=random_state,
        stratify=y_enc,
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=list(le.classes_),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_classification_report": report,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "classes": list(le.classes_),
    }

    return ClassifierArtifacts(model=clf, label_encoder=le), metrics


def predict_proba(artifacts: ClassifierArtifacts, x: np.ndarray) -> Dict[str, float]:
    probs = artifacts.model.predict_proba(x.reshape(1, -1))[0]
    out = {cls: float(p) for cls, p in zip(artifacts.label_encoder.classes_, probs)}
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def save_classifier(artifacts: ClassifierArtifacts, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifacts.model, os.path.join(out_dir, "classifier.joblib"))
    joblib.dump(artifacts.label_encoder, os.path.join(out_dir, "label_encoder.joblib"))


def load_classifier(out_dir: str) -> ClassifierArtifacts:
    model = joblib.load(os.path.join(out_dir, "classifier.joblib"))
    le = joblib.load(os.path.join(out_dir, "label_encoder.joblib"))
    return ClassifierArtifacts(model=model, label_encoder=le)
