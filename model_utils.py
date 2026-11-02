from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET_COLUMNS = ("id", "label", "attack_cat")


def prepare_features(dataframe: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode traffic features and align them to the training schema."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    features = dataframe.drop(columns=list(TARGET_COLUMNS), errors="ignore")
    features = pd.get_dummies(features)
    if feature_columns is not None:
        features = features.reindex(columns=feature_columns, fill_value=0)
    return features


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return binary metrics with explicit zero-division handling."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
