from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import encode_target


def _get_positive_scores(model, X_test: pd.DataFrame) -> np.ndarray:
    """
    Return positive-class scores for a fitted model.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        return proba[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Map decision scores to [0, 1] with a sigmoid-like transform
        return 1 / (1 + np.exp(-scores))

    # Fallback: hard predictions
    return model.predict(X_test)


def evaluate_models(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, dict]:
    """
    Evaluate all models and return a metrics dictionary.

    Returns
    -------
    dict
        Example:
        {
            "logistic_regression": {"roc_auc": ..., "f1": ..., ...},
            "random_forest": {...},
            "xgboost": {...}
        }
    """
    y_true = encode_target(y_test)
    results: Dict[str, dict] = {}

    for name, model in models.items():
        y_scores = _get_positive_scores(model, X_test)
        y_pred = (y_scores >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_scores)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, y_scores)),
        }
        results[name] = metrics

    return results