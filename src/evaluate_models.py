from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from src.utils import encode_target


def evaluate_models(models, X_test, y_test):
    y_true = encode_target(y_test)

    results = {}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test)

        preds = (probs > 0.5).astype(int)

        results[name] = {
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "f1": float(f1_score(y_true, preds)),
            "recall": float(recall_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds)),
        }

    return results
