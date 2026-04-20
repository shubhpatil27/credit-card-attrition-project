from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError("The 'shap' package is required for explainability.") from exc


def generate_shap(model: Any, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate SHAP values for a fitted model.

    Returns a dictionary so the caller can save or visualize the outputs later.
    """
    # Tree models
    tree_model_types = ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier")
    model_name = model.__class__.__name__

    if model_name in tree_model_types:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)
    else:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "feature_names": list(X.columns),
    }