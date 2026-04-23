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

    Uses a sample of the dataset for efficiency.

    Parameters
    ----------
    model : trained model
    X : pd.DataFrame
        Feature dataset (must be numeric)

    Returns
    -------
    dict containing:
        - explainer
        - shap_values
        - feature_names
    """

    # Ensure numeric input
    X = X.select_dtypes(include=[np.number])

    # Sample data for performance (SHAP can be slow on large datasets)
    X_sample = X.sample(n=min(500, len(X)), random_state=42)

    model_name = model.__class__.__name__

    # Tree-based models
    if model_name in ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    # Linear models
    elif model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_sample)

    # Fallback (generic models)
    else:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "feature_names": list(X_sample.columns),
    }
