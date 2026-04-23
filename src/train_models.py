from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils import encode_target

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    y = encode_target(y_train)

    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train, y)
    models["logistic_regression"] = lr

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y)
    models["random_forest"] = rf

    # XGBoost / fallback
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss"
        )
    else:
        xgb = GradientBoostingClassifier()

    xgb.fit(X_train, y)
    models["xgboost"] = xgb

    return models
