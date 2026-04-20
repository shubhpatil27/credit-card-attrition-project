from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import encode_target

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """
    Train multiple machine learning models and return them in a dictionary.
    """
    y = encode_target(y_train)

    models: Dict[str, object] = {}

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    lr.fit(X_train, y)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y)
    models["random_forest"] = rf

    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        xgb = GradientBoostingClassifier(random_state=42)

    xgb.fit(X_train, y)
    models["xgboost"] = xgb

    return models