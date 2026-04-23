import pandas as pd
from src.train_models import train_models

def test_train_models_returns_dict():
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8]
    })
    y = pd.Series([0, 1, 0, 1])

    models = train_models(X, y)

    assert isinstance(models, dict)


def test_models_exist():
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8]
    })
    y = pd.Series([0, 1, 0, 1])

    models = train_models(X, y)

    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "xgboost" in models
