def evaluate_task(**kwargs):
    import pandas as pd
    import json
    import joblib

    # Load processed data
    df = pd.read_csv("/opt/airflow/project/data/processed/features.csv")

    X = df.drop("Attrition_Flag", axis=1)
    y = df["Attrition_Flag"]

    # Load trained models
    models = {
        "logistic_regression": joblib.load("/opt/airflow/project/models/logistic_regression.pkl"),
        "random_forest": joblib.load("/opt/airflow/project/models/random_forest.pkl"),
        "xgboost": joblib.load("/opt/airflow/project/models/xgboost.pkl"),
    }

    # Evaluate
    metrics = evaluate_models(models, X, y)

    # ✅ SAVE HERE
    with open("/opt/airflow/project/models/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
