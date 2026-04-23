def evaluate_task(**kwargs):
    import pandas as pd
    import joblib
    import json
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # Load data
    df = pd.read_csv("/opt/airflow/project/data/processed/features.csv")

    X = df.drop("Attrition_Flag", axis=1)
    y = df["Attrition_Flag"]

    # Load models
    models = {
        "random_forest": joblib.load("/opt/airflow/project/models/random_forest.pkl"),
    }

    model = models["random_forest"]

    # --- ROC CURVE ---
    probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.title("ROC Curve")

    plt.savefig("/opt/airflow/project/dashboard/static/roc.png")
    plt.close()

    # --- FEATURE IMPORTANCE ---
    importances = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.title("Feature Importance")

    plt.savefig("/opt/airflow/project/dashboard/static/feature_importance.png")
    plt.close()

    # --- METRICS ---
    from src.evaluate_models import evaluate_models

    metrics = evaluate_models(models, X, y)

    with open("/opt/airflow/project/models/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
