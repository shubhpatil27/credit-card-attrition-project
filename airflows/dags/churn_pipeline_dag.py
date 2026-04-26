import sys
sys.path.append("/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os

# ----------------------------
# Task 1: Preprocess
# ----------------------------
def preprocess_data():
    from src.preprocessing import clean_data
    from src.feature_engineering import create_features

    df = pd.read_excel("/opt/airflow/data/raw/credit_card_og.xlsx")

    df = clean_data(df)

    y = df["Attrition_Flag"].map({
        "Existing Customer": 0,
        "Attrited Customer": 1
    })

    X = df.drop("Attrition_Flag", axis=1)
    X = create_features(X)

    X["target"] = y

    os.makedirs("/opt/airflow/data/processed", exist_ok=True)
    X.to_csv("/opt/airflow/data/processed/data.csv", index=False)


# ----------------------------
# Task 2: Train model (NEW)
# ----------------------------
def train_model():
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    import joblib

    df = pd.read_csv("/opt/airflow/data/processed/data.csv")

    y = df["target"]
    X = df.drop("target", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    os.makedirs("/opt/airflow/models", exist_ok=True)
    joblib.dump(model, "/opt/airflow/models/xgboost.pkl")


# ----------------------------
# Task 3: Predict
# ----------------------------
def predict():
    import joblib

    df = pd.read_csv("/opt/airflow/data/processed/data.csv")

    model = joblib.load("/opt/airflow/models/xgboost.pkl")

    probs = model.predict_proba(df.drop("target", axis=1))[:, 1]

    risk_df = pd.DataFrame({"risk_score": probs})

    risk_df["risk_level"] = risk_df["risk_score"].apply(
        lambda x: "Low" if x < 0.5 else "High"
    )

    os.makedirs("/opt/airflow/outputs/predictions", exist_ok=True)
    risk_df.to_csv("/opt/airflow/outputs/predictions/risk_scores.csv", index=False)


# ----------------------------
# DAG
# ----------------------------
with DAG(
    dag_id="churn_risk_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id="train",
        python_callable=train_model
    )

    predict_task = PythonOperator(
        task_id="predict",
        python_callable=predict
    )

    preprocess >> train >> predict_task