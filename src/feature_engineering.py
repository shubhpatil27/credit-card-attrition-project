from __future__ import annotations

import numpy as np
import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features WITHOUT requiring target column.
    Safe for train/test pipelines.
    """
    df = df.copy()

   

    if {"Total_Revolving_Bal", "Credit_Limit"}.issubset(df.columns):
        df["balance_to_limit"] = df["Total_Revolving_Bal"] / (df["Credit_Limit"] + 1)

    if {"Total_Trans_Amt", "Total_Trans_Ct"}.issubset(df.columns):
        df["avg_spend_per_txn"] = df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1)

    if {"Total_Trans_Ct", "Months_on_book"}.issubset(df.columns):
        df["txn_per_month"] = df["Total_Trans_Ct"] / (df["Months_on_book"] + 1)

    if "Avg_Utilization_Ratio" in df.columns:
        df["high_utilization"] = (df["Avg_Utilization_Ratio"] > 0.5).astype(int)

    if {"Avg_Open_To_Buy", "Credit_Limit"}.issubset(df.columns):
        df["open_to_buy_ratio"] = df["Avg_Open_To_Buy"] / (df["Credit_Limit"] + 1)

    if "Credit_Limit" in df.columns:
        df["log_credit_limit"] = np.log1p(df["Credit_Limit"])

    # --- Encode categorical features ---
    df = pd.get_dummies(df, drop_first=True)

    # --- Clean numeric issues ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df