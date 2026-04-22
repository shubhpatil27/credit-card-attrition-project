from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import TARGET_COLUMN, get_categorical_columns, safe_divide


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features and transform categorical variables into numeric dummy columns.

    The target column Attrition_Flag is preserved.
    """
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    # --- Existing reusable ratio/index features ---
    if {"Credit_Limit", "Total_Revolving_Bal"}.issubset(df.columns):
        df["Balance_to_Limit_Ratio"] = safe_divide(df["Total_Revolving_Bal"], df["Credit_Limit"])

    if {"Total_Trans_Amt", "Total_Trans_Ct"}.issubset(df.columns):
        df["Transaction_Amt_per_Ct"] = safe_divide(df["Total_Trans_Amt"], df["Total_Trans_Ct"])

    if {"Months_Inactive_12_mon", "Total_Trans_Ct"}.issubset(df.columns):
        df["Inactive_to_Trans_Ratio"] = safe_divide(df["Months_Inactive_12_mon"], df["Total_Trans_Ct"] + 1)

    if {"Months_on_book", "Total_Trans_Ct"}.issubset(df.columns):
        df["Transactions_per_Month"] = safe_divide(df["Total_Trans_Ct"], df["Months_on_book"] + 1)

    if {"Contacts_Count_12_mon", "Months_Inactive_12_mon"}.issubset(df.columns):
        df["Inactivity_and_Contact_Index"] = df["Months_Inactive_12_mon"] + df["Contacts_Count_12_mon"]

    # --- Notebook logic features ---
    if {"Total_Revolving_Bal", "Total_Trans_Ct"}.issubset(df.columns):
        df["avg_revolving_per_trans"] = safe_divide(df["Total_Revolving_Bal"], df["Total_Trans_Ct"] + 1)

    if "Months_Inactive_12_mon" in df.columns:
        df["is_inactive"] = (df["Months_Inactive_12_mon"] >= 3).astype(int)

    if "Total_Trans_Ct" in df.columns:
        df["low_activity"] = (df["Total_Trans_Ct"] < 50).astype(int)

    if "Avg_Utilization_Ratio" in df.columns:
        df["low_utilization"] = (df["Avg_Utilization_Ratio"] < 0.2).astype(int)
        df["High_Utilization_Flag"] = (df["Avg_Utilization_Ratio"] >= 0.5).astype(int)

    if "Contacts_Count_12_mon" in df.columns:
        df["high_contact_count"] = (df["Contacts_Count_12_mon"] >= 4).astype(int)

    if {"Avg_Open_To_Buy", "Credit_Limit"}.issubset(df.columns):
        df["open_to_buy_ratio"] = safe_divide(df["Avg_Open_To_Buy"], df["Credit_Limit"] + 1)

    if "Credit_Limit" in df.columns:
        df["log_credit_limit"] = np.log1p(df["Credit_Limit"])

    # --- One-hot encode categorical columns ---
    categorical_cols = get_categorical_columns(df, exclude=[TARGET_COLUMN])
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure target remains present
    if TARGET_COLUMN not in df.columns:
        raise RuntimeError("Target column was lost during feature engineering.")

    # Replace any remaining inf/nan values
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.fillna(0)

    return df
