from __future__ import annotations

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

    # --- Feature creation using common credit-card churn dataset columns ---
    if {"Credit_Limit", "Total_Revolving_Bal"}.issubset(df.columns):
        df["Balance_to_Limit_Ratio"] = safe_divide(df["Total_Revolving_Bal"], df["Credit_Limit"])

    if {"Total_Trans_Amt", "Total_Trans_Ct"}.issubset(df.columns):
        df["Transaction_Amt_per_Ct"] = safe_divide(df["Total_Trans_Amt"], df["Total_Trans_Ct"])

    if {"Months_Inactive_12_mon", "Total_Trans_Ct"}.issubset(df.columns):
        df["Inactive_to_Trans_Ratio"] = safe_divide(df["Months_Inactive_12_mon"], df["Total_Trans_Ct"] + 1)

    if "Avg_Utilization_Ratio" in df.columns:
        df["High_Utilization_Flag"] = (df["Avg_Utilization_Ratio"] >= 0.5).astype(int)

    if {"Months_on_book", "Total_Trans_Ct"}.issubset(df.columns):
        df["Transactions_per_Month"] = safe_divide(df["Total_Trans_Ct"], df["Months_on_book"] + 1)

    if {"Contacts_Count_12_mon", "Months_Inactive_12_mon"}.issubset(df.columns):
        df["Inactivity_and_Contact_Index"] = df["Months_Inactive_12_mon"] + df["Contacts_Count_12_mon"]

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