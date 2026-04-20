from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import TARGET_COLUMN, get_categorical_columns, get_numeric_columns


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling duplicates, missing values, and basic type cleanup.

    Returns a cleaned DataFrame while keeping the target column intact.
    """
    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Strip string values in object columns
    obj_cols = get_categorical_columns(df)
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Try to coerce numeric-like columns
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if df[col].dtype == "object":
            coerced = pd.to_numeric(df[col], errors="ignore")
            df[col] = coerced

    # Fill missing numeric columns with median
    numeric_cols = get_numeric_columns(df, exclude=[TARGET_COLUMN])
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Fill missing categorical columns with mode
    categorical_cols = get_categorical_columns(df, exclude=[TARGET_COLUMN])
    for col in categorical_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # Final clean-up for obvious infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # If any numeric NaNs remain, fill with median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df