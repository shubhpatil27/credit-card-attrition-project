from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np
import pandas as pd


TARGET_COLUMN = "Attrition_Flag"


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names and replace repeated spaces with underscores.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def encode_target(y: pd.Series) -> pd.Series:
    """
    Convert Attrition_Flag into binary labels.

    Common mapping:
    - Existing Customer -> 0
    - Attrited Customer  -> 1
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    y_clean = y.astype(str).str.strip().str.lower()

    mapping = {
        "existing customer": 0,
        "attrited customer": 1,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
    }

    encoded = y_clean.map(mapping)

    if encoded.isna().any():
        unknown = sorted(y_clean[encoded.isna()].unique().tolist())
        raise ValueError(f"Unrecognized target values in {TARGET_COLUMN}: {unknown}")

    return encoded.astype(int)


def get_categorical_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """
    Return object/category columns excluding any columns in `exclude`.
    """
    exclude_set = set(exclude or [])
    cols = [
        c
        for c in df.columns
        if c not in exclude_set and (
            pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
        )
    ]
    return cols


def get_numeric_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """
    Return numeric columns excluding any columns in `exclude`.
    """
    exclude_set = set(exclude or [])
    cols = [
        c for c in df.columns
        if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Safe division that avoids divide-by-zero problems.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)