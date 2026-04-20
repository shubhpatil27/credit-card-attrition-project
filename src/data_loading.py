from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils import standardize_column_names


def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file and standardize column names.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset as a DataFrame.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = standardize_column_names(df)
    return df