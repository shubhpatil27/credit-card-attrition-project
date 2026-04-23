import pandas as pd
from src.preprocessing import clean_data

def test_clean_data_returns_dataframe():
    df = pd.DataFrame({
        "Attrition_Flag": [0, 1],
        "CLIENTNUM": [1, 2],
        "Year": [2019, 2019]
    })

    result = clean_data(df)

    assert isinstance(result, pd.DataFrame)


def test_clean_data_keeps_target():
    df = pd.DataFrame({
        "Attrition_Flag": [0, 1],
        "CLIENTNUM": [1, 2],
        "Year": [2019, 2019]
    })

    result = clean_data(df)

    assert "Attrition_Flag" in result.columns
