import pandas as pd
from src.feature_engineering import create_features

def test_create_features_returns_dataframe():
    df = pd.DataFrame({
        "Attrition_Flag": [0, 1],
        "Total_Trans_Ct": [100, 50],
        "Total_Revolving_Bal": [500, 200]
    })

    result = create_features(df)

    assert isinstance(result, pd.DataFrame)


def test_create_features_keeps_target():
    df = pd.DataFrame({
        "Attrition_Flag": [0, 1],
        "Total_Trans_Ct": [100, 50]
    })

    result = create_features(df)

    assert "Attrition_Flag" in result.columns
