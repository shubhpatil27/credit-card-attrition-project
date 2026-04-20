# Phase 1 Contracts: Function Interfaces

## Overview

This document defines the **standard function interfaces** that all team members must follow.

These contracts ensure that:

* All modules integrate smoothly
* The Airflow pipeline runs without errors
* Code remains consistent across the project

⚠️ **Important Rule:**
If a function does not follow this contract, it should NOT be merged into `main`.

---

## General Rules

* All functions must use **pandas DataFrames**
* The target column must remain:

  ```
  Attrition_Flag
  ```
* Functions should **not write files unless specified**
* All paths and saving logic will be handled by Airflow

---

## 1. Data Loading

**File:** `src/data_loading.py`

```python
def load_data(path: str) -> pd.DataFrame:
```

### Description

Loads the raw dataset from the given path.

### Input

* `path`: string path to CSV file

### Output

* pandas DataFrame containing raw data

---

## 2. Data Preprocessing

**File:** `src/preprocessing.py`

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
```

### Description

Cleans the dataset by:

* handling missing values
* removing duplicates
* correcting data types

### Input

* raw DataFrame

### Output

* cleaned DataFrame

### Rules

* Must NOT remove `Attrition_Flag`
* Must return a valid DataFrame

---

## 3. Feature Engineering

**File:** `src/feature_engineering.py`

```python
def create_features(df: pd.DataFrame) -> pd.DataFrame:
```

### Description

Creates new features and transforms variables.

### Input

* cleaned DataFrame

### Output

* feature-engineered DataFrame

### Rules

* Must retain `Attrition_Flag`
* Must not introduce data leakage

---

## 4. Model Training

**File:** `src/train_models.py`

```python
def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
```

### Description

Trains multiple machine learning models.

### Input

* `X_train`: feature matrix
* `y_train`: target variable

### Output

Dictionary of trained models:

```python
{
    "logistic_regression": model1,
    "random_forest": model2,
    "xgboost": model3
}
```

### Rules

* Must return all three models
* Models must be fitted (trained)

---

## 5. Model Evaluation

**File:** `src/evaluate_models.py`

```python
def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
```

### Description

Evaluates model performance.

### Input

* dictionary of models
* test data

### Output

Dictionary of metrics:

```python
{
    "roc_auc": float,
    "f1": float,
    "recall": float,
    "precision": float
}
```

### Rules

* Must include all metrics
* Must return numeric values

---

## 6. Explainability (SHAP)

**File:** `src/explainability.py`

```python
def generate_shap(model, X: pd.DataFrame):
```

### Description

Generates SHAP values for model interpretability.

### Input

* trained model
* feature dataset

### Output

* SHAP values object or array

### Rules

* Should NOT save files directly (Airflow handles saving)

---

## Integration Notes

* Airflow will call these functions in sequence
* Functions should NOT depend on global variables
* All inputs must come from parameters

---

## Team Agreement

All team members must:

* Follow these contracts exactly
* Not change function signatures without discussion
* Ensure outputs match the defined format

---

## Enforcement

* Code must pass basic tests before merging
* Pull Requests must be reviewed
* Any violation of contracts must be corrected before merge
