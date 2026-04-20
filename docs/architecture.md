# Project Architecture

## Overview
This project is designed as an end-to-end machine learning system with Airflow as the backend orchestration layer and a Flask-based dashboard for presenting results.

## Architecture Components

### 1. Data Layer
- Raw dataset stored in `data/raw/`
- Processed data stored in `data/processed/`
- Single credit card dataset used for the project

### 2. Airflow Backend (Orchestration Layer)
Airflow manages the entire workflow and ensures all steps run in sequence.

Pipeline tasks:
1. Data ingestion
2. Data cleaning and validation
3. Feature engineering
4. Model training (Logistic Regression, Random Forest, XGBoost)
5. Model evaluation (ROC-AUC, F1, Recall, PR-AUC)
6. SHAP explainability generation
7. Export outputs for dashboard

### 3. Modeling Layer
Models used:
- Logistic Regression (baseline)
- Random Forest (intermediate)
- XGBoost or LightGBM (advanced)
- SHAP for model interpretability

### 4. Output Layer
Generated outputs:
- Model metrics (JSON)
- Predictions (CSV)
- Visualizations (plots)
- SHAP results

Stored in:
- `outputs/figures/`
- `outputs/predictions/`
- `outputs/tables/`

### 5. Dashboard Layer
A Flask-based dashboard displays:
- Attrition overview
- Model performance
- Feature importance
- SHAP explanations
- Customer risk segmentation

The dashboard reads precomputed outputs and does not perform heavy computation.

## Data Flow

Raw Data → Airflow Pipeline → Outputs → Flask Dashboard

## Key Advantages
- Automated workflow using Airflow
- Advanced machine learning models
- Explainability using SHAP
- Clear separation between backend and frontend
- Business-focused insights