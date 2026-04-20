# Team Roles and Responsibilities

This project is divided among three members to ensure a balanced and collaborative workflow.

---

## Ji hye Lee — Data, Reporting & Presentation Lead

Responsibilities:
- Perform exploratory data analysis (EDA)
- Clean and preprocess the dataset
- Handle missing values and outliers
- Perform feature engineering
- Generate initial visualizations
- Lead final report writing
- Lead presentation creation (slides and structure)

Key Files:
- notebooks/01_eda.ipynb
- notebooks/02_feature_engineering.ipynb
- src/preprocessing.py
- src/feature_engineering.py
- reports/

---

## Venkata Lacha Reddy Peram — Modeling & Explainability

Responsibilities:
- Train machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost or LightGBM
- Compare model performance
- Evaluate models using:
  - ROC-AUC
  - F1-score
  - Recall
  - Precision-Recall AUC
- Implement SHAP for explainability
- Generate feature importance and SHAP plots
- Contribute results and insights to the report

Key Files:
- notebooks/03_model_training.ipynb
- notebooks/04_shap_analysis.ipynb
- src/train_models.py
- src/evaluate_models.py
- src/explainability.py
- models/
- outputs/

---

## Shubham Suryakant Patil — Airflow Backend & Dashboard Integration

Responsibilities:
- Build and manage the Airflow DAG
- Automate:
  - data loading
  - preprocessing
  - feature engineering
  - model training
  - evaluation
  - output generation
- Save outputs for dashboard use
- Build and maintain the Flask dashboard
- Connect Airflow outputs to the dashboard
- Document system architecture for the report

Key Files:
- airflows/dags/churn_pipeline_dag.py
- dashboard/
- outputs/
- docs/architecture.md

---

## Collaboration Strategy

- Use GitHub for version control
- Each member works on a separate branch
- Use pull requests for merging changes
- Maintain clean and consistent code structure
- Communicate regularly to ensure integration across components