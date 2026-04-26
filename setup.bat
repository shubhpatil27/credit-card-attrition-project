@echo off
echo Starting setup...

REM Create virtual environment
python -m venv churn_env
call churn_env\Scripts\activate

REM Install packages
pip install --upgrade pip
pip install -r requirements.txt

REM Run notebooks (optional)
jupyter nbconvert --to notebook --execute notebooks\02_preprocessing_feature_engineering.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\03_model_training.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks\04_shap_analysis.ipynb --inplace

REM Start dashboard
start cmd /k "cd dashboard && python dashboard.py"

REM Start airflow
docker-compose down --volumes --remove-orphans
docker-compose build --no-cache
docker-compose up

echo Setup complete!
pause