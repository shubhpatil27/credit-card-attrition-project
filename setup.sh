#!/bin/bash

echo "🚀 Starting setup for Customer Risk System..."

# ----------------------------
# STEP 1: Create virtual env
# ----------------------------
echo "📦 Creating virtual environment..."
python3 -m venv churn_env

echo "🔌 Activating environment..."
source churn_env/bin/activate

# ----------------------------
# STEP 2: Install dependencies
# ----------------------------
echo "📥 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# ----------------------------
# STEP 3: Run notebooks (optional auto-run)
# ----------------------------
echo "📊 Running preprocessing and training..."

jupyter nbconvert --to notebook --execute notebooks/02_preprocessing_feature_engineering.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_shap_analysis.ipynb --inplace

# ----------------------------
# STEP 4: Start Dashboard
# ----------------------------
echo "🌐 Starting dashboard..."
cd dashboard
python dashboard.py &

cd ..

# ----------------------------
# STEP 5: Start Airflow (Docker)
# ----------------------------
echo "🐳 Starting Airflow..."

docker-compose down --volumes --remove-orphans
docker-compose build --no-cache
docker-compose up -d

echo ""
echo "✅ SETUP COMPLETE!"
echo ""
echo "👉 Dashboard: http://127.0.0.1:5000"
echo "👉 Airflow:   http://localhost:8080"
echo "👉 Login: admin / admin"
echo ""