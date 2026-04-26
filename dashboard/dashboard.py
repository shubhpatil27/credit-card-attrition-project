from flask import Flask, render_template
import pandas as pd
import json

app = Flask(__name__)

@app.route("/")
def home():

    # ----------------------------
    # Load model metrics
    # ----------------------------
    try:
        with open("../models/model_metrics.json") as f:
            metrics = json.load(f)
    except:
        metrics = {}

    # ----------------------------
    # Load risk predictions
    # ----------------------------
    try:
        preds = pd.read_csv("../outputs/predictions/risk_scores.csv")

        total = len(preds)

        high_risk = (preds["risk_level"] == "High").sum()
        low_risk = (preds["risk_level"] == "Low").sum()

        risk_rate = (high_risk / total) * 100 if total > 0 else 0

    except:
        total = 0
        high_risk = 0
        low_risk = 0
        risk_rate = 0

    # ----------------------------
    # Dynamic Insights
    # ----------------------------
    insights = [
    f"{high_risk} customers are at high risk of churn.",
    f"{round(risk_rate,2)}% of customers are likely to leave.",
    "Low transaction activity is a strong indicator of churn.",
    "Inactive customers have significantly higher attrition rates.",
    "Target high-risk customers with engagement campaigns."
    ]

    # ----------------------------
    # Render dashboard
    # ----------------------------
    return render_template(
        "index.html",
        metrics=metrics,
        total=total,
        high_risk=high_risk,
        low_risk=low_risk,
        risk_rate=round(risk_rate, 2),
        insights=insights
    )


if __name__ == "__main__":
    app.run(debug=True)