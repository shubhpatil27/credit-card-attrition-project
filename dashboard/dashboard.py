from flask import Flask, render_template
import json
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    # Load metrics
    try:
        with open("../models/model_metrics.json") as f:
            metrics = json.load(f)
    except:
        metrics = {}

    # Load predictions (optional for later)
    try:
        preds = pd.read_csv("../outputs/predictions/predictions.csv")
        total = len(preds)
        churn = preds["prediction"].sum()
    except:
        total = 0
        churn = 0

    return render_template(
        "index.html",
        metrics=metrics,
        total=total,
        churn=churn
    )

if __name__ == "__main__":
    app.run(debug=True)
