
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS = ["precision", "recall", "f1_score"]


def load_metrics(filename):
    path = OUTPUT_DIR / filename
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Load existing model evaluation results; no models are retrained.
lstm = load_metrics("lstm_metrics.json")
gru = load_metrics("gru_metrics.json")
xgb_v1 = load_metrics("xgboost_metrics.json")
xgb_v2 = load_metrics("xgboost_metrics_v2.json")
xgb_v3_model = load_metrics("xgboost_model_only_metrics_v3.json")
xgb_v3_hybrid = load_metrics("xgboost_metrics_v3.json")

models = [
    {"Model": "LSTM", **{m: lstm[m] for m in METRICS}},
    {"Model": "GRU", **{m: gru[m] for m in METRICS}},
    {"Model": "XGBoost V1", **{m: xgb_v1[m] for m in METRICS}},
    {"Model": "XGBoost V2", **{m: xgb_v2[m] for m in METRICS}},
    {
        "Model": "XGBoost V3 Model Only",
        **{m: xgb_v3_model[m] for m in METRICS},
    },
    {
        "Model": "CloudPulse V3 Hybrid",
        **{m: xgb_v3_hybrid[m] for m in METRICS},
    },
]

comparison = pd.DataFrame(models)

# Keep the original output filename, now with all six reported models.
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nCloudPulse: Model Performance Comparison")
print("(Reported metrics; evaluation procedures may differ.)")
print(
    comparison.to_string(
        index=False,
        formatters={m: "{:.2%}".format for m in METRICS},
    )
)

# Save a separate chart, preserving the existing LSTM/GRU chart.
comparison.set_index("Model")[METRICS].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1.05),
)

plt.title("CloudPulse: LSTM, GRU, XGBoost and Hybrid Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=20, ha="right")
plt.legend(["Precision", "Recall", "F1-score"])
plt.tight_layout()

chart_path = OUTPUT_DIR / "model_comparison_all_models.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("\nSaved:")
print("  outputs/model_comparison.csv")
print("  outputs/model_comparison_all_models.png")
print("Original LSTM/GRU chart preserved.")
