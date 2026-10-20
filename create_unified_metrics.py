
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
out = base / "outputs"

# Main comparison: common test set
fair = pd.read_csv(out / "fair_model_comparison.csv")

# Additional detector experiments
adaptive = pd.read_csv(out / "adaptive_network_metrics_v1.csv")
arima = pd.read_csv(out / "arima_network_metrics_v1.csv")

extra_rows = []

for filename, model_name in [
    ("adaptive_network_metrics_v1.csv", "Adaptive Network Detector V1"),
    ("arima_network_metrics_v1.csv", "ARIMA Network Detector V1"),
]:
    df = pd.read_csv(out / filename)
    test = df[df["split"].str.lower() == "test"]

    if test.empty:
        continue

    row = test.iloc[0]
    extra_rows.append({
        "model": model_name,
        "test_rows": int(row["rows"]),
        "actual_anomalies": int(row["actual_anomalies"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tp": int(row["tp"]),
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1_score": float(row["f1_score"]),
        "false_alarm_rate": float(row["false_alarm_rate"]),
    })

# Standardize the main comparison columns
summary = fair.copy()

for row in extra_rows:
    summary = pd.concat(
        [summary, pd.DataFrame([row])],
        ignore_index=True
    )

destination = out / "cloudpulse_unified_metrics.csv"
summary.to_csv(destination, index=False)

print("Unified metrics CSV created:")
print(destination)
print()
print(summary[[
    "model", "test_rows", "precision",
    "recall", "f1_score", "false_alarm_rate"
]].to_string(index=False))
