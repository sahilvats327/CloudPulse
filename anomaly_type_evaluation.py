
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "cloud_metrics.csv"
PREDICTIONS_PATH = BASE / "outputs" / "fair_model_predictions.csv"
OUTPUT_DIR = BASE / "outputs"

# Load original dataset and common-test-set predictions.
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

predictions = pd.read_csv(PREDICTIONS_PATH)
predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])

# Match every prediction to its original timestamp and anomaly type.
data_columns = ["timestamp", "is_anomaly", "anomaly_type"]
joined = predictions.merge(
    df[data_columns],
    on="timestamp",
    how="left",
    validate="many_to_one",
)

if joined["anomaly_type"].isna().any():
    raise ValueError("Some prediction timestamps did not match the dataset.")

if not (joined["actual_anomaly"] == joined["is_anomaly"]).all():
    raise ValueError("Prediction labels do not match the original dataset.")

# Include each anomaly type and the normal class.
types = ["cpu_spike", "memory_drift", "network_drop", "combined", "normal"]
models = [
    "LSTM",
    "GRU",
    "XGBoost V3 Model Only",
    "CloudPulse V3 Hybrid",
]

rows = []

for model_name in models:
    model_data = joined[joined["model"] == model_name]

    for anomaly_type in types:
        if anomaly_type == "normal":
            subset = model_data[model_data["is_anomaly"] == 0]
        else:
            subset = model_data[model_data["anomaly_type"] == anomaly_type]

        if subset.empty:
            continue

        actual = subset["is_anomaly"].astype(int)
        predicted = subset["prediction"].astype(int)

        # For anomaly types, report detected / actual anomalies.
        # For normal observations, report false alarms / normal observations.
        if anomaly_type == "normal":
            total = len(subset)
            false_alarms = int(predicted.sum())
            rows.append({
                "model": model_name,
                "category": anomaly_type,
                "observations": total,
                "actual_anomalies": 0,
                "detected_anomalies": 0,
                "missed_anomalies": 0,
                "false_alarms": false_alarms,
                "detection_rate_percent": None,
                "false_alarm_percent": round(
                    100 * false_alarms / total, 3
                ) if total else 0.0,
            })
        else:
            total = len(subset)
            detected = int(predicted.sum())
            rows.append({
                "model": model_name,
                "category": anomaly_type,
                "observations": total,
                "actual_anomalies": total,
                "detected_anomalies": detected,
                "missed_anomalies": total - detected,
                "false_alarms": 0,
                "detection_rate_percent": round(
                    100 * detected / total, 2
                ) if total else 0.0,
                "false_alarm_percent": None,
            })

results = pd.DataFrame(rows)

csv_path = OUTPUT_DIR / "anomaly_type_comparison.csv"
results.to_csv(csv_path, index=False)

print("=" * 76)
print("CLOUDPULSE — PER-ANOMALY-TYPE EVALUATION")
print("=" * 76)
print("Common test timestamps:", joined["timestamp"].nunique())
print("Models evaluated:", joined["model"].nunique())
print("\nDetection rates by anomaly type (%):")

detection = results[results["category"] != "normal"].pivot(
    index="category",
    columns="model",
    values="detection_rate_percent",
)

print(detection.to_string())

print("\nFalse alarms on normal observations:")
normal = results[results["category"] == "normal"][
    ["model", "observations", "false_alarms", "false_alarm_percent"]
]
print(normal.to_string(index=False))

print("\nDetailed results saved to:", csv_path)
