
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent

data = pd.read_csv(BASE_DIR / "data" / "cloud_metrics.csv")
pred = pd.read_csv(
    BASE_DIR / "outputs" / "xgboost_test_predictions_v2.csv"
)

data["timestamp"] = pd.to_datetime(data["timestamp"])
pred["timestamp"] = pd.to_datetime(pred["timestamp"])

df = pred.merge(
    data[["timestamp", "network_mbps", "anomaly_type", "is_anomaly"]],
    on="timestamp",
    how="left",
    validate="one_to_one",
)

# Verify that the prediction file and source labels agree.
if not (df["actual_anomaly"] == df["is_anomaly"]).all():
    raise ValueError("Prediction labels do not match source dataset.")

# Baseline: original XGBoost V2 predictions
baseline = df["predicted_anomaly"].astype(int)

# Hybrid: V2 OR a low-throughput network-drop rule
network_rule = (df["network_mbps"] < 10).astype(int)
hybrid = ((baseline == 1) | (network_rule == 1)).astype(int)

actual = df["is_anomaly"].astype(int)

def report(name, predictions):
    tn, fp, fn, tp = confusion_matrix(
        actual, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("-" * 50)
    print(f"TN:        {tn}")
    print(f"FP:        {fp}")
    print(f"FN:        {fn}")
    print(f"TP:        {tp}")
    print(f"Accuracy:  {accuracy_score(actual, predictions):.4f}")
    print(f"Precision: {precision_score(actual, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(actual, predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(actual, predictions, zero_division=0):.4f}")

print("CLOUDPULSE V3 — HYBRID EVALUATION")
print("=" * 50)
print(f"Test rows: {len(df)}")
print("Network rule: network_mbps < 10")

report("XGBoost V2 baseline", baseline)
report("Hybrid V3", hybrid)

print("\nHybrid results by anomaly type")
print("-" * 50)

for anomaly_type, group in df[actual == 1].groupby("anomaly_type"):
    detected = int(hybrid.loc[group.index].sum())
    total = len(group)
    print(f"{anomaly_type:15s}: {detected}/{total} detected ({detected / total:.2%})")

print("\nEvaluation completed. No model or existing output files were changed.")
