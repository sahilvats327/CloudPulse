
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cloud_metrics.csv"

df = pd.read_csv(DATA_PATH)

# Preserve chronological order.
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Use only earlier observations to estimate the current baseline.
# shift(1) prevents the current observation from influencing its baseline.
window = 12
df["network_baseline"] = (
    df["network_mbps"]
    .shift(1)
    .rolling(window=window, min_periods=window)
    .median()
)

df["network_ratio"] = (
    df["network_mbps"] / df["network_baseline"].clip(lower=1.0)
)

# A relative-drop score: larger values mean a bigger drop
# relative to recent throughput.
df["relative_drop"] = 1.0 - df["network_ratio"]

# Match the existing chronological 70% / 15% / 15% split.
usable = df.dropna(subset=["network_baseline"]).copy()
n = len(usable)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = usable.iloc[:train_end].copy()
validation = usable.iloc[train_end:val_end].copy()
test = usable.iloc[val_end:].copy()

# Learn a threshold from validation data only.
# This initial experiment limits the validation false-alarm rate
# to approximately 2% among normal observations.
normal_scores = validation.loc[
    validation["is_anomaly"] == 0, "relative_drop"
]

threshold = normal_scores.quantile(0.98)

# Keep this detector independent of the existing hybrid rule.
test["context_prediction"] = (
    test["relative_drop"] >= threshold
).astype(int)

y_true = test["is_anomaly"].astype(int)
y_pred = test["context_prediction"]

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

print("\nCONTEXT-AWARE NETWORK DETECTOR")
print("=" * 65)
print(f"History window:              {window} observations")
print(f"Validation threshold:        {threshold:.6f}")
print(f"Test rows:                   {len(test)}")
print(f"Actual anomalies:            {int(y_true.sum())}")
print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
print(f"Precision:                   {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:                      {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 score:                    {f1_score(y_true, y_pred, zero_division=0):.4f}")
print(f"False alarm rate:            {fp / (tn + fp):.4f}")

output_path = BASE_DIR / "outputs" / "context_network_test_predictions.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
test.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print("\nNote: this is an experimental detector, not yet a replacement")
print("for the existing CloudPulse hybrid model.")
print("\nRESULTS BY ANOMALY TYPE")
print("=" * 65)

for anomaly_type, group in test.groupby("anomaly_type"):
    actual = group["is_anomaly"].astype(int)
    predicted = group["context_prediction"]

    detected = int(((actual == 1) & (predicted == 1)).sum())
    total = int(actual.sum())
    false_alarms = int(((actual == 0) & (predicted == 1)).sum())

    if total > 0:
        print(
            f"{anomaly_type:15s}: "
            f"detected {detected}/{total}, "
            f"missed {total - detected}"
        )

    if anomaly_type == "normal":
        print(f"Normal false alarms: {false_alarms}/{len(group)}")