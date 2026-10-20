
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# CloudPulse ARIMA Network Detector V1
# Read-only input; separate experiment outputs
# --------------------------------------------------

DATA_PATH = Path("data/cloud_metrics.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "arima_network_predictions_v1.csv"
METRICS_PATH = OUTPUT_DIR / "arima_network_metrics_v1.csv"
CONFIG_PATH = OUTPUT_DIR / "arima_network_config_v1.json"

# Chronological 70% / 15% / 15% split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# Small ARIMA model for a first experiment
ARIMA_ORDER = (2, 1, 2)

# Allow at most 2% false-alarm rate on normal validation rows
MAX_VALIDATION_FAR = 0.02

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

if df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps found.")
if df["network_mbps"].isna().any():
    raise ValueError("Missing network_mbps values found.")

n = len(df)
train_end = int(n * TRAIN_RATIO)
validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

# Fit only on the training period.
train_series = df.loc[: train_end - 1, "network_mbps"].astype(float)

print("Fitting ARIMA model...")
print("Training rows:", len(train_series))
print("ARIMA order:", ARIMA_ORDER)

model = ARIMA(
    train_series,
    order=ARIMA_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted = model.fit()

# One-step-ahead predictions: each forecast uses observations
# available before the timestamp being forecast.
# Parameters are trained on the training period and kept fixed.
# The observed history is updated sequentially after each forecast.
history = list(train_series.to_numpy())
forecast_values = []

for i in range(train_end, n):
    current_model = ARIMA(
        history,
        order=ARIMA_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    current_fit = current_model.filter(fitted.params)
    forecast = float(current_fit.forecast(steps=1)[0])
    forecast_values.append(forecast)
    history.append(float(df.loc[i, "network_mbps"]))

# Keep rows from validation and test periods.
evaluation = df.iloc[train_end:].copy().reset_index(drop=True)
evaluation["forecast_network_mbps"] = forecast_values
evaluation["forecast_error"] = (
    evaluation["network_mbps"] - evaluation["forecast_network_mbps"]
)
evaluation["absolute_error"] = evaluation["forecast_error"].abs()

# Threshold selected using only validation observations labelled normal.
validation_count = validation_end - train_end
validation = evaluation.iloc[:validation_count].copy()
validation_normal = validation.loc[
    validation["is_anomaly"] == 0, "absolute_error"
].dropna()

if validation_normal.empty:
    raise ValueError("No normal observations in validation period.")

# Candidate thresholds are empirical normal-error quantiles.
candidates = np.unique(
    np.quantile(validation_normal, np.linspace(0.90, 0.999, 100))
)

best = None
for threshold in candidates:
    pred = (validation["absolute_error"].to_numpy() >= threshold).astype(int)
    actual = validation["is_anomaly"].to_numpy().astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual, pred, labels=[0, 1]
    ).ravel()

    far = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    if far <= MAX_VALIDATION_FAR:
        candidate_result = (f1, recall, -far, float(threshold))
        if best is None or candidate_result > best:
            best = candidate_result

if best is None:
    raise RuntimeError("No threshold met the validation FAR constraint.")

threshold = best[3]
evaluation["predicted_anomaly"] = (
    evaluation["absolute_error"] >= threshold
).astype(int)
evaluation["split"] = [
    "validation" if i < validation_count else "test"
    for i in range(len(evaluation))
]
evaluation["threshold"] = threshold

def calculate_metrics(part):
    actual = part["is_anomaly"].astype(int).to_numpy()
    predicted = part["predicted_anomaly"].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    far = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(part) if len(part) else 0.0

    return {
        "split": part["split"].iloc[0],
        "threshold": threshold,
        "rows": len(part),
        "actual_anomalies": int(actual.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_alarm_rate": round(far, 4),
    }

metrics = pd.DataFrame([
    calculate_metrics(evaluation[evaluation["split"] == "validation"]),
    calculate_metrics(evaluation[evaluation["split"] == "test"]),
])

# Save separate files; original project outputs remain unchanged.
evaluation.to_csv(PREDICTIONS_PATH, index=False)
metrics.to_csv(METRICS_PATH, index=False)

config = {
    "model": "ARIMA",
    "order": list(ARIMA_ORDER),
    "train_rows": train_end,
    "validation_rows": validation_count,
    "test_rows": n - validation_end,
    "threshold": threshold,
    "max_validation_false_alarm_rate": MAX_VALIDATION_FAR,
    "forecast_mode": "one-step-ahead; fixed training parameters; observed history updated",
    "warning": (
        "Synthetic dataset experiment. Forecast errors may detect multiple "
        "causes of traffic change and are not proof of a network attack."
    ),
}

CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

print("\nARIMA NETWORK DETECTOR V1")
print("Selected validation threshold:", round(threshold, 4))
print("Validation F1:", round(best[0], 4))
print("Validation recall:", round(best[1], 4))
print("\nValidation and test metrics:")
print(metrics.to_string(index=False))
print("\nSaved files:")
print(PREDICTIONS_PATH)
print(METRICS_PATH)
print(CONFIG_PATH)
