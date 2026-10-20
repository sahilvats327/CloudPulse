
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "cloud_metrics.csv"
MODEL_PATH = BASE / "models" / "xgboost_model_v3.json"
CONFIG_PATH = BASE / "models" / "xgboost_config_v3.json"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

METRICS = ["cpu_percent", "memory_percent", "network_mbps"]
LAGS = [1, 3, 6, 12]
WINDOWS = [3, 6, 12]


def build_features(df):
    features = df[METRICS].copy()

    for col in METRICS:
        for lag in LAGS:
            features[f"{col}_lag_{lag}"] = df[col].shift(lag)

        past = df[col].shift(1)
        for window in WINDOWS:
            features[f"{col}_mean_{window}"] = past.rolling(window).mean()
            features[f"{col}_std_{window}"] = past.rolling(window).std()

    for window in WINDOWS:
        baseline = df["network_mbps"].shift(1).rolling(window).mean()
        features[f"network_ratio_{window}"] = (
            df["network_mbps"] / baseline.clip(lower=1.0)
        )
        features[f"network_drop_from_baseline_{window}"] = (
            baseline - df["network_mbps"]
        )

    return features


def evaluate(name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1_score": f1_score(actual, predicted, zero_division=0),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

features = build_features(df)
valid = features.notna().all(axis=1)

X = features.loc[valid].reset_index(drop=True)
y = df.loc[valid, "is_anomaly"].astype(int).reset_index(drop=True)
network = df.loc[valid, "network_mbps"].reset_index(drop=True)
types = df.loc[valid, "anomaly_type"].reset_index(drop=True)

n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_val = X.iloc[train_end:val_end][config["features"]]
y_val = y.iloc[train_end:val_end].to_numpy()
network_val = network.iloc[train_end:val_end].to_numpy()

X_test = X.iloc[val_end:][config["features"]]
y_test = y.iloc[val_end:].to_numpy()
network_test = network.iloc[val_end:].to_numpy()
types_test = types.iloc[val_end:].to_numpy()

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

prob_threshold = float(config["model_probability_threshold"])
network_threshold = float(config["network_threshold_mbps"])

val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

# Keep the existing XGBoost threshold unchanged.
val_xgb = (val_prob >= prob_threshold).astype(int)
val_network = (network_val < network_threshold).astype(int)

# Select the extra gate using validation only.
# Among thresholds meeting the 2% validation false-alarm limit,
# choose the one with the highest validation F1.
candidates = np.unique(np.concatenate([
    np.linspace(prob_threshold, 1.0, 125),
    [prob_threshold],
]))

best = None
for gate in candidates:
    gated_network = (
        (network_val < network_threshold) & (val_prob >= gate)
    ).astype(int)

    pred = ((val_xgb == 1) | (gated_network == 1)).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, pred, labels=[0, 1]
    ).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = f1_score(y_val, pred, zero_division=0)

    if far <= 0.02 and (best is None or f1 > best["f1"]):
        best = {"threshold": float(gate), "f1": float(f1), "far": far}

if best is None:
    raise RuntimeError("No gate met the validation false-alarm constraint.")

# Test each approach once using frozen thresholds.
xgb_test = (test_prob >= prob_threshold).astype(int)
network_test_flag = (network_test < network_threshold).astype(int)

existing_hybrid = ((xgb_test == 1) | (network_test_flag == 1)).astype(int)
gated_network_test = (
    (network_test_flag == 1) & (test_prob >= best["threshold"])
).astype(int)
gated_hybrid = ((xgb_test == 1) | (gated_network_test == 1)).astype(int)

results = pd.DataFrame([
    evaluate("XGBoost only", y_test, xgb_test),
    evaluate("Existing hybrid", y_test, existing_hybrid),
    evaluate("Gated hybrid", y_test, gated_hybrid),
])

print("\nGATED HYBRID EXPERIMENT")
print("=" * 65)
print(f"Existing XGBoost threshold: {prob_threshold:.4f}")
print(f"Network threshold:          {network_threshold:.4f}")
print(f"Gate selected on validation: {best['threshold']:.4f}")
print(f"Validation F1 at gate:       {best['f1']:.4f}")
print(f"Validation false alarm rate: {best['far']:.4f}")

print("\nTEST RESULTS")
print(results.to_string(index=False, float_format="%.4f"))

predictions = pd.DataFrame({
    "timestamp": df.loc[valid, "timestamp"].reset_index(drop=True).iloc[val_end:].to_numpy(),
    "anomaly_type": types_test,
    "actual_anomaly": y_test,
    "xgboost_probability": test_prob,
    "xgboost_only": xgb_test,
    "existing_hybrid": existing_hybrid,
    "gated_hybrid": gated_hybrid,
})

pred_path = OUT / "gated_hybrid_test_predictions.csv"
results_path = OUT / "gated_hybrid_test_results.csv"
predictions.to_csv(pred_path, index=False)
results.to_csv(results_path, index=False)

print(f"\nSaved: {pred_path}")
print(f"Saved: {results_path}")
print("\nThis evaluates baseline data only; stress scenarios remain to be tested.")
