
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cloud_metrics.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 55)
print("CloudPulse - XGBoost V2 (Network-Relative Features)")
print("=" * 55)

# 1. Load and sort the data chronologically
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

metrics = ["cpu_percent", "memory_percent", "network_mbps"]
target = "is_anomaly"

# 2. Create features using past observations only.
# Current measurements are included to classify the current observation.
features = df[metrics].copy()

for column in metrics:
    # Lag features
    for lag in [1, 3, 6, 12]:
        features[f"{column}_lag_{lag}"] = df[column].shift(lag)

    # Historical rolling statistics
    past_values = df[column].shift(1)

    for window in [3, 6, 12]:
        features[f"{column}_mean_{window}"] = (
            past_values.rolling(window).mean()
        )
        features[f"{column}_std_{window}"] = (
            past_values.rolling(window).std()
        )

# 3. Network-relative features
# Compare current throughput with baselines from past observations only.
for window in [3, 6, 12]:
    baseline = df["network_mbps"].shift(1).rolling(window).mean()

    features[f"network_ratio_{window}"] = (
        df["network_mbps"] / baseline.clip(lower=1.0)
    )
    features[f"network_drop_from_baseline_{window}"] = (
        baseline - df["network_mbps"]
    )

# Remove rows without sufficient historical observations
valid = features.notna().all(axis=1)

features = features.loc[valid].reset_index(drop=True)
labels = df.loc[valid, target].astype(int).reset_index(drop=True)
timestamps = df.loc[valid, "timestamp"].reset_index(drop=True)

print(f"Usable rows: {len(features)}")
print(f"Number of features: {features.shape[1]}")

# 4. Chronological 70/15/15 split
n = len(features)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = features.iloc[:train_end]
y_train = labels.iloc[:train_end]

X_val = features.iloc[train_end:val_end]
y_val = labels.iloc[train_end:val_end]

X_test = features.iloc[val_end:]
y_test = labels.iloc[val_end:]
test_timestamps = timestamps.iloc[val_end:]

print(f"\nTraining rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Testing rows:    {len(X_test)}")
print(f"Training anomalies: {int(y_train.sum())}")
print(f"Validation anomalies: {int(y_val.sum())}")
print(f"Test anomalies: {int(y_test.sum())}")

if y_train.sum() == 0 or y_train.sum() == len(y_train):
    raise ValueError("Training data must contain both classes.")

# 5. Train XGBoost with class weighting
scale_pos_weight = (
    (len(y_train) - int(y_train.sum())) / int(y_train.sum())
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=4,
    tree_method="hist",
)

print("\nTraining XGBoost V2...")
model.fit(X_train, y_train)
print("Training complete.")

# 6. Select the probability threshold using validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

best_threshold = None
best_f1 = -1.0

for threshold in np.arange(0.05, 0.951, 0.01):
    val_predictions = (val_probabilities >= threshold).astype(int)

    tn_val, fp_val, fn_val, tp_val = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1],
    ).ravel()

    false_alarm_rate_val = (
        fp_val / (fp_val + tn_val)
        if (fp_val + tn_val)
        else 0.0
    )

    score = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    # Maximize validation F1 while keeping false alarms <= 2%.
    if false_alarm_rate_val <= 0.02 and score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

# Fallback if no candidate threshold meets the constraint
if best_threshold is None:
    best_threshold = 0.50
    best_f1 = f1_score(
        y_val,
        (val_probabilities >= best_threshold).astype(int),
        zero_division=0,
    )

print(f"Selected validation threshold: {best_threshold:.2f}")
print(f"Validation F1 at selected threshold: {best_f1:.4f}")

# 7. Evaluate once on the held-out test data
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0

test_metrics = {
    "model": "XGBoost V2",
    "threshold": float(best_threshold),
    "test_rows": int(len(y_test)),
    "actual_anomalies": int(y_test.sum()),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "false_alarm_rate": float(false_alarm_rate),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
}

# 8. Save the V2 model and configuration
model.save_model(str(MODEL_DIR / "xgboost_model_v2.json"))

config = {
    "model": "XGBoost V2",
    "threshold": float(best_threshold),
    "features": list(X_train.columns),
    "lag_steps": [1, 3, 6, 12],
    "rolling_windows": [3, 6, 12],
    "network_relative_windows": [3, 6, 12],
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "scale_pos_weight": float(scale_pos_weight),
}

with open(MODEL_DIR / "xgboost_config_v2.json", "w") as file:
    json.dump(config, file, indent=4)

with open(OUTPUT_DIR / "xgboost_metrics_v2.json", "w") as file:
    json.dump(test_metrics, file, indent=4)

# Save test predictions
predictions_df = pd.DataFrame({
    "timestamp": test_timestamps,
    "actual_anomaly": y_test,
    "predicted_anomaly": test_predictions,
    "anomaly_probability": test_probabilities,
})

predictions_df.to_csv(
    OUTPUT_DIR / "xgboost_test_predictions_v2.csv",
    index=False,
)

# Save feature importance
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "xgboost_feature_importance_v2.csv",
    index=False,
)

# 9. Display results
print("\n" + "=" * 55)
print("XGBOOST V2 TEST RESULTS")
print("=" * 55)

for key, value in test_metrics.items():
    if isinstance(value, float):
        print(f"{key:22s}: {value:.4f}")
    else:
        print(f"{key:22s}: {value}")

print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))

print("\nSaved files:")
print("  models/xgboost_model_v2.json")
print("  models/xgboost_config_v2.json")
print("  outputs/xgboost_metrics_v2.json")
print("  outputs/xgboost_test_predictions_v2.csv")
print("  outputs/xgboost_feature_importance_v2.csv")

print("\nXGBoost V2 completed successfully.")
