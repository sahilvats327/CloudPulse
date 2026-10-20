
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

print("=" * 60)
print("CloudPulse - XGBoost V3 Hybrid")
print("=" * 60)

# 1. Load data in chronological order
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

metrics = ["cpu_percent", "memory_percent", "network_mbps"]
target = "is_anomaly"

# 2. Build features using current measurements and past history
features = df[metrics].copy()

for column in metrics:
    for lag in [1, 3, 6, 12]:
        features[f"{column}_lag_{lag}"] = df[column].shift(lag)

    past_values = df[column].shift(1)

    for window in [3, 6, 12]:
        features[f"{column}_mean_{window}"] = (
            past_values.rolling(window).mean()
        )
        features[f"{column}_std_{window}"] = (
            past_values.rolling(window).std()
        )

# Network-relative features based only on preceding observations
for window in [3, 6, 12]:
    baseline = df["network_mbps"].shift(1).rolling(window).mean()

    features[f"network_ratio_{window}"] = (
        df["network_mbps"] / baseline.clip(lower=1.0)
    )
    features[f"network_drop_from_baseline_{window}"] = (
        baseline - df["network_mbps"]
    )

# Keep rows with complete feature history
valid = features.notna().all(axis=1)

features = features.loc[valid].reset_index(drop=True)
labels = df.loc[valid, target].astype(int).reset_index(drop=True)
timestamps = df.loc[valid, "timestamp"].reset_index(drop=True)
network_values = (
    df.loc[valid, "network_mbps"].reset_index(drop=True)
)

# 3. Chronological 70/15/15 split
n = len(features)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = features.iloc[:train_end], labels.iloc[:train_end]
X_val, y_val = features.iloc[train_end:val_end], labels.iloc[train_end:val_end]
X_test, y_test = features.iloc[val_end:], labels.iloc[val_end:]

test_timestamps = timestamps.iloc[val_end:]
test_network = network_values.iloc[val_end:]

print(f"Usable rows: {n}")
print(f"Features: {features.shape[1]}")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows: {len(X_test)}")
print(f"Training anomalies: {int(y_train.sum())}")
print(f"Validation anomalies: {int(y_val.sum())}")
print(f"Test anomalies: {int(y_test.sum())}")

if y_train.nunique() < 2:
    raise ValueError("Training data must contain both classes.")

# 4. Learn the network threshold from TRAINING DATA ONLY.
# Select the largest threshold that detects every training network-drop
# example while producing zero false positives on training normal rows.
train_network = network_values.iloc[:train_end].reset_index(drop=True)
train_labels = y_train.reset_index(drop=True)

train_drops = train_network[
    (train_labels == 1)
    & (df.loc[valid, "anomaly_type"].reset_index(drop=True).iloc[:train_end]
       .eq("network_drop"))
]
train_normal = train_network[train_labels == 0]

if train_drops.empty:
    raise ValueError("No network-drop examples in training data.")

# A valid separating threshold must be above every network-drop value
# and no higher than the minimum normal throughput. Use midpoint.
max_drop = float(train_drops.max())
min_normal = float(train_normal.min())

if max_drop >= min_normal:
    raise ValueError(
        "Training data does not cleanly separate network drops "
        "from normal throughput. A more robust threshold strategy is needed."
    )

network_threshold = (max_drop + min_normal) / 2.0

print(f"\nTraining maximum network-drop throughput: {max_drop:.4f} Mbps")
print(f"Training minimum normal throughput: {min_normal:.4f} Mbps")
print(f"Training-derived network threshold: {network_threshold:.4f} Mbps")

# 5. Train XGBoost using training data only
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

print("\nTraining XGBoost V3...")
model.fit(X_train, y_train)
print("Training complete.")

# 6. Select model probability threshold on validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

best_threshold = None
best_f1 = -1.0

for threshold in np.arange(0.05, 0.951, 0.01):
    val_predictions = (val_probabilities >= threshold).astype(int)

    tn_v, fp_v, fn_v, tp_v = confusion_matrix(
        y_val, val_predictions, labels=[0, 1]
    ).ravel()

    false_alarm_v = fp_v / (fp_v + tn_v) if (fp_v + tn_v) else 0.0
    score = f1_score(y_val, val_predictions, zero_division=0)

    if false_alarm_v <= 0.02 and score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

if best_threshold is None:
    best_threshold = 0.50
    best_f1 = f1_score(
        y_val,
        (val_probabilities >= best_threshold).astype(int),
        zero_division=0,
    )

print(f"Validation-selected model threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_f1:.4f}")

# 7. Evaluate once on the untouched test data
test_probabilities = model.predict_proba(X_test)[:, 1]
model_predictions = (test_probabilities >= best_threshold).astype(int)

# Hybrid: XGBoost OR training-derived low-throughput rule
network_rule_predictions = (test_network.to_numpy() < network_threshold).astype(int)
hybrid_predictions = np.logical_or(
    model_predictions == 1,
    network_rule_predictions == 1,
).astype(int)


def calculate_metrics(name, actual, predicted, probabilities=None):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    result = {
        "model": name,
        "test_rows": int(len(actual)),
        "actual_anomalies": int(np.sum(actual)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }

    if probabilities is not None:
        result["roc_auc"] = float(roc_auc_score(actual, probabilities))

    return result


model_metrics = calculate_metrics(
    "XGBoost V3 Model Only",
    y_test,
    model_predictions,
    test_probabilities,
)

hybrid_metrics = calculate_metrics(
    "CloudPulse V3 Hybrid",
    y_test,
    hybrid_predictions,
)

# 8. Save V3 files only; V1 and V2 files remain untouched
model.save_model(str(MODEL_DIR / "xgboost_model_v3.json"))

config = {
    "model": "CloudPulse V3 Hybrid",
    "model_probability_threshold": float(best_threshold),
    "network_rule": "network_mbps < network_threshold_mbps",
    "network_threshold_mbps": float(network_threshold),
    "threshold_selection": (
        "network threshold learned from training data only; "
        "model probability threshold selected on validation data"
    ),
    "training_max_network_drop_mbps": max_drop,
    "training_min_normal_network_mbps": min_normal,
    "features": list(X_train.columns),
    "lag_steps": [1, 3, 6, 12],
    "rolling_windows": [3, 6, 12],
    "network_relative_windows": [3, 6, 12],
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "scale_pos_weight": float(scale_pos_weight),
}

with open(MODEL_DIR / "xgboost_config_v3.json", "w") as file:
    json.dump(config, file, indent=4)

with open(OUTPUT_DIR / "xgboost_metrics_v3.json", "w") as file:
    json.dump(hybrid_metrics, file, indent=4)

with open(OUTPUT_DIR / "xgboost_model_only_metrics_v3.json", "w") as file:
    json.dump(model_metrics, file, indent=4)

predictions_df = pd.DataFrame({
    "timestamp": test_timestamps.to_numpy(),
    "actual_anomaly": y_test.to_numpy(),
    "xgboost_probability": test_probabilities,
    "xgboost_prediction": model_predictions,
    "network_rule_prediction": network_rule_predictions,
    "hybrid_prediction": hybrid_predictions,
    "network_mbps": test_network.to_numpy(),
})

predictions_df.to_csv(
    OUTPUT_DIR / "xgboost_test_predictions_v3.csv",
    index=False,
)

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "xgboost_feature_importance_v3.csv",
    index=False,
)

comparison = pd.DataFrame([model_metrics, hybrid_metrics])
comparison.to_csv(
    OUTPUT_DIR / "xgboost_v3_comparison.csv",
    index=False,
)

# Per-anomaly-type evaluation on test set, for diagnostic reporting only
test_start = val_end
test_types = (
    df.loc[valid, "anomaly_type"]
    .reset_index(drop=True)
    .iloc[test_start:]
    .reset_index(drop=True)
)

type_rows = []
for anomaly_type in sorted(test_types.unique()):
    mask = test_types.eq(anomaly_type).to_numpy()
    if mask.sum() == 0:
        continue

    type_rows.append({
        "anomaly_type": anomaly_type,
        "test_rows": int(mask.sum()),
        "actual_anomalies": int(y_test.to_numpy()[mask].sum()),
        "xgboost_detected": int(
            ((model_predictions == 1) & mask).sum()
        ),
        "hybrid_detected": int(
            ((hybrid_predictions == 1) & mask).sum()
        ),
    })

pd.DataFrame(type_rows).to_csv(
    OUTPUT_DIR / "xgboost_v3_anomaly_type_metrics.csv",
    index=False,
)

print("\n" + "=" * 60)
print("V3 TEST RESULTS")
print("=" * 60)

for result in [model_metrics, hybrid_metrics]:
    print(f"\n{result['model']}")
    for key, value in result.items():
        if key == "model":
            continue
        if isinstance(value, float):
            print(f"{key:22s}: {value:.4f}")
        else:
            print(f"{key:22s}: {value}")

print("\nPer-anomaly-type detections:")
print(pd.DataFrame(type_rows).to_string(index=False))

print("\nV3 files saved:")
print("  models/xgboost_model_v3.json")
print("  models/xgboost_config_v3.json")
print("  outputs/xgboost_metrics_v3.json")
print("  outputs/xgboost_model_only_metrics_v3.json")
print("  outputs/xgboost_test_predictions_v3.csv")
print("  outputs/xgboost_feature_importance_v3.csv")
print("  outputs/xgboost_v3_comparison.csv")
print("  outputs/xgboost_v3_anomaly_type_metrics.csv")

print("\nV3 completed successfully.")
