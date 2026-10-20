
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cloud_metrics.csv"
CONFIG_PATH = BASE_DIR / "models" / "xgboost_config_v3.json"
OUTPUT_DIR = BASE_DIR / "outputs"

WINDOW = 12
MAX_VALIDATION_FAR = 0.02
EPSILON = 1e-6


def calculate_network_features(df):
    """Calculate historical network baseline and robust deviation."""
    network = df["network_mbps"].astype(float)

    # Shift first so the current observation is excluded.
    history = network.shift(1)

    rolling_median = history.rolling(
        window=WINDOW, min_periods=WINDOW
    ).median()

    def rolling_mad(values):
        median = np.median(values)
        return np.median(np.abs(values - median))

    rolling_mad_values = history.rolling(
        window=WINDOW, min_periods=WINDOW
    ).apply(rolling_mad, raw=True)

    # Robust scale estimate; floor prevents division by zero.
    robust_scale = (1.4826 * rolling_mad_values).clip(lower=EPSILON)

    # Positive scores indicate unusually low network throughput.
    score = (
        (rolling_median - network) / robust_scale
    )

    result = pd.DataFrame({
        "network_baseline": rolling_median,
        "network_mad": rolling_mad_values,
        "network_anomaly_score": score,
    }, index=df.index)

    return result


def calculate_metrics(actual, predicted, split_name, threshold):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "split": split_name,
        "threshold": float(threshold),
        "rows": int(len(actual)),
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
        "false_alarm_rate": float(
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
    }


def select_threshold(y_true, scores, max_far):
    """Select threshold on validation data only."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    if len(normal_scores) == 0 or len(anomaly_scores) == 0:
        raise ValueError(
            "Validation data must contain normal and anomalous rows."
        )

    candidates = np.unique(
        np.concatenate([
            normal_scores,
            anomaly_scores,
            [np.max(scores) + 1.0],
        ])
    )

    best = None

    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).ravel()

        far = fp / (fp + tn) if (fp + tn) else 0.0

        if far > max_far:
            continue

        precision = precision_score(
            y_true, predicted, zero_division=0
        )
        recall = recall_score(
            y_true, predicted, zero_division=0
        )
        f1 = f1_score(
            y_true, predicted, zero_division=0
        )

        candidate = {
            "threshold": float(threshold),
            "far": float(far),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

        if (
            best is None
            or candidate["f1"] > best["f1"]
            or (
                candidate["f1"] == best["f1"]
                and candidate["recall"] > best["recall"]
            )
        ):
            best = candidate

    if best is None:
        raise ValueError(
            "No threshold satisfies the validation false-alarm constraint."
        )

    return best


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read the existing configuration for reference and verify files exist.
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required_columns = {
        "timestamp", "network_mbps", "is_anomaly", "anomaly_type"
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Match the chronological split convention used by CloudPulse V3.
    usable_features = (
        df[["cpu_percent", "memory_percent", "network_mbps"]]
        .shift(1)
        .rolling(WINDOW, min_periods=WINDOW)
        .mean()
        .notna()
        .all(axis=1)
    )

    # Keep a common chronological split over usable rows.
    usable_indices = np.flatnonzero(usable_features.to_numpy())
    usable_count = len(usable_indices)

    train_end = int(usable_count * 0.70)
    val_end = int(usable_count * 0.85)

    feature_df = calculate_network_features(df)
    valid = feature_df["network_anomaly_score"].notna()

    data = pd.concat(
        [
            df[["timestamp", "is_anomaly", "anomaly_type", "network_mbps"]],
            feature_df,
        ],
        axis=1,
    ).loc[valid].reset_index(drop=True)

    # Use actual valid rows for the final split boundaries.
    usable_count = len(data)
    train_end = int(usable_count * 0.70)
    val_end = int(usable_count * 0.85)

    validation = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()

    y_val = validation["is_anomaly"].astype(int).to_numpy()
    val_scores = validation["network_anomaly_score"].to_numpy()

    selected = select_threshold(
        y_val, val_scores, MAX_VALIDATION_FAR
    )
    threshold = selected["threshold"]

    y_test = test["is_anomaly"].astype(int).to_numpy()
    test_scores = test["network_anomaly_score"].to_numpy()
    test_predictions = (test_scores >= threshold).astype(int)

    test_metrics = calculate_metrics(
        y_test, test_predictions, "test", threshold
    )

    val_predictions = (val_scores >= threshold).astype(int)
    validation_metrics = calculate_metrics(
        y_val, val_predictions, "validation", threshold
    )

    metrics = pd.DataFrame([
        validation_metrics,
        test_metrics,
    ])

    prediction_output = test[
        [
            "timestamp",
            "anomaly_type",
            "network_mbps",
            "network_baseline",
            "network_mad",
            "network_anomaly_score",
        ]
    ].copy()

    prediction_output["actual_anomaly"] = y_test
    prediction_output["predicted_anomaly"] = test_predictions
    prediction_output["threshold"] = threshold

    metrics_path = OUTPUT_DIR / "adaptive_network_metrics_v1.csv"
    predictions_path = OUTPUT_DIR / "adaptive_network_predictions_v1.csv"
    config_path = OUTPUT_DIR / "adaptive_network_config_v1.json"

    metrics.to_csv(metrics_path, index=False)
    prediction_output.to_csv(predictions_path, index=False)

    output_config = {
        "method": "rolling_median_mad",
        "window": WINDOW,
        "max_validation_false_alarm_rate": MAX_VALIDATION_FAR,
        "selected_threshold": threshold,
        "validation_metrics": selected,
        "test_rows": int(len(test)),
        "test_anomalies": int(np.sum(y_test)),
        "source_config_reference": CONFIG_PATH.name,
        "source_probability_threshold_reference": config.get(
            "model_probability_threshold"
        ),
        "note": (
            "Standalone adaptive network detector. "
            "Synthetic dataset; not real-world AWS validation."
        ),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2)

    print("\nADAPTIVE NETWORK DETECTOR V1")
    print(f"Rolling window: {WINDOW} observations")
    print(f"Maximum validation FAR: {MAX_VALIDATION_FAR:.2%}")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Validation FAR: {selected['far']:.4f}")
    print(f"Validation F1: {selected['f1']:.4f}")

    print("\nVALIDATION AND TEST METRICS")
    print(
        metrics.to_string(
            index=False, float_format="%.4f"
        )
    )

    print("\nSaved new files:")
    print(metrics_path)
    print(predictions_path)
    print(config_path)
    print("\nSynthetic-data results are not real-world validation.")


if __name__ == "__main__":
    main()
