
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = Path("data/cloud_metrics.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

FEATURES = [
    "cpu_percent",
    "memory_percent",
    "network_mbps",
]

TEST_RATIO_START = 0.85


def evaluate_model(model_name, model_file, scaler_file, config_file):
    print(f"\nEvaluating {model_name}...")

    df = pd.read_csv(DATA_PATH)
    values = df[FEATURES].to_numpy(dtype=np.float32)
    labels = df["is_anomaly"].to_numpy(dtype=np.int32)
    anomaly_types = df["anomaly_type"].to_numpy()

    with open(MODEL_DIR / config_file, "r") as f:
        config = json.load(f)

    model = tf.keras.models.load_model(MODEL_DIR / model_file)
    scaler = joblib.load(MODEL_DIR / scaler_file)

    scaled = scaler.transform(values).astype(np.float32)

    window_size = int(config["window_size"])
    threshold = float(config["threshold"])
    test_start = int(len(df) * TEST_RATIO_START)

    indices = np.arange(
        max(test_start, window_size),
        len(df),
    )

    X_test = np.asarray([
        scaled[i - window_size:i]
        for i in indices
    ], dtype=np.float32)

    predictions = model.predict(X_test, verbose=0)
    actual_values = scaled[indices]

    errors = np.mean(
        (predictions - actual_values) ** 2,
        axis=1,
    )

    y_true = labels[indices]
    y_pred = (errors > threshold).astype(int)
    test_types = anomaly_types[indices]

    # Evaluate each anomaly category against all other test rows.
    rows = []

    for anomaly_type in [
        "cpu_spike",
        "memory_drift",
        "network_drop",
        "combined",
    ]:
        actual = (test_types == anomaly_type).astype(int)
        detected = y_pred

        tp = int(((actual == 1) & (detected == 1)).sum())
        fn = int(((actual == 1) & (detected == 0)).sum())
        fp = int(((actual == 0) & (detected == 1)).sum())
        tn = int(((actual == 0) & (detected == 0)).sum())

        rows.append({
            "model": model_name,
            "anomaly_type": anomaly_type,
            "actual_count": int(actual.sum()),
            "detected_count": tp,
            "missed_count": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "precision": precision_score(
                actual, detected, zero_division=0
            ),
            "recall": recall_score(
                actual, detected, zero_division=0
            ),
            "f1_score": f1_score(
                actual, detected, zero_division=0
            ),
        })

    return rows


all_results = []

all_results.extend(evaluate_model(
    "LSTM",
    "lstm_model.keras",
    "scaler.pkl",
    "lstm_config.json",
))

all_results.extend(evaluate_model(
    "GRU",
    "gru_model.keras",
    "gru_scaler.pkl",
    "gru_config.json",
))

results_df = pd.DataFrame(all_results)

OUTPUT_DIR.mkdir(exist_ok=True)

csv_path = OUTPUT_DIR / "anomaly_type_metrics.csv"
results_df.to_csv(csv_path, index=False)

display_columns = [
    "model",
    "anomaly_type",
    "actual_count",
    "detected_count",
    "missed_count",
    "precision",
    "recall",
    "f1_score",
]

print("\nCloudPulse: Performance by anomaly type")
print(
    results_df[display_columns].to_string(
        index=False,
        formatters={
            "precision": "{:.2%}".format,
            "recall": "{:.2%}".format,
            "f1_score": "{:.2%}".format,
        },
    )
)

print(f"\nSaved results: {csv_path.resolve()}")
