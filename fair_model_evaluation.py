
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "cloud_metrics.csv"
MODELS = BASE / "models"
OUTPUTS = BASE / "outputs"

FEATURES = ["cpu_percent", "memory_percent", "network_mbps"]
WINDOW_SIZE = 12


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_row(name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
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


def main():
    OUTPUTS.mkdir(exist_ok=True)

    # Load the original chronological dataset.
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    values = df[FEATURES].to_numpy(dtype=np.float32)
    labels = df["is_anomaly"].to_numpy(dtype=np.int32)

    # Match the exact timestamps in the saved XGBoost V3 test file.
    xgb_path = OUTPUTS / "xgboost_test_predictions_v3.csv"
    xgb = pd.read_csv(xgb_path)
    xgb["timestamp"] = pd.to_datetime(xgb["timestamp"])

    timestamp_to_index = {
        timestamp: i for i, timestamp in enumerate(df["timestamp"])
    }

    test_indices = np.array([
        timestamp_to_index[timestamp]
        for timestamp in xgb["timestamp"]
    ], dtype=np.int32)

    actual = labels[test_indices]

    if not np.array_equal(actual, xgb["actual_anomaly"].to_numpy()):
        raise ValueError(
            "Dataset labels do not match the XGBoost V3 prediction file."
        )

    print("Shared test timestamps:", len(test_indices))
    print("First:", df.loc[test_indices[0], "timestamp"])
    print("Last: ", df.loc[test_indices[-1], "timestamp"])
    print("Actual anomalies:", int(actual.sum()))

    results = []
    prediction_frames = []

    # Load saved XGBoost V3 predictions; do not retrain.
    xgb_model_pred = xgb["xgboost_prediction"].to_numpy(dtype=int)
    hybrid_pred = xgb["hybrid_prediction"].to_numpy(dtype=int)

    results.append(metrics_row(
        "XGBoost V3 Model Only", actual, xgb_model_pred
    ))
    results.append(metrics_row(
        "CloudPulse V3 Hybrid", actual, hybrid_pred
    ))

    prediction_frames.append(pd.DataFrame({
        "timestamp": xgb["timestamp"],
        "actual_anomaly": actual,
        "model": "XGBoost V3 Model Only",
        "prediction": xgb_model_pred,
    }))
    prediction_frames.append(pd.DataFrame({
        "timestamp": xgb["timestamp"],
        "actual_anomaly": actual,
        "model": "CloudPulse V3 Hybrid",
        "prediction": hybrid_pred,
    }))

    # LSTM and GRU use their original scalers, models, and thresholds.
    for name in ["lstm", "gru"]:
        config = load_json(MODELS / f"{name}_config.json")
        scaler_path = (
            MODELS / ("scaler.pkl" if name == "lstm" else "gru_scaler.pkl")
        )
        model_path = MODELS / f"{name}_model.keras"

        scaler = joblib.load(scaler_path)
        model = tf.keras.models.load_model(model_path, compile=False)
        scaled = scaler.transform(values).astype(np.float32)

        # Predict the same test timestamps as XGBoost V3.
        # Historical observations are used only as model input.
        X_test = np.asarray([
            scaled[i - WINDOW_SIZE:i]
            for i in test_indices
        ], dtype=np.float32)

        if len(X_test) != len(actual):
            raise ValueError(f"{name}: test input length mismatch.")

        predicted_values = model.predict(X_test, verbose=0)
        actual_values = scaled[test_indices]

        errors = np.mean(
            (predicted_values - actual_values) ** 2,
            axis=1,
        )

        threshold = float(config["threshold"])
        predictions = (errors > threshold).astype(int)

        results.append(metrics_row(name.upper(), actual, predictions))

        prediction_frames.append(pd.DataFrame({
            "timestamp": xgb["timestamp"],
            "actual_anomaly": actual,
            "model": name.upper(),
            "prediction": predictions,
            "prediction_error": errors,
            "threshold": threshold,
        }))

        print(
            f"{name.upper()}: loaded saved model; "
            f"threshold={threshold:.6f}"
        )

    # ARIMA: load saved predictions; do not refit or retrain.
    arima_path = OUTPUTS / "arima_network_predictions_v1.csv"
    arima = pd.read_csv(arima_path)
    arima["timestamp"] = pd.to_datetime(arima["timestamp"])

    # Use only ARIMA's held-out test rows.
    arima_test = arima.loc[arima["split"] == "test"].copy()

    # Validate timestamps and labels before aligning predictions.
    if arima_test["timestamp"].duplicated().any():
        raise ValueError("ARIMA test predictions contain duplicate timestamps.")

    arima_lookup = arima_test.set_index("timestamp")

    missing = [
        timestamp for timestamp in xgb["timestamp"]
        if timestamp not in arima_lookup.index
    ]
    if missing:
        raise ValueError(
            f"ARIMA is missing {len(missing)} shared test timestamps."
        )

    aligned_arima = arima_lookup.loc[xgb["timestamp"]]

    if not np.array_equal(
        actual,
        aligned_arima["is_anomaly"].to_numpy(dtype=np.int32),
    ):
        raise ValueError(
            "ARIMA labels do not match the shared dataset labels."
        )

    arima_pred = aligned_arima["predicted_anomaly"].to_numpy(dtype=int)

    results.append(metrics_row("ARIMA Network Detector V1", actual, arima_pred))

    prediction_frames.append(pd.DataFrame({
        "timestamp": xgb["timestamp"],
        "actual_anomaly": actual,
        "model": "ARIMA Network Detector V1",
        "prediction": arima_pred,
        "prediction_error": aligned_arima["absolute_error"].to_numpy(),
        "threshold": aligned_arima["threshold"].to_numpy(),
    }))

    print(
        "ARIMA: loaded saved test predictions; "
        f"aligned rows={len(arima_pred)}"
    )


    # Save new files only.
    comparison = pd.DataFrame(results)
    comparison.to_csv(
        OUTPUTS / "fair_model_comparison.csv", index=False
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        OUTPUTS / "fair_model_predictions.csv", index=False
    )

    with open(
        OUTPUTS / "fair_model_comparison.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    print("\nFAIR TEST-SET COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nNew files saved:")
    print("outputs/fair_model_comparison.csv")
    print("outputs/fair_model_comparison.json")
    print("outputs/fair_model_predictions.csv")
    print("\nOriginal model and metrics files were not modified.")


if __name__ == "__main__":
    main()
