
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cloud_metrics.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Candidate thresholds are calibrated using clean validation observations.
QUANTILES = [0.95, 0.97, 0.98, 0.985, 0.99, 0.995]
MAX_VALIDATION_FPR = 0.02  # Target: no more than 2% false alarms


def get_value(config, possible_keys, default=None):
    for key in possible_keys:
        if key in config:
            return config[key]
    return default


def calculate_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    return {
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            labels, predictions, zero_division=0
        ),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def evaluate_model(model_name, data, features, labels):
    config_path = MODELS_DIR / f"{model_name.lower()}_config.json"
    model_path = MODELS_DIR / f"{model_name.lower()}_model.keras"
    scaler_path = MODELS_DIR / (
        "scaler.pkl" if model_name == "LSTM" else "gru_scaler.pkl"
    )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    window_size = int(
        get_value(config, ["window_size", "sequence_length"], 12)
    )
    train_end = int(
        get_value(config, ["train_end", "train_end_index"], 7000)
    )
    validation_end = int(
        get_value(
            config,
            ["validation_end", "val_end", "validation_end_index"],
            8500,
        )
    )

    current_threshold = float(config["threshold"])

    scaler = joblib.load(scaler_path)
    model = keras.models.load_model(model_path)

    scaled = scaler.transform(data[features].to_numpy(dtype=float))

    # Score each timestamp using the preceding window.
    # Validation and test predictions are evaluated separately.
    start_index = max(train_end, window_size)
    errors = np.full(len(data), np.nan)

    for i in range(start_index, len(data)):
        sequence = scaled[i - window_size:i]
        prediction = model.predict(
            sequence[np.newaxis, :, :], verbose=0
        )[0]
        errors[i] = float(np.mean((prediction - scaled[i]) ** 2))

    # Validation observations: train_end through validation_end - 1.
    val_indices = np.arange(train_end, validation_end)
    val_indices = val_indices[
        (val_indices >= window_size) & (val_indices < len(data))
    ]

    val_errors = errors[val_indices]
    val_labels = labels[val_indices]

    # Use only normal validation sequences to calibrate thresholds.
    # Exclude a target if its preceding window contains any anomaly too,
    # matching the clean-sequence rule used during model training.
    clean_normal_mask = np.array([
        labels[i - window_size:i + 1].sum() == 0
        for i in val_indices
    ])

    clean_errors = val_errors[clean_normal_mask]

    if len(clean_errors) == 0:
        raise ValueError(
            f"{model_name}: no clean normal validation errors found."
        )

    candidate_thresholds = {
        float(np.quantile(clean_errors, q))
        for q in QUANTILES
    }
    candidate_thresholds.add(current_threshold)

    rows = []
    for threshold in sorted(candidate_thresholds):
        predictions = (val_errors > threshold).astype(int)
        metrics = calculate_metrics(val_labels, predictions)

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"],
            "validation_f1": metrics["f1_score"],
            "validation_false_alarms": metrics["false_positives"],
            "validation_missed": metrics["false_negatives"],
            "validation_fpr": metrics["false_alarm_rate"],
        })

    results = pd.DataFrame(rows)

    # Select using validation data only. Prefer the highest F1 among
    # thresholds meeting the 2% false-alarm target.
    eligible = results[
        results["validation_fpr"] <= MAX_VALIDATION_FPR
    ]

    if eligible.empty:
        selected = results.sort_values(
            ["validation_f1", "threshold"],
            ascending=[False, False],
        ).iloc[0]
        selection_note = (
            "No candidate met the 2% validation false-alarm target; "
            "selected the best validation F1."
        )
    else:
        selected = eligible.sort_values(
            ["validation_f1", "threshold"],
            ascending=[False, False],
        ).iloc[0]
        selection_note = (
            "Selected the highest validation F1 among candidates "
            "with validation false-alarm rate <= 2%."
        )

    selected_threshold = float(selected["threshold"])

    # Evaluate the selected threshold on the untouched test split.
    test_indices = np.arange(validation_end, len(data))
    test_indices = test_indices[
        (test_indices >= window_size) & np.isfinite(errors[test_indices])
    ]

    test_labels = labels[test_indices]
    test_errors = errors[test_indices]

    current_test = calculate_metrics(
        test_labels, (test_errors > current_threshold).astype(int)
    )
    selected_test = calculate_metrics(
        test_labels, (test_errors > selected_threshold).astype(int)
    )

    print(f"\n{'=' * 65}")
    print(f"{model_name}: THRESHOLD TUNING")
    print(f"{'=' * 65}")
    print(f"Current threshold:   {current_threshold:.6f}")
    print(f"Selected threshold:  {selected_threshold:.6f}")
    print(selection_note)

    print("\nValidation results (used for threshold selection):")
    display_cols = [
        "threshold",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_false_alarms",
        "validation_missed",
        "validation_fpr",
    ]
    print(
        results[display_cols].to_string(
            index=False,
            formatters={
                "threshold": "{:.6f}".format,
                "validation_precision": "{:.2%}".format,
                "validation_recall": "{:.2%}".format,
                "validation_f1": "{:.2%}".format,
                "validation_fpr": "{:.2%}".format,
            },
        )
    )

    print("\nHeld-out test comparison (not used for selection):")
    for label, metrics in [
        ("Current", current_test),
        ("Selected", selected_test),
    ]:
        print(
            f"{label:8} | "
            f"Precision: {metrics['precision']:.2%} | "
            f"Recall: {metrics['recall']:.2%} | "
            f"F1: {metrics['f1_score']:.2%} | "
            f"False alarms: {metrics['false_positives']} | "
            f"Missed: {metrics['false_negatives']} | "
            f"FPR: {metrics['false_alarm_rate']:.2%}"
        )

    summary = {
        "model": model_name,
        "current_threshold": current_threshold,
        "selected_threshold": selected_threshold,
        "selection_note": selection_note,
        "validation": {
            key: (
                float(selected[key])
                if key in selected and pd.notna(selected[key])
                else None
            )
            for key in [
                "validation_precision",
                "validation_recall",
                "validation_f1",
                "validation_false_alarms",
                "validation_missed",
                "validation_fpr",
            ]
        },
        "test_current_threshold": current_test,
        "test_selected_threshold": selected_test,
    }

    return results, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    features = ["cpu_percent", "memory_percent", "network_mbps"]
    labels = data["is_anomaly"].to_numpy(dtype=int)

    all_results = []
    summaries = []

    for model_name in ["LSTM", "GRU"]:
        results, summary = evaluate_model(
            model_name, data, features, labels
        )
        all_results.append(results)
        summaries.append(summary)

    csv_path = OUTPUT_DIR / "threshold_tuning_results.csv"
    json_path = OUTPUT_DIR / "threshold_tuning_summary.json"

    pd.concat(all_results, ignore_index=True).to_csv(
        csv_path, index=False
    )

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print(f"\nDetailed results saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")
    print("\nNo model weights or configuration files were changed.")


if __name__ == "__main__":
    main()
