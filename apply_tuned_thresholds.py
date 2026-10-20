
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
OUTPUTS = BASE / "outputs"

summary_path = OUTPUTS / "threshold_tuning_summary.json"

with open(summary_path, "r", encoding="utf-8") as f:
    summaries = json.load(f)

# Read and validate all files before modifying anything.
updates = []

for result in summaries:
    name = result["model"]
    model_key = name.lower()

    config_path = MODELS / f"{model_key}_config.json"
    metrics_path = OUTPUTS / f"{model_key}_metrics.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    selected_threshold = float(result["selected_threshold"])
    test = result["test_selected_threshold"]

    required = [
        "precision", "recall", "f1_score",
        "true_positives", "true_negatives",
        "false_positives", "false_negatives",
        "false_alarm_rate",
    ]
    missing = [key for key in required if key not in test]
    if missing:
        raise ValueError(f"{name}: missing test metrics: {missing}")

    # Prepare updated copies.
    new_config = dict(config)
    new_config["threshold"] = selected_threshold
    new_config["threshold_tuning"] = {
        "method": "validation F1 with false-alarm rate <= 2%",
        "selected_threshold": selected_threshold,
        "test_data_used_for_selection": False,
    }

    new_metrics = dict(metrics)
    new_metrics.update({
        "threshold": selected_threshold,
        "true_positives": int(test["true_positives"]),
        "true_negatives": int(test["true_negatives"]),
        "false_positives": int(test["false_positives"]),
        "false_negatives": int(test["false_negatives"]),
        "precision": float(test["precision"]),
        "recall": float(test["recall"]),
        "f1_score": float(test["f1_score"]),
        "false_alarm_rate": float(test["false_alarm_rate"]),
        "confusion_matrix": [
            [int(test["true_negatives"]), int(test["false_positives"])],
            [int(test["false_negatives"]), int(test["true_positives"])],
        ],
    })

    updates.append((name, config_path, metrics_path, new_config, new_metrics))

# Save configuration and metrics only after validation succeeds.
for name, config_path, metrics_path, config, metrics in updates:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{name} updated successfully.")
    print(f"  Threshold: {metrics['threshold']:.6f}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-score:  {metrics['f1_score']:.2%}")
    print(f"  False alarms: {metrics['false_positives']}")
    print(f"  Missed anomalies: {metrics['false_negatives']}")

print("\nThresholds and saved test metrics updated.")
print("Model weights were not retrained or modified.")
