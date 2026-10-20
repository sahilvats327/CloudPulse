
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent

data = pd.read_csv(BASE_DIR / "data" / "cloud_metrics.csv")
pred = pd.read_csv(
    BASE_DIR / "outputs" / "xgboost_test_predictions_v2.csv"
)

data["timestamp"] = pd.to_datetime(data["timestamp"])
pred["timestamp"] = pd.to_datetime(pred["timestamp"])

results = pred.merge(
    data[["timestamp", "anomaly_type"]],
    on="timestamp",
    how="left",
    validate="one_to_one",
)

print("\nXGBOOST V2 — RESULTS BY ANOMALY TYPE")
print("=" * 65)

for anomaly_type, group in results[
    results["actual_anomaly"] == 1
].groupby("anomaly_type"):
    actual = group["actual_anomaly"]
    predicted = group["predicted_anomaly"]

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    print(f"\nType: {anomaly_type}")
    print(f"Actual anomalies: {len(group)}")
    print(f"Detected: {tp}")
    print(f"Missed: {fn}")
    print(
        f"Recall: {recall_score(actual, predicted, zero_division=0):.2%}"
    )
    print(
        f"Precision: {precision_score(actual, predicted, zero_division=0):.2%}"
    )
    print(
        f"F1 score: {f1_score(actual, predicted, zero_division=0):.2%}"
    )

print("\nDiagnostic completed.")
