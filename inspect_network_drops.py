
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

data = pd.read_csv(BASE_DIR / "data" / "cloud_metrics.csv")
pred = pd.read_csv(
    BASE_DIR / "outputs" / "xgboost_test_predictions_v2.csv"
)

data["timestamp"] = pd.to_datetime(data["timestamp"])
pred["timestamp"] = pd.to_datetime(pred["timestamp"])

df = pred.merge(
    data[["timestamp", "network_mbps", "anomaly_type"]],
    on="timestamp",
    how="left",
    validate="one_to_one",
)

drops = df[df["anomaly_type"] == "network_drop"].copy()
normal = df[df["actual_anomaly"] == 0].copy()

print("\nNETWORK-DROP PROBABILITY DIAGNOSTIC")
print("=" * 60)
print(f"Network-drop test rows: {len(drops)}")
print(f"Detected: {int(drops['predicted_anomaly'].sum())}")
print(f"Missed: {int((drops['predicted_anomaly'] == 0).sum())}")

print("\nNetwork-drop probability distribution:")
print(drops["anomaly_probability"].describe().to_string())

print("\nNormal-row probability distribution:")
print(normal["anomaly_probability"].describe().to_string())

print("\nNetwork-drop rows with lowest probabilities:")
print(
    drops.sort_values("anomaly_probability")[
        ["timestamp", "network_mbps", "anomaly_probability",
         "predicted_anomaly"]
    ].head(10).to_string(index=False)
)

print("\nNetwork-drop rows with highest probabilities:")
print(
    drops.sort_values("anomaly_probability", ascending=False)[
        ["timestamp", "network_mbps", "anomaly_probability",
         "predicted_anomaly"]
    ].head(10).to_string(index=False)
)

print("\nDiagnostic completed.")
