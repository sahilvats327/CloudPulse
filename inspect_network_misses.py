
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "outputs" / "context_network_test_predictions.csv"

df = pd.read_csv(path)

network = df[df["anomaly_type"] == "network_drop"].copy()
normal = df[df["is_anomaly"] == 0].copy()

print("\nNETWORK-DROP DETECTION")
print("=" * 65)

for label, group in [
    ("Detected network drops", network[network["context_prediction"] == 1]),
    ("Missed network drops", network[network["context_prediction"] == 0]),
    ("Normal observations", normal),
]:
    print(f"\n{label}: {len(group)} rows")

    if not group.empty:
        print(
            group[["network_mbps", "network_baseline", "network_ratio",
                   "relative_drop"]]
            .describe()
            .round(3)
            .to_string()
        )

print("\nABSOLUTE THROUGHPUT CHECK")
print("=" * 65)

for label, group in [
    ("Detected network drops", network[network["context_prediction"] == 1]),
    ("Missed network drops", network[network["context_prediction"] == 0]),
    ("Normal observations", normal),
]:
    if not group.empty:
        print(
            f"{label}: "
            f"below 10 Mbps = {(group['network_mbps'] < 10).sum()}/{len(group)}; "
            f"below 22.3734 Mbps = "
            f"{(group['network_mbps'] < 22.3734).sum()}/{len(group)}"
        )
