
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data" / "cloud_metrics.csv")

print("\nNETWORK THROUGHPUT BY ANOMALY TYPE")
print("=" * 65)

summary = df.groupby("anomaly_type")["network_mbps"].agg(
    ["count", "min", "median", "max", "mean"]
).round(2)

print(summary.to_string())

print("\nLOW-THROUGHPUT CHECK: network_mbps < 10")
print("=" * 65)

df["low_throughput"] = df["network_mbps"] < 10

for anomaly_type, group in df.groupby("anomaly_type"):
    low_count = int(group["low_throughput"].sum())
    total = len(group)

    print(
        f"{anomaly_type:15s}: "
        f"{low_count:4d}/{total:<5d} "
        f"({low_count / total:.2%})"
    )

normal = df[df["is_anomaly"] == 0]
drops = df[df["anomaly_type"] == "network_drop"]

print("\nSUMMARY")
print(f"Normal rows below 10 Mbps: {int((normal['network_mbps'] < 10).sum())}")
print(f"Network-drop rows below 10 Mbps: {int((drops['network_mbps'] < 10).sum())}")
print(f"Total normal rows: {len(normal)}")
print(f"Total network-drop rows: {len(drops)}")
