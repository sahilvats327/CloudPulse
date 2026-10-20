
import pandas as pd
from pathlib import Path

file_path = Path("data/cloud_metrics.csv")
df = pd.read_csv(file_path, parse_dates=["timestamp"])

print("=" * 50)
print("CLOUDPULSE DATASET VALIDATION")
print("=" * 50)

# 1. Dataset structure
print("\n1. Dataset shape:")
print(df.shape)

print("\n2. Missing values:")
print(df.isnull().sum())

# 2. Duplicate and timestamp checks
print("\n3. Duplicate rows:", df.duplicated().sum())
print(
    "Timestamps ordered:",
    df["timestamp"].is_monotonic_increasing
)
print(
    "Duplicate timestamps:",
    df["timestamp"].duplicated().sum()
)

# 3. Numeric ranges
metrics = [
    "cpu_percent",
    "memory_percent",
    "network_mbps",
]

print("\n4. Metric summary:")
print(df[metrics].describe().round(2))

print("\n5. Range checks:")
print("CPU within 0-100:",
      df["cpu_percent"].between(0, 100).all())
print("Memory within 0-100:",
      df["memory_percent"].between(0, 100).all())
print("Network non-negative:",
      (df["network_mbps"] >= 0).all())

# 4. Anomaly labels
print("\n6. Anomaly labels:")
print(df["is_anomaly"].value_counts().sort_index())

print("\n7. Anomaly counts by type:")
print(df["anomaly_type"].value_counts())

print("\n8. Label consistency:")
print(
    "Normal rows labeled normal:",
    (df.loc[df["is_anomaly"] == 0, "anomaly_type"] == "normal").all()
)
print(
    "Anomaly rows have anomaly types:",
    (df.loc[df["is_anomaly"] == 1, "anomaly_type"] != "normal").all()
)

print("\nValidation finished.")
