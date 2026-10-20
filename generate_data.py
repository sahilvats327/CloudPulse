
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible simulated cloud metrics
np.random.seed(42)

NUM_ROWS = 10000

timestamps = pd.date_range(
    start="2026-01-01",
    periods=NUM_ROWS,
    freq="5min",
)

time_index = np.arange(NUM_ROWS)

# Normal daily workload patterns
daily_cycle = np.sin(2 * np.pi * time_index / 288)

cpu = 35 + 12 * daily_cycle + np.random.normal(0, 5, NUM_ROWS)
memory = 48 + 8 * daily_cycle + np.random.normal(0, 4, NUM_ROWS)
network = 120 + 35 * daily_cycle + np.random.normal(0, 15, NUM_ROWS)

cpu = np.clip(cpu, 1, 100)
memory = np.clip(memory, 1, 100)
network = np.clip(network, 1, 500)

# Labels: 0 = normal, 1 = anomaly
anomaly = np.zeros(NUM_ROWS, dtype=int)
anomaly_type = np.full(NUM_ROWS, "normal", dtype=object)

def mark_anomaly(start, end, kind):
    """Label an interval and apply its simulated pattern."""
    anomaly[start:end] = 1
    anomaly_type[start:end] = kind

    length = end - start

    if kind == "cpu_spike":
        # CPU becomes abnormal; other metrics stay unchanged.
        cpu[start:end] = np.random.uniform(85, 100, length)

    elif kind == "memory_drift":
        # Gradual memory growth, resembling a possible memory leak.
        baseline = memory[start]
        memory[start:end] = np.linspace(
            baseline + 5,
            96,
            length,
        )

    elif kind == "network_drop":
        # Sudden throughput loss; CPU and memory stay unchanged.
        network[start:end] = np.random.uniform(1, 10, length)

    elif kind == "combined":
        # Multiple metrics become abnormal together.
        cpu[start:end] = np.random.uniform(85, 100, length)
        memory[start:end] = np.random.uniform(80, 98, length)
        network[start:end] = np.random.uniform(300, 500, length)

# Training-period examples
mark_anomaly(1200, 1220, "cpu_spike")
mark_anomaly(3500, 3550, "memory_drift")
mark_anomaly(6100, 6120, "network_drop")

# Validation-period example
mark_anomaly(8200, 8240, "combined")

# Held-out test-period examples: all four patterns
mark_anomaly(9000, 9020, "cpu_spike")
mark_anomaly(9300, 9350, "memory_drift")
mark_anomaly(9500, 9530, "network_drop")
mark_anomaly(9800, 9820, "combined")

df = pd.DataFrame({
    "timestamp": timestamps,
    "cpu_percent": cpu,
    "memory_percent": memory,
    "network_mbps": network,
    "is_anomaly": anomaly,
    "anomaly_type": anomaly_type,
})

output_path = Path("data/cloud_metrics.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("CloudPulse dataset created successfully!")
print(f"Rows: {len(df)}")
print(f"Anomalous rows: {int(df['is_anomaly'].sum())}")
print(f"Normal rows: {int((df['is_anomaly'] == 0).sum())}")
print("\nAnomaly counts by type:")
print(df.loc[df["is_anomaly"] == 1, "anomaly_type"].value_counts())
print(f"\nSaved to: {output_path.resolve()}")
print("\nFirst five rows:")
print(df.head())
