
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the generated dataset
df = pd.read_csv("data/cloud_metrics.csv", parse_dates=["timestamp"])

# Create an output folder
Path("outputs").mkdir(exist_ok=True)

metrics = [
    ("cpu_percent", "CPU Utilization (%)"),
    ("memory_percent", "Memory Utilization (%)"),
    ("network_mbps", "Network Traffic (Mbps)")
]

for column, title in metrics:
    plt.figure(figsize=(14, 5))

    plt.plot(
        df["timestamp"],
        df[column],
        label=title,
        linewidth=0.8
    )

    anomalies = df["is_anomaly"] == 1

    plt.scatter(
        df.loc[anomalies, "timestamp"],
        df.loc[anomalies, column],
        color="red",
        s=12,
        label="Injected anomalies"
    )

    plt.title(title + " Over Time")
    plt.xlabel("Time")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = column.replace("_percent", "").replace("_mbps", "")
    output_file = Path("outputs") / f"{filename}_usage.png"

    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved graph: {output_file}")

print("\nAll visualizations created successfully!")
