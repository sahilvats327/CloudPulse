
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
CHARTS = OUT / "visualizations"
CHARTS.mkdir(parents=True, exist_ok=True)

# 1. Main model comparison
metrics = pd.read_csv(OUT / "cloudpulse_unified_metrics.csv")

names = metrics["model"].tolist()
x = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, metrics["precision"], width, label="Precision")
ax.bar(x, metrics["recall"], width, label="Recall")
ax.bar(x + width, metrics["f1_score"], width, label="F1 Score")

ax.set_ylabel("Score")
ax.set_ylim(0, 1.08)
ax.set_title("CloudPulse: Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(CHARTS / "model_performance_comparison.png", dpi=200)
plt.close(fig)

# 2. Confusion matrices for the common comparison models
fair = pd.read_csv(OUT / "fair_model_comparison.csv")
model_names = fair["model"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, (_, row) in zip(axes, fair.iterrows()):
    matrix = np.array([
        [int(row["tn"]), int(row["fp"])],
        [int(row["fn"]), int(row["tp"])]
    ])

    ax.imshow(matrix, cmap="Blues")
    ax.set_title(row["model"])
    ax.set_xticks([0, 1], labels=["Predicted Normal", "Predicted Anomaly"])
    ax.set_yticks([0, 1], labels=["Actual Normal", "Actual Anomaly"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center", color="black")

fig.suptitle("CloudPulse: Confusion Matrices")
fig.tight_layout()
fig.savefig(CHARTS / "confusion_matrices.png", dpi=200)
plt.close(fig)

# 3. Anomaly-type detections: XGBoost vs Hybrid
types = pd.read_csv(OUT / "xgboost_v3_anomaly_type_metrics.csv")
types = types[types["anomaly_type"] != "normal"].copy()

x = np.arange(len(types))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(
    x - width / 2, types["xgboost_detected"], width,
    label="XGBoost V3"
)
ax.bar(
    x + width / 2, types["hybrid_detected"], width,
    label="CloudPulse V3 Hybrid"
)

ax.set_ylabel("Detected anomalies")
ax.set_title("Anomaly Detection by Type")
ax.set_xticks(x)
ax.set_xticklabels(types["anomaly_type"], rotation=15, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(CHARTS / "anomaly_type_comparison.png", dpi=200)
plt.close(fig)

print("Charts created in:", CHARTS)
for file in sorted(CHARTS.glob("*.png")):
    print(f"{file.name}: {file.stat().st_size:,} bytes")
