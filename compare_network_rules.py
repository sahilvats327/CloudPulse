
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(
    BASE_DIR / "outputs" / "context_network_test_predictions.csv"
)

# Absolute threshold learned previously from training data.
absolute_threshold = 22.3734111

# Relative threshold learned previously from validation data.
relative_threshold = 0.331064

df["absolute_prediction"] = (
    df["network_mbps"] < absolute_threshold
).astype(int)

df["relative_prediction"] = (
    df["relative_drop"] >= relative_threshold
).astype(int)

df["combined_prediction"] = (
    (df["absolute_prediction"] == 1)
    | (df["relative_prediction"] == 1)
).astype(int)


def evaluate(name, predictions):
    y_true = df["is_anomaly"].astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("-" * 50)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_true, predictions, zero_division=0):.4f}")
    print(f"False alarm rate: {fp / (tn + fp):.4f}")


print("NETWORK RULE COMPARISON")
print("=" * 50)

evaluate("Absolute threshold only", df["absolute_prediction"])
evaluate("Relative threshold only", df["relative_prediction"])
evaluate("Absolute OR relative", df["combined_prediction"])

print("\nNetwork-drop-specific results")
print("=" * 50)

drops = df[df["anomaly_type"] == "network_drop"]

for name, column in [
    ("Absolute", "absolute_prediction"),
    ("Relative", "relative_prediction"),
    ("Combined", "combined_prediction"),
]:
    detected = int(drops[column].sum())
    print(f"{name}: detected {detected}/{len(drops)} network drops")
