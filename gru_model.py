
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = Path("data/cloud_metrics.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["cpu_percent", "memory_percent", "network_mbps"]
WINDOW_SIZE = 12
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# 1. Load chronological data
df = pd.read_csv(DATA_PATH)

values = df[FEATURES].to_numpy(dtype=np.float32)
labels = df["is_anomaly"].to_numpy(dtype=np.int32)

n = len(df)
train_end = int(n * TRAIN_RATIO)
validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

# 2. Fit scaler on normal training observations only
normal_train = labels[:train_end] == 0

scaler = StandardScaler()
scaler.fit(values[:train_end][normal_train])

scaled = scaler.transform(values).astype(np.float32)

# 3. Create normal sequences for training and validation
def make_sequences(data, target_labels, start, end):
    X, y = [], []

    for i in range(max(start, WINDOW_SIZE), end):
        if target_labels[i - WINDOW_SIZE:i + 1].any():
            continue

        X.append(data[i - WINDOW_SIZE:i])
        y.append(data[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )

X_train, y_train = make_sequences(
    scaled, labels, WINDOW_SIZE, train_end
)

X_val, y_val = make_sequences(
    scaled, labels, train_end, validation_end
)

if len(X_train) == 0 or len(X_val) == 0:
    raise RuntimeError("Not enough normal training or validation data.")

# 4. Build the GRU network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, len(FEATURES))),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(FEATURES)),
])

model.compile(
    optimizer="adam",
    loss="mse",
)

# 5. Train
print("Training GRU model...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    shuffle=False,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

# 6. Choose threshold from normal validation errors
val_predictions = model.predict(X_val, verbose=0)
val_errors = np.mean((val_predictions - y_val) ** 2, axis=1)

threshold = float(np.quantile(val_errors, 0.99))

# 7. Save GRU model and configuration
model.save(MODEL_DIR / "gru_model.keras")

joblib.dump(scaler, MODEL_DIR / "gru_scaler.pkl")

with open(MODEL_DIR / "gru_config.json", "w") as f:
    json.dump({
        "features": FEATURES,
        "window_size": WINDOW_SIZE,
        "threshold": threshold,
        "train_end": train_end,
        "validation_end": validation_end,
    }, f, indent=2)

# 8. Evaluate every test timestamp
test_start = validation_end

eligible_indices = np.arange(
    max(test_start, WINDOW_SIZE),
    n,
    dtype=np.int32,
)

X_test = np.asarray([
    scaled[i - WINDOW_SIZE:i]
    for i in eligible_indices
], dtype=np.float32)

actual_scaled = scaled[eligible_indices]

test_predictions = model.predict(X_test, verbose=0)
test_errors = np.mean(
    (test_predictions - actual_scaled) ** 2,
    axis=1,
)

y_true = labels[eligible_indices]
y_pred = (test_errors > threshold).astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

results = {
    "model": "GRU",
    "test_observations": int(len(eligible_indices)),
    "actual_anomalies": int(y_true.sum()),
    "threshold": threshold,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "precision": float(precision_score(
        y_true, y_pred, zero_division=0
    )),
    "recall": float(recall_score(
        y_true, y_pred, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_true, y_pred, zero_division=0
    )),
    "confusion_matrix": cm.tolist(),
}

with open(OUTPUT_DIR / "gru_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nGRU evaluation:")
print(json.dumps(results, indent=2))
print("\nSaved model: models/gru_model.keras")
print("Saved metrics: outputs/gru_metrics.json")
