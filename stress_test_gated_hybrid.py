
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cloud_metrics.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model_v3.json"
CONFIG_PATH = BASE_DIR / "models" / "xgboost_config_v3.json"
OUTPUT_DIR = BASE_DIR / "outputs"

# Fixed seed makes the synthetic stress tests reproducible.
RNG = np.random.default_rng(42)

METRICS = ["cpu_percent", "memory_percent", "network_mbps"]
LAGS = [1, 3, 6, 12]
WINDOWS = [3, 6, 12]

# Frozen gate threshold from the previous gated-hybrid experiment.
GATE_THRESHOLD = 0.38


def build_features(df):
    """Reproduce the feature engineering used by XGBoost V3."""
    features = df[METRICS].copy()

    for column in METRICS:
        for lag in LAGS:
            features[f"{column}_lag_{lag}"] = df[column].shift(lag)

        past = df[column].shift(1)

        for window in WINDOWS:
            features[f"{column}_mean_{window}"] = (
                past.rolling(window).mean()
            )
            features[f"{column}_std_{window}"] = (
                past.rolling(window).std()
            )

    for window in WINDOWS:
        baseline = (
            df["network_mbps"].shift(1).rolling(window).mean()
        )
        features[f"network_ratio_{window}"] = (
            df["network_mbps"] / baseline.clip(lower=1.0)
        )
        features[f"network_drop_from_baseline_{window}"] = (
            baseline - df["network_mbps"]
        )

    return features


def metrics_for(name, actual, predicted):
    """Calculate binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "scenario": name,
        "test_rows": int(len(actual)),
        "actual_anomalies": int(np.sum(actual)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "false_alarm_rate": float(
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
    }


def make_scenario(original, scenario):
    """Modify test-period network values only; retain original labels."""
    df = original.copy()
    usable = build_features(df).notna().all(axis=1)
    usable_indices = np.flatnonzero(usable.to_numpy())

    n = int(usable.sum())
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    test_raw_indices = usable_indices[val_end:]

    # Only perturb observations in the chronological test period.
    test = df.loc[test_raw_indices]
    normal_indices = test.index[
        test["anomaly_type"].eq("normal")
    ].to_numpy()
    drop_indices = test.index[
        test["anomaly_type"].eq("network_drop")
    ].to_numpy()

    if scenario == "baseline":
        pass

    elif scenario == "low_normal_traffic":
        # 20% of normal test observations get low throughput.
        chosen = RNG.choice(
            normal_indices,
            size=max(1, int(len(normal_indices) * 0.20)),
            replace=False,
        )
        df.loc[chosen, "network_mbps"] = RNG.uniform(
            12.0, 22.0, size=len(chosen)
        )

    elif scenario == "mild_network_drops":
        # Network anomalies are less severe than in the original data.
        df.loc[drop_indices, "network_mbps"] = RNG.uniform(
            15.0, 30.0, size=len(drop_indices)
        )

    elif scenario == "overlapping_traffic":
        # Normal low traffic and anomalous drops share a range.
        chosen = RNG.choice(
            normal_indices,
            size=max(1, int(len(normal_indices) * 0.20)),
            replace=False,
        )
        df.loc[chosen, "network_mbps"] = RNG.uniform(
            15.0, 27.0, size=len(chosen)
        )
        df.loc[drop_indices, "network_mbps"] = RNG.uniform(
            15.0, 27.0, size=len(drop_indices)
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return df, test_raw_indices, train_end, val_end


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    probability_threshold = float(
        config["model_probability_threshold"]
    )
    network_threshold = float(config["network_threshold_mbps"])
    expected_features = config["features"]

    original = pd.read_csv(DATA_PATH)
    original["timestamp"] = pd.to_datetime(original["timestamp"])
    original = original.sort_values("timestamp").reset_index(drop=True)

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    scenarios = [
        "baseline",
        "low_normal_traffic",
        "mild_network_drops",
        "overlapping_traffic",
    ]

    all_metrics = []
    all_predictions = []

    for scenario in scenarios:
        # Reset the random generator so each scenario uses the same
        # perturbations as the original stress-test script.
        # The original script uses one shared generator, so we preserve
        # that sequence across scenarios by not resetting RNG here.
        scenario_df, test_raw_indices, train_end, val_end = (
            make_scenario(original, scenario)
        )

        features = build_features(scenario_df)
        valid = features.notna().all(axis=1)

        X = features.loc[valid].reset_index(drop=True)
        y = (
            scenario_df.loc[valid, "is_anomaly"]
            .astype(int)
            .reset_index(drop=True)
        )
        timestamps = (
            scenario_df.loc[valid, "timestamp"]
            .reset_index(drop=True)
        )
        network = (
            scenario_df.loc[valid, "network_mbps"]
            .reset_index(drop=True)
        )
        anomaly_types = (
            scenario_df.loc[valid, "anomaly_type"]
            .reset_index(drop=True)
        )

        X_test = X.iloc[val_end:].copy()
        y_test = y.iloc[val_end:].to_numpy()
        time_test = timestamps.iloc[val_end:].to_numpy()
        network_test = network.iloc[val_end:].to_numpy()
        anomaly_type_test = anomaly_types.iloc[val_end:].to_numpy()

        # Ensure feature order exactly matches the saved configuration.
        X_test = X_test[expected_features]

        probabilities = model.predict_proba(X_test)[:, 1]

        xgb_pred = (
            probabilities >= probability_threshold
        ).astype(int)

        network_pred = (
            network_test < network_threshold
        ).astype(int)

        # Apply the frozen gate to the independent network rule.
        gated_network_pred = (
            (network_pred == 1)
            & (probabilities >= GATE_THRESHOLD)
        ).astype(int)

        # Final prediction: XGBoost OR gated network rule.
        hybrid_pred = (
            (xgb_pred == 1) | (gated_network_pred == 1)
        ).astype(int)

        for model_name, prediction in [
            ("XGBoost only", xgb_pred),
            ("Gated Hybrid", hybrid_pred),
        ]:
            result = metrics_for(
                scenario, y_test, prediction
            )
            result["model"] = model_name
            all_metrics.append(result)

            if model_name == "Gated Hybrid":
                all_predictions.append(pd.DataFrame({
                    "scenario": scenario,
                    "timestamp": time_test,
                    "actual_anomaly": y_test,
                    "anomaly_type": anomaly_type_test,
                    "network_mbps": network_test,
                    "xgboost_probability": probabilities,
                    "xgboost_prediction": xgb_pred,
                    "network_rule_prediction": network_pred,
                    "gated_network_prediction": gated_network_pred,
                    "hybrid_prediction": hybrid_pred,
                }))

        print(f"Completed: {scenario}")

    results = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)

    # Separate output files; original results are not overwritten.
    results_path = (
        OUTPUT_DIR / "gated_hybrid_stress_test_results.csv"
    )
    predictions_path = (
        OUTPUT_DIR / "gated_hybrid_stress_test_predictions.csv"
    )

    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nFIXED THRESHOLDS — NO RETUNING")
    print(f"XGBoost probability threshold: {probability_threshold:.4f}")
    print(f"Network threshold: {network_threshold:.4f} Mbps")
    print(f"Gate threshold: {GATE_THRESHOLD:.4f}")

    columns = [
        "scenario", "model", "fp", "fn", "precision",
        "recall", "f1_score", "false_alarm_rate",
    ]

    print("\nGATED HYBRID STRESS-TEST RESULTS")
    print(
        results[columns].to_string(
            index=False, float_format="%.4f"
        )
    )

    print("\nSaved new files:")
    print(results_path)
    print(predictions_path)
    print(
        "\nThese are synthetic stress tests, "
        "not real-world validation."
    )


if __name__ == "__main__":
    main()
