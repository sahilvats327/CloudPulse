
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score
from anomaly_explainer import explain_anomaly
from incident_analyzer import summarize_incidents

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cloud_metrics.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES = ["cpu_percent", "memory_percent", "network_mbps"]
ANOMALY_TYPES = [
    "cpu_spike",
    "memory_drift",
    "network_drop",
    "combined",
]

SEVERITY_ORDER = ["High", "Medium", "Low"]

st.set_page_config(
    page_title="CloudPulse | Cloud Anomaly Detection",
    page_icon="☁️",
    layout="wide",
)


# --------------------------------------------------
# Load saved files
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(
        MODEL_DIR / f"{model_name}_model.keras"
    )


@st.cache_resource
def load_scaler(model_name):
    scaler_file = (
        "scaler.pkl" if model_name == "lstm"
        else "gru_scaler.pkl"
    )
    return joblib.load(MODEL_DIR / scaler_file)


@st.cache_data
def load_config(model_name):
    with open(
        MODEL_DIR / f"{model_name}_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@st.cache_data
def load_metrics(model_name):
    with open(
        OUTPUT_DIR / f"{model_name}_metrics.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("CloudPulse")
st.sidebar.caption("Cloud Resource Anomaly Detection")

model_choice = st.sidebar.selectbox(
    "Select detection model",
    ["GRU", "LSTM"],
)
model_key = model_choice.lower()

st.sidebar.info(
    "This dashboard uses simulated cloud metrics. "
    "It does not connect to a live AWS account."
)

# --------------------------------------------------
# Load data and model
# --------------------------------------------------
try:
    df = load_data().copy()
    model = load_model(model_key)
    scaler = load_scaler(model_key)
    config = load_config(model_key)
    metrics = load_metrics(model_key)
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.info(
        "Check that the dataset, models, scalers, and configuration "
        "files exist in your CloudPulse project."
    )
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("☁️ CloudPulse")
st.subheader("Cloud Resource Monitoring & Anomaly Detection")
st.write(
    "Monitor CPU, memory, and network metrics. Detect unusual "
    "patterns and inspect their estimated severity and explanations."
)

# --------------------------------------------------
# Predict anomalies
# --------------------------------------------------
window_size = int(config["window_size"])
threshold = float(config["threshold"])

scaled_values = scaler.transform(
    df[FEATURES].to_numpy(dtype=np.float32)
).astype(np.float32)

predictions = np.full(len(df), np.nan)
errors = np.full(len(df), np.nan)

if len(df) > window_size:
    X = np.asarray([
        scaled_values[i - window_size:i]
        for i in range(window_size, len(df))
    ], dtype=np.float32)

    predicted_values = model.predict(X, verbose=0)
    actual_values = scaled_values[window_size:]

    sequence_errors = np.mean(
        (predicted_values - actual_values) ** 2,
        axis=1,
    )

    errors[window_size:] = sequence_errors
    predictions[window_size:] = (
        sequence_errors > threshold
    ).astype(int)

df["anomaly_score"] = errors
df["predicted_anomaly"] = predictions
df["actual_anomaly"] = df["is_anomaly"].astype(int)

# --------------------------------------------------
# Explain detected anomalies
# --------------------------------------------------
df["severity"] = "Normal"
df["explanation"] = "No anomaly detected"
df["score_threshold_ratio"] = np.nan

for idx in df.index[df["predicted_anomaly"] == 1]:
    result = explain_anomaly(
        cpu=df.at[idx, "cpu_percent"],
        memory=df.at[idx, "memory_percent"],
        network=df.at[idx, "network_mbps"],
        anomaly_score=df.at[idx, "anomaly_score"],
        threshold=threshold,
    )

    df.at[idx, "severity"] = result["severity"]
    df.at[idx, "explanation"] = result["explanation"]
    df.at[idx, "score_threshold_ratio"] = (
        result["score_to_threshold_ratio"]
    )

# --------------------------------------------------
# Date filters
# --------------------------------------------------
st.sidebar.subheader("Data filters")

start_date = st.sidebar.date_input(
    "Start date",
    value=df["timestamp"].min().date(),
    min_value=df["timestamp"].min().date(),
    max_value=df["timestamp"].max().date(),
)

end_date = st.sidebar.date_input(
    "End date",
    value=df["timestamp"].max().date(),
    min_value=df["timestamp"].min().date(),
    max_value=df["timestamp"].max().date(),
)

if start_date > end_date:
    st.warning("Start date must not be after end date.")
    st.stop()

filtered = df[
    (df["timestamp"].dt.date >= start_date)
    & (df["timestamp"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("No observations exist for this date range.")
    st.stop()

evaluated = filtered[
    filtered["predicted_anomaly"].notna()
].copy()
evaluated["predicted_anomaly"] = (
    evaluated["predicted_anomaly"].astype(int)
)

# --------------------------------------------------
# KPI cards
# --------------------------------------------------
average_cpu = filtered["cpu_percent"].mean()
average_memory = filtered["memory_percent"].mean()
average_network = filtered["network_mbps"].mean()

detected = evaluated[evaluated["predicted_anomaly"] == 1]
detected_count = len(detected)

high_count = int((detected["severity"] == "High").sum())
medium_count = int((detected["severity"] == "Medium").sum())
low_count = int((detected["severity"] == "Low").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average CPU", f"{average_cpu:.1f}%")
c2.metric("Average Memory", f"{average_memory:.1f}%")
c3.metric("Average Network", f"{average_network:.1f} Mbps")
c4.metric("Predicted Anomalies", detected_count)

st.divider()
st.subheader("Anomaly Severity Overview")

s1, s2, s3 = st.columns(3)
s1.metric("High Severity", high_count)
s2.metric("Medium Severity", medium_count)
s3.metric("Low Severity", low_count)

st.caption(
    "Severity is estimated by initial rule-based heuristics and "
    "prediction-error ratios. It is not an AWS incident severity."
)

# --------------------------------------------------
# Resource charts
# --------------------------------------------------
st.subheader("Cloud Resource Metrics")

tab_cpu, tab_memory, tab_network = st.tabs(
    ["CPU", "Memory", "Network"]
)

with tab_cpu:
    st.line_chart(
        filtered.set_index("timestamp")[["cpu_percent"]]
    )

with tab_memory:
    st.line_chart(
        filtered.set_index("timestamp")[["memory_percent"]]
    )

with tab_network:
    st.line_chart(
        filtered.set_index("timestamp")[["network_mbps"]]
    )

# --------------------------------------------------
# Anomaly visualization
# --------------------------------------------------
st.subheader(f"{model_choice} Anomaly Detection")
st.caption(
    f"Observations with prediction error above {threshold:.6f} "
    "are flagged as anomalies."
)

for feature in FEATURES:
    st.markdown(f"**{feature.replace('_', ' ').title()}**")
    st.line_chart(
        filtered.set_index("timestamp")[[feature]]
    )

# --------------------------------------------------
# Detailed anomaly report
# --------------------------------------------------
st.subheader("Detected Anomalies")

severity_filter = st.sidebar.multiselect(
    "Filter anomaly severity",
    options=SEVERITY_ORDER,
    default=SEVERITY_ORDER,
)

anomalies = evaluated[
    evaluated["predicted_anomaly"] == 1
].copy()

visible_anomalies = anomalies[
    anomalies["severity"].isin(severity_filter)
].copy()

if anomalies.empty:
    st.success("No anomalies were flagged in this date range.")
elif visible_anomalies.empty:
    st.info("No anomalies match the selected severity filter.")
else:
    st.warning(
        f"{len(visible_anomalies)} anomalies displayed "
        f"out of {len(anomalies)} detected."
    )

    report_columns = [
        "timestamp",
        "severity",
        "explanation",
        "cpu_percent",
        "memory_percent",
        "network_mbps",
        "anomaly_score",
        "score_threshold_ratio",
        "actual_anomaly",
        "anomaly_type",
    ]

    report = visible_anomalies[report_columns].sort_values(
        "anomaly_score",
        ascending=False,
    )

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download explained anomaly report (CSV)",
        data=csv_data,
        file_name=f"cloudpulse_{model_key}_explained_anomalies.csv",
        mime="text/csv",
    )
# --------------------------------------------------
# Incident Analysis
# --------------------------------------------------
st.divider()
st.subheader("Incident Timeline & Summary")

st.caption(
    "Nearby detected anomalies are grouped into incidents. "
    "Incident severity is based on the highest estimated severity "
    "among its detected anomalies."
)

if evaluated.empty:
    st.info("No evaluable observations are available for incident analysis.")
else:
    try:
        incidents = summarize_incidents(evaluated, threshold=threshold)

        if incidents.empty:
            st.success(
                "No incidents were detected in the selected date range."
            )
        else:
            # Incident overview
            total_incidents = len(incidents)
            high_incidents = int(
                (incidents["severity"] == "High").sum()
            )
            medium_incidents = int(
                (incidents["severity"] == "Medium").sum()
            )
            low_incidents = int(
                (incidents["severity"] == "Low").sum()
            )

            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Total Incidents", total_incidents)
            i2.metric("High", high_incidents)
            i3.metric("Medium", medium_incidents)
            i4.metric("Low", low_incidents)

            # Timeline
            st.markdown("**Incident Timeline**")

            timeline = incidents.copy()
            timeline["start_time"] = pd.to_datetime(
                timeline["start_time"]
            )
            timeline["severity_level"] = timeline[
                "severity"
            ].map({"Low": 1, "Medium": 2, "High": 3})

            st.scatter_chart(
                timeline,
                x="start_time",
                y="severity_level",
                size="anomaly_count",
                x_label="Incident start time",
                y_label="Severity (1 = Low, 3 = High)",
            )

             # Incident details
            st.markdown("**Incident Details**")

            display_columns = [
                "incident_id",
                "start_time",
                "end_time",
                "duration",
                "anomaly_count",
                "severity",
                "likely_indicators",
                "peak_anomaly_score",
                "score_to_threshold_ratio",
                "avg_cpu_percent",
                "peak_memory_percent",
                "min_network_mbps",
            ]

            incident_report = incidents[display_columns].copy()

            incident_report = incident_report.rename(columns={
                "incident_id": "Incident ID",
                "start_time": "Start Time",
                "end_time": "End Time",
                "duration": "Observed Duration",
                "anomaly_count": "Anomaly Count",
                "severity": "Severity",
                "likely_indicators": "Likely Indicators",
                "peak_anomaly_score": "Peak Anomaly Score",
                "score_to_threshold_ratio": "Peak Score / Threshold",
                "avg_cpu_percent": "Average CPU (%)",
                "peak_memory_percent": "Peak Memory (%)",
                "min_network_mbps": "Minimum Network (Mbps)",
            })

            st.dataframe(
                incident_report.style.format({
                    "Peak Anomaly Score": "{:.4f}",
                    "Peak Score / Threshold": "{:.2f}x",
                    "Average CPU (%)": "{:.2f}",
                    "Peak Memory (%)": "{:.2f}",
                    "Minimum Network (Mbps)": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            incident_csv = incident_report.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Incident Report (CSV)",
                data=incident_csv,
                file_name=(
                    f"cloudpulse_{model_key}_incident_report.csv"
                ),
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Incident analysis failed: {exc}")
        st.caption(
            "Check that incident_analyzer.py exists and contains "
            "the summarize_incidents function."
        )


# --------------------------------------------------
# Evaluation for selected date range
# --------------------------------------------------
st.divider()
st.subheader("Evaluation for Selected Date Range")

if evaluated.empty or evaluated["actual_anomaly"].nunique() < 2:
    st.info(
        "Select a date range containing both normal observations "
        "and anomalies to calculate meaningful evaluation metrics."
    )
else:
    y_true = evaluated["actual_anomaly"].astype(int).to_numpy()
    y_pred = evaluated["predicted_anomaly"].astype(int).to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision:.2%}")
    m2.metric("Recall", f"{recall:.2%}")
    m3.metric("F1-score", f"{f1:.2%}")
    m4.metric("Normal False Alarm Rate", f"{false_alarm_rate:.2%}")

    st.markdown("**Classification Results**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positives", tp)
    c2.metric("False Positives", fp)
    c3.metric("True Negatives", tn)
    c4.metric("False Negatives", fn)

    st.markdown("**Confusion Matrix**")
    st.caption(
        "Rows are actual labels; columns are model predictions."
    )

    confusion_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    )

    st.dataframe(confusion_df, use_container_width=True)

    st.caption(
        "These metrics describe the selected date range. "
        "They are not the saved held-out test results unless "
        "the selected observations match that test set exactly."
    )
# --------------------------------------------------
# False Positive Analysis
# --------------------------------------------------
st.divider()
st.subheader("False Positive Analysis")

false_positives = evaluated[
    (evaluated["actual_anomaly"] == 0)
    & (evaluated["predicted_anomaly"] == 1)
].copy()

normal_observations = evaluated[
    evaluated["actual_anomaly"] == 0
]

if normal_observations.empty:
    st.info("No normal observations are available for analysis.")
else:
    fp_count = len(false_positives)
    normal_count = len(normal_observations)
    fp_rate = fp_count / normal_count

    fp1, fp2, fp3 = st.columns(3)

    fp1.metric("False Positives", fp_count)
    fp2.metric("Normal Observations", normal_count)
    fp3.metric("False Positive Rate", f"{fp_rate:.2%}")

    if false_positives.empty:
        st.success("No false positives in the selected date range.")
    else:
        st.markdown("**Highest-scoring false positives**")

        fp_columns = [
            "timestamp",
            "severity",
            "cpu_percent",
            "memory_percent",
            "network_mbps",
            "anomaly_score",
            "score_threshold_ratio",
        ]

        fp_report = false_positives[fp_columns].sort_values(
            "anomaly_score",
            ascending=False,
        )

        st.dataframe(
            fp_report.head(20).style.format({
                "cpu_percent": "{:.2f}",
                "memory_percent": "{:.2f}",
                "network_mbps": "{:.2f}",
                "anomaly_score": "{:.4f}",
                "score_threshold_ratio": "{:.2f}x",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Metric summary for false positives**")

        metric_summary = false_positives[
            ["cpu_percent", "memory_percent", "network_mbps",
             "anomaly_score"]
        ].agg(["mean", "min", "max"]).T

        metric_summary.columns = ["Mean", "Minimum", "Maximum"]

        st.dataframe(
            metric_summary.style.format("{:.2f}"),
            use_container_width=True,
        )


# --------------------------------------------------
# Anomaly-type recall
# --------------------------------------------------
st.divider()
st.subheader("Detection by Anomaly Type")

if evaluated.empty:
    st.info("No evaluable observations in this date range.")
else:
    type_rows = []

    for anomaly_type in ANOMALY_TYPES:
        actual_type = evaluated["anomaly_type"] == anomaly_type
        actual_count = int(actual_type.sum())
        detected_type = int(
            (
                actual_type
                & (evaluated["predicted_anomaly"] == 1)
            ).sum()
        )
        missed_type = actual_count - detected_type

        type_rows.append({
            "Anomaly Type": anomaly_type,
            "Actual": actual_count,
            "Detected": detected_type,
            "Missed": missed_type,
            "Recall": (
                detected_type / actual_count
                if actual_count else np.nan
            ),
        })

    type_df = pd.DataFrame(type_rows)
    type_df["Recall (%)"] = type_df["Recall"] * 100

    st.dataframe(
        type_df[
            ["Anomaly Type", "Actual", "Detected", "Missed", "Recall (%)"]
        ].style.format({"Recall (%)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    chart = type_df.set_index("Anomaly Type")[["Recall (%)"]].dropna()
    if not chart.empty:
        st.bar_chart(chart)

# --------------------------------------------------
# Saved LSTM vs GRU comparison
# --------------------------------------------------
st.divider()
st.subheader("LSTM vs GRU — Held-Out Test Comparison")

try:
    comparison = pd.DataFrame([
        load_metrics("lstm"),
        load_metrics("gru"),
    ])

    comparison = comparison.rename(columns={
        "model": "Model",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "false_positives": "False Alarms",
        "false_negatives": "Missed Anomalies",
        "test_observations": "Test Observations",
    })

    for column in ["Precision", "Recall", "F1 Score"]:
        comparison[column] *= 100

    columns = [
        "Model",
        "Test Observations",
        "Precision",
        "Recall",
        "F1 Score",
        "False Alarms",
        "Missed Anomalies",
    ]

    st.dataframe(
        comparison[columns].style.format({
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "Peak Score / Threshold": "{:.2f}x",

        }),
        use_container_width=True,
        hide_index=True,
    )

except Exception as exc:
    st.warning(f"Could not display model comparison: {exc}")

with st.expander(f"View saved {model_choice} test metrics"):
    st.json(metrics)

st.caption(
    "CloudPulse prototype | Synthetic telemetry | LSTM and GRU. "
    "Severity uses heuristic rules; results do not establish "
    "real-world AWS accuracy."
)

# --------------------------------------------------
# Five-model held-out test comparison
# --------------------------------------------------
st.divider()
st.subheader("Five-Model Held-Out Test Comparison")
st.caption(
    "All five models are evaluated on the same 1,499 held-out "
    "timestamps with 120 actual anomalies. These are saved offline "
    "evaluation results, not live AWS performance."
)

comparison_path = OUTPUT_DIR / "fair_model_comparison.csv"

try:
    five_model_comparison = pd.read_csv(comparison_path)

    if five_model_comparison.empty:
        st.info("The saved five-model comparison file is empty.")
    else:
        display_comparison = five_model_comparison.rename(columns={
            "model": "Model",
            "test_rows": "Test Rows",
            "actual_anomalies": "Actual Anomalies",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score",
            "false_alarm_rate": "False Alarm Rate",
            "tn": "True Negatives",
            "fp": "False Positives",
            "fn": "False Negatives",
            "tp": "True Positives",
        })

        percentage_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "False Alarm Rate",
        ]

        for column in percentage_columns:
            display_comparison[column] *= 100

        display_columns = [
            "Model",
            "Test Rows",
            "Actual Anomalies",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "False Alarm Rate",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
        ]

        st.dataframe(
            display_comparison[display_columns].style.format({
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1 Score": "{:.2f}%",
                "False Alarm Rate": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**F1 Score Comparison**")
        f1_chart = display_comparison.set_index("Model")[["F1 Score"]]
        st.bar_chart(f1_chart)

        best_model = display_comparison.loc[
            display_comparison["F1 Score"].idxmax()
        ]

        st.success(
            f"Best F1 Score: {best_model['Model']} "
            f"({best_model['F1 Score']:.2f}%)"
        )

        st.download_button(
            "Download Five-Model Comparison CSV",
            data=five_model_comparison.to_csv(index=False).encode("utf-8"),
            file_name="cloudpulse_five_model_comparison.csv",
            mime="text/csv",
        )

except FileNotFoundError:
    st.warning(
        "Five-model comparison results were not found. "
        "Expected file: outputs/fair_model_comparison.csv"
    )
except Exception as exc:
    st.warning(f"Could not display five-model comparison: {exc}")

# --------------------------------------------------
# Five-model confusion matrix analysis
# --------------------------------------------------
st.divider()
st.subheader("Five-Model Confusion Matrix Analysis")
st.caption(
    "Select a model to inspect its predictions on the same held-out test set. "
    "These results use saved offline predictions."
)

predictions_path = OUTPUT_DIR / "fair_model_predictions.csv"

try:
    prediction_data = pd.read_csv(predictions_path)

    required_columns = {"model", "actual_anomaly", "prediction"}
    if not required_columns.issubset(prediction_data.columns):
        st.warning("The saved predictions file is missing required columns.")
    else:
        model_names = sorted(prediction_data["model"].dropna().unique())
        selected_comparison_model = st.selectbox(
            "Select model for confusion matrix",
            model_names,
            key="five_model_confusion_matrix",
        )

        selected_predictions = prediction_data[
            prediction_data["model"] == selected_comparison_model
        ].copy()

        actual = selected_predictions["actual_anomaly"].astype(int)
        predicted = selected_predictions["prediction"].astype(int)

        tn = int(((actual == 0) & (predicted == 0)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())
        tp = int(((actual == 1) & (predicted == 1)).sum())

        st.markdown("**Confusion Matrix**")
        st.caption("Rows are actual labels; columns are predicted labels.")

        confusion_matrix_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual Normal", "Actual Anomaly"],
            columns=["Predicted Normal", "Predicted Anomaly"],
        )

        st.dataframe(
            confusion_matrix_df,
            use_container_width=True,
        )

        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("True Positives", tp)
        m2.metric("False Positives", fp)
        m3.metric("False Negatives", fn)
        m4.metric("True Negatives", tn)

        st.markdown("**Metrics from this confusion matrix**")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{accuracy * 100:.2f}%")
        metric_cols[1].metric("Precision", f"{precision * 100:.2f}%")
        metric_cols[2].metric("Recall", f"{recall * 100:.2f}%")
        metric_cols[3].metric("F1 Score", f"{f1 * 100:.2f}%")

except FileNotFoundError:
    st.warning(
        "Saved predictions were not found at "
        "outputs/fair_model_predictions.csv"
    )
except Exception as exc:
    st.warning(f"Could not display confusion matrix: {exc}")

