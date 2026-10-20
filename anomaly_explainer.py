
import numpy as np


def explain_anomaly(cpu, memory, network, anomaly_score, threshold):
    """
    Explain a detected anomaly using metric thresholds and prediction error.

    The thresholds below are initial heuristic values for CloudPulse's
    synthetic dataset. Calibrate them before using real cloud telemetry.
    """

    cpu = float(cpu)
    memory = float(memory)
    network = float(network)
    anomaly_score = float(anomaly_score)
    threshold = float(threshold)

    reasons = []

    # Metric-level signals
    if cpu >= 90:
        reasons.append("CPU usage is critically high")
    elif cpu >= 80:
        reasons.append("CPU usage is unusually high")

    if memory >= 90:
        reasons.append("Memory usage is critically high")
    elif memory >= 80:
        reasons.append("Memory usage is unusually high")

    if network < 5:
        reasons.append("Network throughput is unusually low")

    if cpu >= 80 and memory >= 80:
        reasons.append("CPU and memory are elevated together")

    # Prediction-error signal
    ratio = anomaly_score / max(threshold, 1e-9)

    if ratio >= 2.0:
        reasons.append("Observed metrics differ greatly from the model's prediction")
    elif ratio >= 1.25:
        reasons.append("Observed metrics differ substantially from the model's prediction")

    if not reasons:
        reasons.append(
            "The model detected an unusual pattern; no single metric "
            "crossed the initial explanation thresholds"
        )

    # Severity is a rule-based estimate, not an AWS incident severity.
    if (
        cpu >= 90
        or memory >= 90
        or ratio >= 2.0
        or (cpu >= 80 and memory >= 80)
    ):
        severity = "High"
    elif (
        cpu >= 80
        or memory >= 80
        or network < 5
        or ratio >= 1.25
    ):
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "severity": severity,
        "explanation": "; ".join(reasons),
        "score_to_threshold_ratio": round(ratio, 2),
    }
