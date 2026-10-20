
import pandas as pd


def summarize_incidents(df, max_gap=1, threshold=None):
    """
    Group nearby anomalies into incidents and estimate incident severity.

    Severity is a heuristic based on:
    - Existing row-level severity
    - CPU, memory, and network indicators
    - Peak anomaly score relative to the model threshold
    - Number of anomalous observations in the incident

    This is an initial rule-based approach, not a validated
    production incident severity model.
    """

    required = {
        "timestamp",
        "predicted_anomaly",
        "severity",
        "cpu_percent",
        "memory_percent",
        "network_mbps",
        "anomaly_score",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if threshold is not None and threshold <= 0:
        raise ValueError("threshold must be greater than zero")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    data["predicted_anomaly"] = (
        pd.to_numeric(
            data["predicted_anomaly"], errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    anomaly_rows = data.index[
        data["predicted_anomaly"] == 1
    ].tolist()

    columns = [
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

    if not anomaly_rows:
        return pd.DataFrame(columns=columns)

    groups = []
    current_group = [anomaly_rows[0]]

    for idx in anomaly_rows[1:]:
        if idx - current_group[-1] <= max_gap + 1:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]

    groups.append(current_group)

    severity_rank = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
    }

    incidents = []

    for incident_number, indices in enumerate(groups, start=1):
        part = data.loc[indices]

        severities = part["severity"].astype(str)
        existing_severity = max(
            severities,
            key=lambda value: severity_rank.get(value, 0),
        )

        cpu_values = pd.to_numeric(
            part["cpu_percent"], errors="coerce"
        )
        memory_values = pd.to_numeric(
            part["memory_percent"], errors="coerce"
        )
        network_values = pd.to_numeric(
            part["network_mbps"], errors="coerce"
        )
        score_values = pd.to_numeric(
            part["anomaly_score"], errors="coerce"
        )

        peak_score = score_values.max()
        score_ratio = (
            peak_score / threshold
            if threshold is not None and pd.notna(peak_score)
            else None
        )

        indicators = []

        high_cpu = (cpu_values >= 90).any()
        high_memory = (memory_values >= 90).any()
        elevated_cpu = (cpu_values >= 80).any()
        elevated_memory = (memory_values >= 80).any()
        low_network = (network_values < 5).any()

        combined_pressure = (
            (cpu_values >= 80) & (memory_values >= 80)
        ).any()

        if high_cpu:
            indicators.append("Very high CPU usage")
        elif elevated_cpu:
            indicators.append("High CPU usage")

        if high_memory:
            indicators.append("Very high memory usage")
        elif elevated_memory:
            indicators.append("High memory usage")

        if low_network:
            indicators.append("Low network throughput")

        if combined_pressure:
            indicators.append(
                "Combined CPU and memory pressure"
            )

        if score_ratio is not None and score_ratio >= 2:
            indicators.append(
                "Anomaly score is at least 2x the threshold"
            )
        elif score_ratio is not None and score_ratio >= 1.25:
            indicators.append(
                "Anomaly score is substantially above threshold"
            )

        anomaly_count = len(part)

        # Start with the existing row-level severity.
        severity = (
            existing_severity
            if existing_severity in severity_rank
            else "Low"
        )

        # High severity: strong metric warning, a large score,
        # or a sustained incident with a clearly elevated score.
        if (
            high_cpu
            or high_memory
            or combined_pressure
            or (
                score_ratio is not None
                and score_ratio >= 2
            )
            or (
                anomaly_count >= 10
                and score_ratio is not None
                and score_ratio >= 1.25
            )
        ):
            severity = "High"

        # Medium severity: moderate metric warning, elevated
        # score, or multiple anomalous observations.
        elif (
            elevated_cpu
            or elevated_memory
            or low_network
            or (
                score_ratio is not None
                and score_ratio >= 1.25
            )
            or anomaly_count >= 3
        ):
            if severity_rank.get(severity, 0) < 2:
                severity = "Medium"

        if not indicators:
            indicators.append(
                "Unusual metric pattern; no single indicator identified"
            )

        start_time = part["timestamp"].min()
        end_time = part["timestamp"].max()

        incidents.append({
            "incident_id": f"INC-{incident_number:03d}",
            "start_time": start_time,
            "end_time": end_time,
            "duration": str(end_time - start_time),
            "anomaly_count": anomaly_count,
            "severity": severity,
            "likely_indicators": "; ".join(indicators),
            "peak_anomaly_score": peak_score,
            "score_to_threshold_ratio": score_ratio,
            "avg_cpu_percent": cpu_values.mean(),
            "peak_memory_percent": memory_values.max(),
            "min_network_mbps": network_values.min(),
        })

    return pd.DataFrame(incidents, columns=columns)
