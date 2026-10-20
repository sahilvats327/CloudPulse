
import pytest

from anomaly_explainer import explain_anomaly


def test_high_cpu_and_memory():
    result = explain_anomaly(
        cpu=96,
        memory=92,
        network=35,
        anomaly_score=3.2,
        threshold=1.3,
    )

    assert result["severity"] == "High"
    assert "CPU usage is critically high" in result["explanation"]
    assert "Memory usage is critically high" in result["explanation"]
    assert result["score_to_threshold_ratio"] == 2.46


def test_low_network_throughput():
    result = explain_anomaly(
        cpu=45,
        memory=55,
        network=2,
        anomaly_score=1.0,
        threshold=1.0,
    )

    assert result["severity"] == "Medium"
    assert "Network throughput is unusually low" in result["explanation"]


def test_large_prediction_error():
    result = explain_anomaly(
        cpu=50,
        memory=60,
        network=30,
        anomaly_score=2.8,
        threshold=1.3,
    )

    assert result["severity"] == "High"
    assert "differ greatly from the model's prediction" in result["explanation"]
    assert result["score_to_threshold_ratio"] == 2.15


def test_normal_metrics_and_low_error():
    result = explain_anomaly(
        cpu=40,
        memory=50,
        network=30,
        anomaly_score=0.5,
        threshold=1.0,
    )

    assert result["severity"] == "Low"
    assert "no single metric crossed" in result["explanation"]
    assert result["score_to_threshold_ratio"] == 0.5


def test_unusually_high_cpu_is_medium():
    result = explain_anomaly(
        cpu=85,
        memory=60,
        network=30,
        anomaly_score=1.0,
        threshold=1.0,
    )

    assert result["severity"] == "Medium"
    assert "CPU usage is unusually high" in result["explanation"]


def test_prediction_error_at_high_severity_boundary():
    result = explain_anomaly(
        cpu=50,
        memory=60,
        network=30,
        anomaly_score=2.0,
        threshold=1.0,
    )

    assert result["severity"] == "High"
    assert result["score_to_threshold_ratio"] == 2.0


def test_inputs_are_converted_to_numbers():
    result = explain_anomaly(
        cpu="40",
        memory="50",
        network="30",
        anomaly_score="0.5",
        threshold="1.0",
    )

    assert result["severity"] == "Low"
    assert result["score_to_threshold_ratio"] == 0.5
