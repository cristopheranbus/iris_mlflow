"""Unit tests for alert-only production monitoring."""

from pathlib import Path

import pytest

from iris_mlflow_utils.monitoring import (
    MonitoringConfig,
    MonitoringSnapshot,
    evaluate_monitoring,
    load_monitoring_config,
)

pytestmark = pytest.mark.unit


def test_monitoring_config_is_versioned() -> None:
    config = load_monitoring_config()
    assert config.window_hours == 24
    assert config.max_error_rate == 0.02


def test_monitoring_is_alert_only_and_auditable() -> None:
    snapshot = MonitoringSnapshot("iris", 120, 0.03, 900, logging_errors=1)
    decision = evaluate_monitoring(snapshot, MonitoringConfig())

    assert decision.status == "degraded"
    assert {alert.code for alert in decision.alerts} == {
        "error_rate",
        "p95_latency_ms",
        "logging_errors",
    }
    assert decision.as_dict()["action"] == "alert"
    assert decision.issue_title() == "[ML monitoring] iris degraded"
    assert "Champion" not in decision.issue_body()


def test_monitoring_handles_insufficient_data_and_recovery() -> None:
    config = MonitoringConfig(minimum_observations=10)
    insufficient = evaluate_monitoring(MonitoringSnapshot("iris", 9, 1.0, 9999), config)
    healthy = evaluate_monitoring(MonitoringSnapshot("iris", 10, 0.0, 10), config)

    assert insufficient.status == "insufficient_data"
    assert healthy.status == "healthy"


@pytest.mark.parametrize(
    "changes",
    [
        {"window_hours": 0},
        {"minimum_observations": 0},
        {"max_error_rate": 2},
        {"max_p95_latency_ms": 0},
    ],
)
def test_monitoring_rejects_invalid_thresholds(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        MonitoringConfig(**changes)  # type: ignore[arg-type]


def test_monitoring_config_supports_an_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.toml"
    path.write_text(
        "[monitoring]\nwindow_hours=1\nminimum_observations=2\n"
        "[thresholds]\nmax_error_rate=0.1\nmax_p95_latency_ms=50\nmax_prediction_drift=0.2\n",
        encoding="utf-8",
    )
    assert load_monitoring_config(path).minimum_observations == 2


def test_monitoring_config_environment_override_and_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.toml"
    monkeypatch.setenv("IRIS_MONITORING_CONFIG_PATH", str(missing))
    with pytest.raises(FileNotFoundError, match="monitoring.toml"):
        load_monitoring_config()


def test_monitoring_rejects_non_finite_values_and_detects_drift() -> None:
    config = MonitoringConfig(minimum_observations=1)
    with pytest.raises(ValueError, match="finitas"):
        evaluate_monitoring(MonitoringSnapshot("iris", 1, float("nan"), 1), config)
    drift = evaluate_monitoring(MonitoringSnapshot("iris", 1, 0, 1, prediction_drift=0.2), config)
    assert [alert.code for alert in drift.alerts] == ["prediction_drift"]
