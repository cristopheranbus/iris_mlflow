"""Configurable production-health evaluation without automatic rollback."""

from __future__ import annotations

import json
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MonitoringConfig:
    """Thresholds applied to one aggregated inference-table window."""

    window_hours: int = 24
    minimum_observations: int = 100
    max_error_rate: float = 0.02
    max_p95_latency_ms: float = 750.0
    max_prediction_drift: float = 0.15

    def __post_init__(self) -> None:
        if self.window_hours < 1 or self.minimum_observations < 1:
            raise ValueError("La ventana y minimum_observations deben ser positivos.")
        if not 0 <= self.max_error_rate <= 1 or not 0 <= self.max_prediction_drift <= 1:
            raise ValueError("Los umbrales proporcionales deben estar entre 0 y 1.")
        if self.max_p95_latency_ms <= 0:
            raise ValueError("max_p95_latency_ms debe ser positivo.")


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Aggregated endpoint observations for one monitoring window."""

    endpoint_name: str
    observations: int
    error_rate: float
    p95_latency_ms: float
    logging_errors: int = 0
    prediction_drift: float | None = None


@dataclass(frozen=True)
class MonitoringAlert:
    """One condition requiring human attention."""

    code: str
    observed: float
    threshold: float


@dataclass(frozen=True)
class MonitoringDecision:
    """Auditable alert-only monitoring decision."""

    status: str
    snapshot: MonitoringSnapshot
    alerts: tuple[MonitoringAlert, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": "alert" if self.alerts else "none",
            "snapshot": {
                "endpoint_name": self.snapshot.endpoint_name,
                "observations": self.snapshot.observations,
                "error_rate": self.snapshot.error_rate,
                "p95_latency_ms": self.snapshot.p95_latency_ms,
                "logging_errors": self.snapshot.logging_errors,
                "prediction_drift": self.snapshot.prediction_drift,
            },
            "alerts": [
                {
                    "code": alert.code,
                    "observed": alert.observed,
                    "threshold": alert.threshold,
                }
                for alert in self.alerts
            ],
        }

    def issue_title(self) -> str:
        return f"[ML monitoring] {self.snapshot.endpoint_name} degraded"

    def issue_body(self) -> str:
        return (
            "## Production monitoring alert\n\n```json\n"
            + json.dumps(self.as_dict(), indent=2, sort_keys=True)
            + "\n```\n"
        )


def load_monitoring_config(path: Path | None = None) -> MonitoringConfig:
    """Load versioned monitoring thresholds from TOML."""

    configured = os.getenv("IRIS_MONITORING_CONFIG_PATH", "").strip()
    candidate = Path(configured) if configured else path
    if candidate is None:
        current = Path.cwd().resolve()
        candidates = [current / "config" / "monitoring.toml"] + [
            parent / "config" / "monitoring.toml" for parent in current.parents
        ]
        candidate = next((item for item in candidates if item.is_file()), candidates[0])
    resolved = candidate.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"No se encontró config/monitoring.toml: {resolved}")
    with resolved.open("rb") as config_file:
        payload = tomllib.load(config_file)
    monitoring = dict(payload.get("monitoring", {}))
    thresholds = dict(payload.get("thresholds", {}))
    return MonitoringConfig(
        window_hours=int(monitoring.get("window_hours", 24)),
        minimum_observations=int(monitoring.get("minimum_observations", 100)),
        max_error_rate=float(thresholds.get("max_error_rate", 0.02)),
        max_p95_latency_ms=float(thresholds.get("max_p95_latency_ms", 750)),
        max_prediction_drift=float(thresholds.get("max_prediction_drift", 0.15)),
    )


def evaluate_monitoring(
    snapshot: MonitoringSnapshot, config: MonitoringConfig
) -> MonitoringDecision:
    """Evaluate health signals and return alerts; never mutates Champion."""

    values = (snapshot.error_rate, snapshot.p95_latency_ms)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Las métricas de monitoreo deben ser finitas.")
    if snapshot.observations < config.minimum_observations:
        return MonitoringDecision("insufficient_data", snapshot, ())
    alerts: list[MonitoringAlert] = []
    if snapshot.error_rate > config.max_error_rate:
        alerts.append(MonitoringAlert("error_rate", snapshot.error_rate, config.max_error_rate))
    if snapshot.p95_latency_ms > config.max_p95_latency_ms:
        alerts.append(
            MonitoringAlert("p95_latency_ms", snapshot.p95_latency_ms, config.max_p95_latency_ms)
        )
    if snapshot.logging_errors:
        alerts.append(MonitoringAlert("logging_errors", float(snapshot.logging_errors), 0.0))
    if (
        snapshot.prediction_drift is not None
        and snapshot.prediction_drift > config.max_prediction_drift
    ):
        alerts.append(
            MonitoringAlert(
                "prediction_drift", snapshot.prediction_drift, config.max_prediction_drift
            )
        )
    return MonitoringDecision("degraded" if alerts else "healthy", snapshot, tuple(alerts))
