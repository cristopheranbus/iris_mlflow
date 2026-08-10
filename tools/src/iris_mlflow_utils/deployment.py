"""Reusable gates and promotion helpers for the Databricks deployment job."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import DeploymentConfig


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable result of candidate-versus-Champion validation."""

    passed: bool
    reason: str
    candidate_f1: float
    candidate_accuracy: float
    champion_f1: float | None
    min_f1: float
    min_accuracy: float
    max_regression: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.passed else "failed",
            "reason": self.reason,
            "candidate_test_f1_weighted": self.candidate_f1,
            "candidate_test_accuracy": self.candidate_accuracy,
            "champion_test_f1_weighted": self.champion_f1,
            "min_test_f1_weighted": self.min_f1,
            "min_test_accuracy": self.min_accuracy,
            "max_metric_regression": self.max_regression,
        }


def evaluate_promotion_gate(
    metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    config: DeploymentConfig,
) -> PromotionDecision:
    """Apply absolute quality and Champion-regression gates."""

    candidate_f1 = float(metrics.get("test_f1_weighted", metrics.get("test_f1", 0.0)))
    candidate_accuracy = float(metrics.get("test_accuracy", 0.0))
    champion_f1 = (
        None
        if champion_metrics is None
        else float(champion_metrics.get("test_f1_weighted", champion_metrics.get("test_f1", 0.0)))
    )
    if candidate_f1 < config.min_test_f1_weighted:
        reason = "candidate_f1_below_threshold"
    elif candidate_accuracy < config.min_test_accuracy:
        reason = "candidate_accuracy_below_threshold"
    elif champion_f1 is not None and candidate_f1 < champion_f1 - config.max_metric_regression:
        reason = "candidate_regresses_against_champion"
    else:
        reason = "all_quality_gates_passed"
    return PromotionDecision(
        passed=reason == "all_quality_gates_passed",
        reason=reason,
        candidate_f1=candidate_f1,
        candidate_accuracy=candidate_accuracy,
        champion_f1=champion_f1,
        min_f1=config.min_test_f1_weighted,
        min_accuracy=config.min_test_accuracy,
        max_regression=config.max_metric_regression,
    )


def update_serving_endpoint(
    workspace_client: Any,
    *,
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workload_size: str = "Small",
    scale_to_zero_enabled: bool = True,
) -> Any:
    """Point an endpoint at an exact Unity Catalog model version."""

    return workspace_client.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[
            {
                "entity_name": model_name,
                "entity_version": str(model_version),
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero_enabled,
            }
        ],
    )


def wait_for_endpoint_ready(
    workspace_client: Any,
    endpoint_name: str,
    *,
    timeout_seconds: int = 900,
    poll_seconds: int = 15,
) -> Any:
    """Wait for endpoint readiness and fail on timeout or a terminal failure."""

    deadline = time.monotonic() + timeout_seconds
    last_status: Any = None
    while time.monotonic() < deadline:
        last_status = workspace_client.serving_endpoints.get(endpoint_name)
        state = getattr(last_status, "state", None)
        ready = getattr(state, "ready", state)
        config_update = getattr(state, "config_update", None)
        if str(ready).upper().endswith("READY"):
            return last_status
        if str(ready).upper().endswith(("NOT_READY", "FAILED")):
            raise RuntimeError(f"Endpoint {endpoint_name} no está listo: {last_status!r}")
        if config_update and str(config_update).upper().endswith("FAILED"):
            raise RuntimeError(f"Actualización del endpoint falló: {last_status!r}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Endpoint {endpoint_name} no quedó READY: {last_status!r}")


def promote_champion(
    registry_client: Any,
    *,
    model_name: str,
    model_version: str,
    champion_alias: str = "Champion",
) -> None:
    """Promote an exact version after serving smoke tests have passed."""

    registry_client.set_registered_model_alias(
        name=model_name, alias=champion_alias, version=str(model_version)
    )
