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


@dataclass(frozen=True)
class ServingEndpointSnapshot:
    """Configuration required to restore an endpoint after a failed rollout."""

    endpoint_name: str
    served_entities: list[dict[str, Any]]
    traffic_config: dict[str, Any] | None = None


def _as_dictionary(value: Any) -> dict[str, Any]:
    """Convert SDK models or dictionaries into API-compatible dictionaries."""

    if isinstance(value, dict):
        return dict(value)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    raise TypeError(f"No se puede serializar la configuración de serving: {value!r}")


def capture_serving_endpoint(
    workspace_client: Any,
    endpoint_name: str,
) -> ServingEndpointSnapshot:
    """Capture the active endpoint configuration before changing it."""

    endpoint = workspace_client.serving_endpoints.get(endpoint_name)
    endpoint_config = (
        endpoint.get("config") if isinstance(endpoint, dict) else getattr(endpoint, "config", None)
    )
    if endpoint_config is None:
        raise RuntimeError(f"El endpoint {endpoint_name} no expone una configuración restaurable.")
    served_entities = (
        endpoint_config.get("served_entities", [])
        if isinstance(endpoint_config, dict)
        else getattr(endpoint_config, "served_entities", [])
    )
    if not served_entities:
        raise RuntimeError(f"El endpoint {endpoint_name} no contiene entidades servidas.")
    traffic_config = (
        endpoint_config.get("traffic_config")
        if isinstance(endpoint_config, dict)
        else getattr(endpoint_config, "traffic_config", None)
    )
    return ServingEndpointSnapshot(
        endpoint_name=endpoint_name,
        served_entities=[_as_dictionary(entity) for entity in served_entities],
        traffic_config=None if traffic_config is None else _as_dictionary(traffic_config),
    )


def restore_serving_endpoint(
    workspace_client: Any,
    snapshot: ServingEndpointSnapshot,
) -> Any:
    """Restore a previously captured Model Serving configuration."""

    request: dict[str, Any] = {
        "name": snapshot.endpoint_name,
        "served_entities": snapshot.served_entities,
    }
    if snapshot.traffic_config is not None:
        request["traffic_config"] = snapshot.traffic_config
    return workspace_client.serving_endpoints.update_config(**request)


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
    previous_champion_version: str | None = None,
) -> None:
    """Promote an exact version and persist consistent lifecycle evidence."""

    registry_client.set_model_version_tag(
        model_name, str(model_version), "smoke_test_status", "passed"
    )
    registry_client.set_model_version_tag(
        model_name, str(model_version), "deployment_status", "deployed"
    )
    registry_client.set_registered_model_alias(
        name=model_name, alias=champion_alias, version=str(model_version)
    )
    registry_client.set_model_version_tag(model_name, str(model_version), "lifecycle", "champion")
    if previous_champion_version and str(previous_champion_version) != str(model_version):
        registry_client.set_model_version_tag(
            model_name, str(previous_champion_version), "lifecycle", "previous_champion"
        )
        registry_client.set_model_version_tag(
            model_name, str(previous_champion_version), "deployment_status", "superseded"
        )


def ensure_deployment_job(
    api_client: Any,
    *,
    job_name: str,
    settings: dict[str, Any],
) -> tuple[str, str]:
    """Create or reset one exact-name job without producing duplicates."""

    response = api_client.do("GET", "/api/2.1/jobs/list", query={"name": job_name})
    jobs = [
        job for job in response.get("jobs", []) if job.get("settings", {}).get("name") == job_name
    ]
    if len(jobs) > 1:
        raise RuntimeError(
            f"Existen varios jobs llamados {job_name}; resuelve la duplicidad primero."
        )
    if jobs:
        job_id = str(jobs[0]["job_id"])
        api_client.do(
            "POST",
            "/api/2.1/jobs/reset",
            body={"job_id": int(job_id), "new_settings": settings},
        )
        return job_id, "updated"
    created = api_client.do("POST", "/api/2.1/jobs/create", body=settings)
    return str(created["job_id"]), "created"
