"""Reusable gates and promotion helpers for the Databricks deployment job."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from .config import DeploymentConfig, PromotionPolicy, PromotionRule


@dataclass(frozen=True)
class RuleResult:
    """Auditable outcome for one declarative promotion rule."""

    name: str
    metric: str
    status: str
    candidate_value: float | None
    expected_value: float | None
    baseline_value: float | None = None
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "status": self.status,
            "candidate_value": self.candidate_value,
            "expected_value": self.expected_value,
            "baseline_value": self.baseline_value,
            "required": self.required,
        }


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
    policy_name: str = "legacy"
    policy_version: str = "1"
    rule_results: tuple[RuleResult, ...] = ()

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
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "rules": [result.as_dict() for result in self.rule_results],
        }


@dataclass(frozen=True)
class ServingEndpointSnapshot:
    """Configuration required to restore an endpoint after a failed rollout."""

    endpoint_name: str
    served_entities: list[dict[str, Any]]
    traffic_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class ServingEndpointChange:
    """State required to roll back an endpoint upsert."""

    endpoint_name: str
    created: bool
    snapshot: ServingEndpointSnapshot | None


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
) -> ServingEndpointSnapshot | None:
    """Capture the active endpoint configuration before changing it."""

    try:
        endpoint = workspace_client.serving_endpoints.get(endpoint_name)
    except Exception as error:
        marker = f"{getattr(error, 'error_code', '')} {error}".lower()
        if any(
            value in marker for value in ("resource_does_not_exist", "not found", "does not exist")
        ):
            return None
        raise
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
    """Apply a declarative policy, falling back to the legacy fixed gate."""

    if config.promotion_policy is not None:
        return _evaluate_policy(metrics, champion_metrics, config.promotion_policy)

    candidate_f1 = float(metrics.get("test_f1_weighted", metrics.get("test_f1", 0.0)))
    candidate_accuracy = float(metrics.get("test_accuracy", 0.0))
    champion_f1 = (
        None
        if champion_metrics is None
        else float(champion_metrics.get("test_f1_weighted", champion_metrics.get("test_f1", 0.0)))
    )
    observed_metrics = [candidate_f1, candidate_accuracy]
    if champion_f1 is not None:
        observed_metrics.append(champion_f1)
    if not all(math.isfinite(value) for value in observed_metrics):
        raise ValueError("Las métricas de promoción deben ser valores finitos.")
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


def _compare(left: float, operator: str, right: float) -> bool:
    comparisons = {
        ">=": left >= right,
        "<=": left <= right,
        ">": left > right,
        "<": left < right,
    }
    try:
        return comparisons[operator]
    except KeyError as error:
        raise ValueError(f"Operador de promoción no soportado: {operator}.") from error


def _evaluate_rule(
    rule: PromotionRule,
    metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
) -> RuleResult:
    candidate_raw = metrics.get(rule.metric)
    if candidate_raw is None:
        return RuleResult(
            rule.name,
            rule.metric,
            "failed" if rule.required else "skipped",
            None,
            rule.value,
            required=rule.required,
        )
    candidate = float(candidate_raw)
    if not math.isfinite(candidate):
        raise ValueError(f"La métrica {rule.metric} debe ser finita.")
    baseline_value: float | None = None
    expected = rule.value
    if rule.baseline == "champion":
        if champion_metrics is None:
            return RuleResult(
                rule.name, rule.metric, "skipped", candidate, None, required=rule.required
            )
        champion_raw = champion_metrics.get(rule.metric)
        if champion_raw is None:
            return RuleResult(
                rule.name, rule.metric, "failed", candidate, None, required=rule.required
            )
        baseline_value = float(champion_raw)
        if not math.isfinite(baseline_value):
            raise ValueError(f"La métrica Champion {rule.metric} debe ser finita.")
        expected = (
            baseline_value - rule.allowed_regression
            if rule.operator in {">=", ">"}
            else baseline_value + rule.allowed_regression
        )
    assert expected is not None
    status = "passed" if _compare(candidate, rule.operator, expected) else "failed"
    return RuleResult(
        rule.name,
        rule.metric,
        status,
        candidate,
        expected,
        baseline_value,
        rule.required,
    )


def _evaluate_policy(
    metrics: dict[str, float],
    champion_metrics: dict[str, float] | None,
    policy: PromotionPolicy,
) -> PromotionDecision:
    results = tuple(_evaluate_rule(rule, metrics, champion_metrics) for rule in policy.rules)
    failed = [result for result in results if result.required and result.status == "failed"]
    candidate_f1 = float(metrics.get("test_f1_weighted", metrics.get("test_f1", 0.0)))
    candidate_accuracy = float(metrics.get("test_accuracy", 0.0))
    champion_f1 = None
    if champion_metrics is not None:
        champion_f1 = float(
            champion_metrics.get("test_f1_weighted", champion_metrics.get("test_f1", 0.0))
        )
    absolute_f1 = next(
        (rule.value for rule in policy.rules if rule.metric == "test_f1_weighted" and rule.value),
        0.0,
    )
    absolute_accuracy = next(
        (rule.value for rule in policy.rules if rule.metric == "test_accuracy" and rule.value),
        0.0,
    )
    relative_f1 = next(
        (
            rule.allowed_regression
            for rule in policy.rules
            if rule.metric == "test_f1_weighted" and rule.baseline == "champion"
        ),
        0.0,
    )
    return PromotionDecision(
        passed=not failed,
        reason="all_quality_gates_passed" if not failed else f"rule_failed:{failed[0].name}",
        candidate_f1=candidate_f1,
        candidate_accuracy=candidate_accuracy,
        champion_f1=champion_f1,
        min_f1=float(absolute_f1),
        min_accuracy=float(absolute_accuracy),
        max_regression=float(relative_f1),
        policy_name=policy.name,
        policy_version=policy.version,
        rule_results=results,
    )


def upsert_serving_endpoint(
    workspace_client: Any,
    *,
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workload_size: str = "Small",
    scale_to_zero_enabled: bool = True,
) -> ServingEndpointChange:
    """Create a missing endpoint or update an existing endpoint transactionally."""

    snapshot = capture_serving_endpoint(workspace_client, endpoint_name)
    served_entities = [
        {
            "entity_name": model_name,
            "entity_version": str(model_version),
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
        }
    ]
    if snapshot is None:
        workspace_client.serving_endpoints.create(
            name=endpoint_name,
            config={"served_entities": served_entities},
        )
        return ServingEndpointChange(endpoint_name, True, None)
    workspace_client.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=served_entities,
    )
    return ServingEndpointChange(endpoint_name, False, snapshot)


def rollback_serving_endpoint(workspace_client: Any, change: ServingEndpointChange) -> Any:
    """Restore an updated endpoint or delete one created by a failed rollout."""

    if change.created:
        return workspace_client.serving_endpoints.delete(change.endpoint_name)
    if change.snapshot is None:
        raise RuntimeError("Un endpoint existente requiere snapshot para rollback.")
    return restore_serving_endpoint(workspace_client, change.snapshot)


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
        if str(ready).upper().endswith(("NOT_READY", "FAILED")):
            raise RuntimeError(f"Endpoint {endpoint_name} no está listo: {last_status!r}")
        if config_update and str(config_update).upper().endswith("FAILED"):
            raise RuntimeError(f"Actualización del endpoint falló: {last_status!r}")
        if str(ready).upper().endswith("READY"):
            return last_status
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
