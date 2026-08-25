"""Unit tests for quality gates and transactional serving deployment."""

from types import SimpleNamespace
from typing import Any

import pytest

import iris_mlflow_utils.deployment as deployment_module
from iris_mlflow_utils.config import DeploymentConfig, PromotionPolicy, PromotionRule
from iris_mlflow_utils.deployment import (
    ServingEndpointChange,
    capture_serving_endpoint,
    evaluate_promotion_gate,
    promote_champion,
    restore_serving_endpoint,
    rollback_serving_endpoint,
    upsert_serving_endpoint,
    wait_for_endpoint_ready,
)

pytestmark = pytest.mark.unit


def config() -> DeploymentConfig:
    return DeploymentConfig(model_name="workspace.default.iris", endpoint_name="iris")


@pytest.mark.parametrize(
    "metrics, champion, passed, reason",
    [
        ({"test_f1_weighted": 0.95, "test_accuracy": 0.95}, None, True, "all_quality_gates_passed"),
        (
            {"test_f1_weighted": 0.89, "test_accuracy": 0.99},
            None,
            False,
            "candidate_f1_below_threshold",
        ),
        (
            {"test_f1_weighted": 0.95, "test_accuracy": 0.89},
            None,
            False,
            "candidate_accuracy_below_threshold",
        ),
        (
            {"test_f1_weighted": 0.90, "test_accuracy": 0.90},
            {"test_f1_weighted": 0.91},
            True,
            "all_quality_gates_passed",
        ),
        (
            {"test_f1": 0.93, "test_accuracy": 0.95},
            {"test_f1": 0.95},
            False,
            "candidate_regresses_against_champion",
        ),
        ({}, None, False, "candidate_f1_below_threshold"),
    ],
)
def test_promotion_gate_covers_business_decisions(
    metrics: dict[str, float],
    champion: dict[str, float] | None,
    passed: bool,
    reason: str,
) -> None:
    decision = evaluate_promotion_gate(metrics, champion, config())
    assert decision.passed is passed
    assert decision.reason == reason
    assert decision.as_dict()["status"] == ("passed" if passed else "failed")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_promotion_gate_rejects_non_finite_metrics(value: float) -> None:
    with pytest.raises(ValueError, match="finitos"):
        evaluate_promotion_gate({"test_f1_weighted": value, "test_accuracy": 0.95}, None, config())


def test_declarative_policy_evaluates_absolute_relative_and_advisory_rules() -> None:
    policy = PromotionPolicy(
        "production",
        "2",
        (
            PromotionRule("minimum", "test_f1_weighted", ">=", value=0.90),
            PromotionRule(
                "relative",
                "test_f1_weighted",
                ">=",
                baseline="champion",
                allowed_regression=0.01,
            ),
            PromotionRule("optional", "test_log_loss", "<=", value=0.3, required=False),
        ),
    )
    configured = DeploymentConfig(
        model_name="workspace.default.iris",
        endpoint_name="iris",
        promotion_policy=policy,
    )
    decision = evaluate_promotion_gate(
        {"test_f1_weighted": 0.94}, {"test_f1_weighted": 0.96}, configured
    )

    assert decision.passed is False
    assert decision.reason == "rule_failed:relative"
    assert decision.policy_version == "2"
    assert decision.rule_results[-1].status == "skipped"


def test_relative_rule_is_skipped_for_first_champion() -> None:
    policy = PromotionPolicy(
        "production",
        "1",
        (PromotionRule("relative", "score", ">=", baseline="champion"),),
    )
    configured = DeploymentConfig(
        model_name="workspace.default.iris",
        endpoint_name="iris",
        promotion_policy=policy,
    )
    decision = evaluate_promotion_gate({"score": 0.9}, None, configured)
    assert decision.passed and decision.rule_results[0].status == "skipped"


def test_declarative_policy_rejects_missing_and_non_finite_metrics() -> None:
    policy = PromotionPolicy(
        "production",
        "1",
        (PromotionRule("relative", "loss", "<=", baseline="champion", allowed_regression=0.1),),
    )
    configured = DeploymentConfig(
        model_name="workspace.default.iris", endpoint_name="iris", promotion_policy=policy
    )

    missing = evaluate_promotion_gate({}, {"loss": 0.2}, configured)
    missing_champion = evaluate_promotion_gate({"loss": 0.2}, {}, configured)
    passed = evaluate_promotion_gate({"loss": 0.25}, {"loss": 0.2}, configured)

    assert not missing.passed
    assert not missing_champion.passed
    assert passed.passed
    assert passed.as_dict()["rules"][0]["expected_value"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="debe ser finita"):
        evaluate_promotion_gate({"loss": float("nan")}, {"loss": 0.2}, configured)
    with pytest.raises(ValueError, match="Champion"):
        evaluate_promotion_gate({"loss": 0.2}, {"loss": float("inf")}, configured)
    with pytest.raises(ValueError, match="Operador"):
        deployment_module._compare(1, "==", 1)


class MissingEndpoint(RuntimeError):
    error_code = "RESOURCE_DOES_NOT_EXIST"


class ServingEndpoints:
    def __init__(self, endpoint: object | None = None, missing: bool = False) -> None:
        self.endpoint = endpoint
        self.missing = missing
        self.created: dict[str, object] = {}
        self.updated: dict[str, object] = {}
        self.deleted = ""

    def get(self, name: str) -> object:
        if self.missing:
            raise MissingEndpoint(name)
        return self.endpoint

    def create(self, **kwargs: object) -> None:
        self.created = kwargs

    def update_config(self, **kwargs: object) -> dict[str, object]:
        self.updated = kwargs
        return kwargs

    def delete(self, name: str) -> None:
        self.deleted = name


def workspace(endpoints: Any) -> SimpleNamespace:
    return SimpleNamespace(serving_endpoints=endpoints)


def test_capture_and_restore_endpoint_dictionary() -> None:
    endpoints = ServingEndpoints(
        {
            "config": {
                "served_entities": [{"entity_name": "old", "entity_version": "1"}],
                "traffic_config": {"routes": []},
            }
        }
    )
    snapshot = capture_serving_endpoint(workspace(endpoints), "iris")
    assert snapshot is not None
    restore_serving_endpoint(workspace(endpoints), snapshot)
    assert endpoints.updated["traffic_config"] == {"routes": []}


def test_capture_endpoint_supports_sdk_models() -> None:
    entity = SimpleNamespace(as_dict=lambda: {"entity_name": "old", "entity_version": "1"})
    traffic = SimpleNamespace(as_dict=lambda: {"routes": []})
    endpoint = SimpleNamespace(
        config=SimpleNamespace(served_entities=[entity], traffic_config=traffic)
    )
    snapshot = capture_serving_endpoint(workspace(ServingEndpoints(endpoint)), "iris")
    assert snapshot is not None and snapshot.served_entities[0]["entity_name"] == "old"


def test_capture_endpoint_handles_missing_and_invalid_configuration() -> None:
    assert capture_serving_endpoint(workspace(ServingEndpoints(missing=True)), "iris") is None
    with pytest.raises(RuntimeError, match="configuración restaurable"):
        capture_serving_endpoint(workspace(ServingEndpoints({})), "iris")
    with pytest.raises(RuntimeError, match="entidades servidas"):
        capture_serving_endpoint(
            workspace(ServingEndpoints({"config": {"served_entities": []}})), "iris"
        )
    invalid = SimpleNamespace(
        config=SimpleNamespace(served_entities=[object()], traffic_config=None)
    )
    with pytest.raises(TypeError, match="serializar"):
        capture_serving_endpoint(workspace(ServingEndpoints(invalid)), "iris")


def test_capture_endpoint_propagates_unexpected_errors() -> None:
    class ForbiddenEndpoints(ServingEndpoints):
        def get(self, name: str) -> object:
            raise PermissionError(name)

    with pytest.raises(PermissionError):
        capture_serving_endpoint(workspace(ForbiddenEndpoints()), "iris")


def test_upsert_creates_and_rolls_back_missing_endpoint() -> None:
    endpoints = ServingEndpoints(missing=True)
    change = upsert_serving_endpoint(
        workspace(endpoints), endpoint_name="iris", model_name="w.d.iris", model_version="2"
    )
    rollback_serving_endpoint(workspace(endpoints), change)
    assert change.created is True
    assert endpoints.created["config"] == {
        "served_entities": [
            {
                "entity_name": "w.d.iris",
                "entity_version": "2",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    }
    assert endpoints.deleted == "iris"


def test_upsert_updates_and_restores_existing_endpoint() -> None:
    endpoints = ServingEndpoints(
        {"config": {"served_entities": [{"entity_name": "old", "entity_version": "1"}]}}
    )
    change = upsert_serving_endpoint(
        workspace(endpoints),
        endpoint_name="iris",
        model_name="w.d.iris",
        model_version="2",
        workload_size="Medium",
        scale_to_zero_enabled=False,
    )
    assert endpoints.updated["served_entities"][0]["workload_size"] == "Medium"  # type: ignore[index]
    rollback_serving_endpoint(workspace(endpoints), change)
    assert endpoints.updated["served_entities"] == [{"entity_name": "old", "entity_version": "1"}]


def test_rollback_rejects_existing_endpoint_without_snapshot() -> None:
    with pytest.raises(RuntimeError, match="requiere snapshot"):
        rollback_serving_endpoint(
            workspace(ServingEndpoints()), ServingEndpointChange("iris", False, None)
        )


class StatefulEndpoints:
    def __init__(self, states: list[tuple[str, str | None]]) -> None:
        self.states = states

    def get(self, name: str) -> object:
        ready, update = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(state=SimpleNamespace(ready=ready, config_update=update))


def test_wait_for_endpoint_ready_succeeds() -> None:
    client = workspace(StatefulEndpoints([("UPDATING", None), ("READY", None)]))
    assert wait_for_endpoint_ready(client, "iris", timeout_seconds=1, poll_seconds=0)


@pytest.mark.parametrize("ready", ["NOT_READY", "FAILED"])
def test_wait_for_endpoint_ready_rejects_terminal_state(ready: str) -> None:
    with pytest.raises(RuntimeError, match="no está listo"):
        wait_for_endpoint_ready(
            workspace(StatefulEndpoints([(ready, None)])), "iris", timeout_seconds=1, poll_seconds=0
        )


def test_wait_for_endpoint_ready_rejects_failed_update() -> None:
    with pytest.raises(RuntimeError, match="Actualización"):
        wait_for_endpoint_ready(
            workspace(StatefulEndpoints([("UPDATING", "FAILED")])),
            "iris",
            timeout_seconds=1,
            poll_seconds=0,
        )


def test_wait_for_endpoint_ready_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0.0, 2.0])
    monkeypatch.setattr("iris_mlflow_utils.deployment.time.monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="no quedó READY"):
        wait_for_endpoint_ready(
            workspace(StatefulEndpoints([("UPDATING", None)])),
            "iris",
            timeout_seconds=1,
            poll_seconds=0,
        )


def test_promote_champion_updates_current_and_previous_lifecycle() -> None:
    calls: list[tuple[str, ...]] = []
    registry = SimpleNamespace(
        set_model_version_tag=lambda *args: calls.append(tuple(str(value) for value in args)),
        set_registered_model_alias=lambda **kwargs: calls.append(
            (kwargs["alias"], kwargs["version"])
        ),
    )
    promote_champion(registry, model_name="iris", model_version="2", previous_champion_version="1")
    assert ("Champion", "2") in calls
    assert ("iris", "1", "lifecycle", "previous_champion") in calls


def test_promote_champion_does_not_supersede_same_version() -> None:
    calls: list[tuple[str, ...]] = []
    registry = SimpleNamespace(
        set_model_version_tag=lambda *args: calls.append(tuple(str(value) for value in args)),
        set_registered_model_alias=lambda **kwargs: None,
    )
    promote_champion(registry, model_name="iris", model_version="2", previous_champion_version="2")
    assert not any(
        call[1:3] == ("2", "deployment_status") and call[-1] == "superseded" for call in calls
    )
