"""Unit tests for registry metadata and governed metrics."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_mlflow_utils.config import TrainingConfig
from iris_mlflow_utils.registry import (
    build_registry_client,
    ensure_mlflow_experiment,
    get_model_evaluation_metrics,
    synchronize_model_registry_metadata,
)

pytestmark = pytest.mark.unit


def config(runtime_mode: str = "databricks") -> TrainingConfig:
    local = runtime_mode == "local"
    return TrainingConfig(
        dataset_path=Path("iris.csv") if local else None,
        experiment_name="iris",
        artifact_location="",
        tracking_uri="",
        registry_uri="sqlite" if local else "databricks-uc",
        registered_model_name="iris_classifier" if local else "workspace.default.iris",
        feature_table="workspace.default.features",
        champion_alias="Champion",
        challenger_alias="Challenger",
        run_name="test",
        dataset_version="1",
        project_version="2",
        author="owner",
        purpose="test",
        model_type="xgboost",
        model_framework="xgboost",
        runtime_mode=runtime_mode,  # type: ignore[arg-type]
    )


class FakeClient:
    _registry_uri = "databricks-uc"

    def __init__(self, *, corrupt: bool = False) -> None:
        self.model = SimpleNamespace(tags={}, description="")
        self.version = SimpleNamespace(version="3", tags={}, description="")
        self.alias_version = ""
        self.corrupt = corrupt

    def set_model_version_tag(self, name: str, version: str, key: str, value: Any) -> None:
        self.version.tags[key] = str(value)

    def set_registered_model_tag(self, name: str, key: str, value: Any) -> None:
        self.model.tags[key] = str(value)

    def update_registered_model(self, *, name: str, description: str) -> None:
        self.model.description = description

    def update_model_version(self, *, name: str, version: str, description: str) -> None:
        self.version.description = description

    def set_registered_model_alias(self, *, name: str, alias: str, version: str) -> None:
        self.alias_version = version

    def get_registered_model(self, name: str) -> object:
        return self.model

    def get_model_version(self, name: str, version: str) -> object:
        if self.corrupt:
            self.version.tags.pop("model_type", None)
        return self.version

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        return SimpleNamespace(version=self.alias_version)


@pytest.mark.parametrize(
    "runtime_mode, source", [("databricks", "unity_catalog"), ("local", "local_file")]
)
def test_registry_metadata_sets_and_verifies_governance(runtime_mode: str, source: str) -> None:
    evidence = synchronize_model_registry_metadata(
        FakeClient(), config=config(runtime_mode), version="3", run_id="run-123"
    )
    assert evidence["alias_verified"] is True
    assert evidence["observed_version_tags"]["feature_source"] == source
    assert evidence["verified_at"]


def test_registry_metadata_rejects_failed_verification() -> None:
    with pytest.raises(RuntimeError, match="no pudo verificarse"):
        synchronize_model_registry_metadata(
            FakeClient(corrupt=True), config=config(), version="3", run_id="run-123"
        )


def test_get_model_evaluation_metrics_reads_linked_run() -> None:
    client = SimpleNamespace(
        get_model_version=lambda name, version: SimpleNamespace(tags={"evaluation_run_id": "run"}),
        get_run=lambda run_id: SimpleNamespace(data=SimpleNamespace(metrics={"test_f1": 0.94})),
    )
    metrics, run_id = get_model_evaluation_metrics(client, model_name="iris", model_version=2)
    assert (metrics["test_f1"], run_id) == (0.94, "run")


def test_get_model_evaluation_metrics_rejects_missing_link_or_metrics() -> None:
    no_link = SimpleNamespace(get_model_version=lambda name, version: SimpleNamespace(tags={}))
    with pytest.raises(RuntimeError, match="evaluation_run_id"):
        get_model_evaluation_metrics(no_link, model_name="iris", model_version=2)

    no_metrics = SimpleNamespace(
        get_model_version=lambda name, version: SimpleNamespace(tags={"evaluation_run_id": "run"}),
        get_run=lambda run_id: SimpleNamespace(data=SimpleNamespace(metrics={})),
    )
    with pytest.raises(RuntimeError, match="no contiene métricas"):
        get_model_evaluation_metrics(no_metrics, model_name="iris", model_version=2)


def test_build_registry_client_uses_explicit_registry_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "iris_mlflow_utils.registry.MlflowClient",
        lambda registry_uri: captured.setdefault("registry_uri", registry_uri),
    )
    assert build_registry_client("sqlite:///registry.db") is not None
    assert captured["registry_uri"] == "sqlite:///registry.db"


@pytest.mark.parametrize("tracking_uri", [None, "file:///isolated-mlruns"])
def test_ensure_experiment_supports_default_and_non_sqlite_tracking(
    tracking_uri: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class ExperimentClient:
        def __init__(self, **kwargs: object) -> None:
            calls["client_kwargs"] = kwargs

        @staticmethod
        def get_experiment_by_name(name: str) -> None:
            return None

        @staticmethod
        def create_experiment(name: str, artifact_location: str | None) -> str:
            calls["create"] = (name, artifact_location)
            return "42"

    monkeypatch.setattr("iris_mlflow_utils.registry.MlflowClient", ExperimentClient)
    monkeypatch.setattr(
        "iris_mlflow_utils.registry.mlflow.set_tracking_uri",
        lambda uri: calls.setdefault("tracking_uri", uri),
    )
    monkeypatch.setattr(
        "iris_mlflow_utils.registry.mlflow.set_experiment",
        lambda **kwargs: calls.setdefault("experiment", kwargs),
    )

    assert ensure_mlflow_experiment("iris", tracking_uri=tracking_uri) == "42"
    assert calls["client_kwargs"] == ({"tracking_uri": tracking_uri} if tracking_uri else {})
    assert ("tracking_uri" in calls) is (tracking_uri is not None)
