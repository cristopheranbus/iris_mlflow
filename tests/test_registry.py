"""Tests for model registry metadata synchronization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from iris_mlflow_utils.config import TrainingConfig
from iris_mlflow_utils.registry import (
    ensure_mlflow_experiment,
    synchronize_model_registry_metadata,
)


class FakeModelVersion:
    def __init__(self, version: str, tags: dict[str, str], description: str = "") -> None:
        self.version = version
        self.tags = tags
        self.description = description


class FakeRegisteredModel:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}
        self.description = ""


class FakeClient:
    _registry_uri = "databricks-uc"

    def __init__(self) -> None:
        self.model = FakeRegisteredModel()
        self.version = FakeModelVersion("3", {})
        self.alias_version = ""

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

    def get_registered_model(self, name: str) -> FakeRegisteredModel:
        return self.model

    def get_model_version(self, name: str, version: str) -> FakeModelVersion:
        return self.version

    def get_model_version_by_alias(self, name: str, alias: str) -> FakeModelVersion:
        self.version.version = self.alias_version
        return self.version


def test_registry_metadata_sets_model_type_tags_and_descriptions() -> None:
    config = TrainingConfig(
        dataset_path=None,
        experiment_name="iris_mlflow",
        artifact_location="",
        tracking_uri="",
        registry_uri="databricks-uc",
        registered_model_name="workspace.default.iris_classifier",
        feature_table="workspace.default.iris_features",
        champion_alias="Champion",
        challenger_alias="Challenger",
        run_name="test",
        dataset_version="iris-features-delta",
        project_version="test",
        author="test",
        purpose="test",
        model_type="xgboost",
        model_framework="xgboost",
    )
    client = FakeClient()

    evidence = synchronize_model_registry_metadata(
        client, config=config, version="3", run_id="run-123"
    )

    assert evidence["alias_verified"] is True
    assert evidence["version_tags_verified"] is True
    assert evidence["registered_model_tags_verified"] is True
    assert client.version.tags["model_type"] == "xgboost"
    assert client.model.tags["model_type"] == "xgboost"
    assert client.version.description
    assert client.model.description


def test_ensure_mlflow_experiment_restores_deleted_experiment(tmp_path: Path) -> None:
    database_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_id = ensure_mlflow_experiment("iris_mlflow_local", tracking_uri=database_uri)
    client = mlflow.MlflowClient(tracking_uri=database_uri)
    client.delete_experiment(experiment_id)

    restored_id = ensure_mlflow_experiment("iris_mlflow_local", tracking_uri=database_uri)

    assert restored_id == experiment_id
    assert client.get_experiment(experiment_id).lifecycle_stage == "active"
