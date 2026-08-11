"""Tests for evaluation artifacts and promotion gates."""

import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from iris_mlflow_utils import (
    approve_locally,
    build_evaluation_artifacts,
    build_probability_metrics,
    detect_runtime,
    evaluate_model,
    evaluate_promotion_gate,
    load_dataset_for_runtime,
    simulate_local_deployment,
)
from iris_mlflow_utils.config import DeploymentConfig, build_deployment_config
from sklearn.datasets import load_iris  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]


def test_multiclass_artifacts_include_roc_lift_gain_and_confusion(
    tmp_path: Path,
) -> None:
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = np.asarray(iris.target)
    model = LogisticRegression(max_iter=500).fit(features, target)
    result = evaluate_model(model, features, target, [0, 1, 2])

    metrics = build_probability_metrics(result, target, [0, 1, 2])
    paths = build_evaluation_artifacts(
        model,
        result,
        labels=[0, 1, 2],
        class_names=tuple(iris.target_names),
        features=features,
        target=target,
        output_dir=tmp_path / "evaluation",
    )

    assert metrics["roc_auc_ovr_macro"] > 0.9
    assert {
        "confusion_matrix.png",
        "roc_curve_by_class.png",
        "lift_curve_by_class.png",
    }.issubset(paths)
    assert (tmp_path / "evaluation" / "cumulative_gain_data.json").is_file()
    assert (tmp_path / "evaluation" / "feature_importance.json").is_file() is False


def test_promotion_gate_blocks_regression_and_accepts_candidate() -> None:
    config = DeploymentConfig(
        model_name="workspace.default.iris_classifier",
        endpoint_name="iris-classifier",
    )
    accepted = evaluate_promotion_gate(
        {"test_f1_weighted": 0.95, "test_accuracy": 0.96},
        {"test_f1_weighted": 0.94},
        config,
    )
    rejected = evaluate_promotion_gate(
        {"test_f1_weighted": 0.90, "test_accuracy": 0.95},
        {"test_f1_weighted": 0.95},
        config,
    )

    assert accepted.passed is True
    assert rejected.passed is False
    assert rejected.reason == "candidate_regresses_against_champion"


def test_deployment_notebooks_keep_only_dynamic_job_inputs() -> None:
    repository_root = Path(__file__).parents[1]
    for name in ("evaluate_model.ipynb", "approval.ipynb", "deploy_model.ipynb"):
        notebook = json.loads(
            (repository_root / "deployment" / name).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "model_name" in source
        assert "model_version" in source
        assert "IRIS_DATA_PATH" not in source
        assert (
            notebook["metadata"]["application/vnd.databricks.v1+notebook"]["widgets"]
            == {}
        )


def test_deployment_job_uses_databricks_approval_task_name() -> None:
    repository_root = Path(__file__).parents[1]
    notebook = json.loads(
        (repository_root / "deployment" / "create_deployment_job.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "'task_key': 'Approval_Check'" in source
    assert "'task_key': 'approval_model'" not in source
    assert "'depends_on': [{'task_key': 'Approval_Check'}]" in source


def test_deployment_config_comes_from_versioned_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_RUNTIME", "databricks")
    config = build_deployment_config()

    assert config.model_name == "workspace.default.iris_classifier"
    assert config.endpoint_name == "iris-classifier"
    assert config.min_test_f1_weighted == 0.90
    assert config.required_approval_tag == "Approval_Check"
    assert config.notebook_root == "/Workspace/Shared/mlflow_deployment"
    assert config.job_name == "model-deployment"


def test_runtime_detection_prefers_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_RUNTIME", "local")
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")

    assert detect_runtime() == "local"


def test_runtime_detection_identifies_dbutils(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_RUNTIME", raising=False)
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    monkeypatch.setattr(
        builtins,
        "get_ipython",
        lambda: type("Shell", (), {"user_ns": {"dbutils": object()}})(),
        raising=False,
    )

    assert detect_runtime() == "databricks"


def test_local_dataset_loader_never_requires_spark() -> None:
    config = build_runtime_config_for_test()
    dataset = load_dataset_for_runtime("local", spark=None, config=config)

    expected_columns = {
        "Id",
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
        "Species",
    }

    assert len(dataset.dataframe) > 0
    assert set(dataset.dataframe.columns) == expected_columns
    assert set(dataset.classes) == {
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    }


def test_local_dataset_matches_feature_contract() -> None:
    repository_root = Path(__file__).parents[1]
    dataset_path = repository_root / "data" / "local" / "iris_features.csv"
    config = build_runtime_config_for_test()
    dataset = load_dataset_for_runtime("local", spark=None, config=config)
    feature_columns = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]

    assert dataset_path.is_file()
    assert not dataset.dataframe["Id"].duplicated().any()
    assert not dataset.dataframe.isna().any().any()
    assert all(
        pd.api.types.is_numeric_dtype(dataset.dataframe[column])
        for column in feature_columns
    )
    assert set(dataset.dataframe["Species"]) == {
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    }


def test_local_approval_and_deployment_write_manifest(tmp_path: Path) -> None:
    class FakeRegistry:
        def __init__(self) -> None:
            self.tags: dict[str, str] = {}
            self.aliases: dict[str, str] = {}

        def set_model_version_tag(
            self, name: str, version: str, key: str, value: str
        ) -> None:
            self.tags[key] = value

        def set_registered_model_alias(
            self, name: str, alias: str, version: str
        ) -> None:
            self.aliases[alias] = version

    client = FakeRegistry()
    assert (
        approve_locally(client, model_name="iris_classifier", model_version="3")[
            "Approval_Check"
        ]
        == "Approved"
    )
    payload = simulate_local_deployment(
        client,
        model_name="iris_classifier",
        model_version="3",
        champion_alias="Champion",
        manifest_path=tmp_path / "manifest.json",
        smoke_test_passed=True,
    )

    assert payload["deployment_skipped"] is True
    assert client.aliases["Champion"] == "3"
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["smoke_test"] == "passed"
    )


def build_runtime_config_for_test() -> object:
    from iris_mlflow_utils.config import build_config

    return build_config(model_slug="random_forest")
