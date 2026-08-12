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
    capture_serving_endpoint,
    detect_runtime,
    ensure_deployment_job,
    evaluate_model,
    evaluate_promotion_gate,
    load_dataset_for_runtime,
    promote_champion,
    restore_serving_endpoint,
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
    assert metrics["log_loss"] >= 0.0
    assert {
        "confusion_matrix.png",
        "roc_curve_by_class.png",
        "lift_curve_by_class.png",
        "lift_curve_macro.png",
    }.issubset(paths)
    assert (tmp_path / "evaluation" / "cumulative_gain_data.json").is_file()
    gain = pd.read_json(tmp_path / "evaluation" / "cumulative_gain_data.json")
    final_gain = gain.groupby("class_id", as_index=False).tail(1)
    assert np.allclose(final_gain["cumulative_gain"], 1.0)
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


def test_evaluation_notebook_links_run_and_logged_model() -> None:
    repository_root = Path(__file__).parents[1]
    notebook = json.loads(
        (repository_root / "deployment" / "evaluate_model.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "ensure_mlflow_experiment" in source
    assert "model_id=logged_model_id" in source
    assert "evaluation_run_id" in source
    assert "evaluation_model_id" in source


def test_deployment_notebook_restores_endpoint_before_failing() -> None:
    repository_root = Path(__file__).parents[1]
    notebook = json.loads(
        (repository_root / "deployment" / "deploy_model.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "capture_serving_endpoint" in source
    assert "restore_serving_endpoint" in source
    assert "rollback_status" in source
    rollback_branch = source[source.index("except Exception as deployment_error") :]
    assert rollback_branch.index("restore_serving_endpoint") < rollback_branch.rindex(
        "raise"
    )


def test_bundle_defines_governed_deployment_job() -> None:
    repository_root = Path(__file__).parents[1]
    bundle = (repository_root / "databricks.yml").read_text(encoding="utf-8")

    assert "task_key: Approval_Check" in bundle
    assert "task_key: approval_model" not in bundle
    assert "max_retries: 0" in bundle
    assert "max_concurrent_runs: 1" in bundle
    assert "service_principal_name: ${var.production_service_principal}" in bundle
    assert "deployment_job_id: ${resources.jobs.model_deployment.id}" in bundle


def test_connector_notebook_does_not_create_or_reset_jobs() -> None:
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

    assert "deployment_job_id" in source
    assert "update_registered_model" in source
    assert "ensure_deployment_job" not in source
    assert "IRIS_DEPLOYMENT_CLUSTER_ID" not in source


def test_bundle_ci_uses_oidc_without_long_lived_secret() -> None:
    repository_root = Path(__file__).parents[1]
    workflow = (
        repository_root / ".github" / "workflows" / "databricks-bundle.yml"
    ).read_text(encoding="utf-8")

    assert "DATABRICKS_AUTH_TYPE: github-oidc" in workflow
    assert "id-token: write" in workflow
    assert "DATABRICKS_CLIENT_SECRET" not in workflow
    assert "databricks bundle deploy -t prod" in workflow


def test_serving_snapshot_can_restore_exact_configuration() -> None:
    class FakeServingEndpoints:
        def __init__(self) -> None:
            self.updated: dict[str, object] = {}

        @staticmethod
        def get(name: str) -> dict[str, object]:
            return {
                "config": {
                    "served_entities": [
                        {
                            "entity_name": "workspace.default.iris_classifier",
                            "entity_version": "1",
                            "workload_size": "Small",
                            "scale_to_zero_enabled": True,
                        }
                    ],
                    "traffic_config": {
                        "routes": [
                            {"served_model_name": "iris-1", "traffic_percentage": 100}
                        ]
                    },
                }
            }

        def update_config(self, **kwargs: object) -> dict[str, object]:
            self.updated = kwargs
            return kwargs

    class FakeWorkspace:
        serving_endpoints = FakeServingEndpoints()

    workspace = FakeWorkspace()
    snapshot = capture_serving_endpoint(workspace, "iris-classifier")
    restore_serving_endpoint(workspace, snapshot)

    assert snapshot.served_entities[0]["entity_version"] == "1"
    assert workspace.serving_endpoints.updated["name"] == "iris-classifier"
    assert (
        workspace.serving_endpoints.updated["traffic_config"] == snapshot.traffic_config
    )


def test_champion_promotion_updates_lifecycle_tags() -> None:
    class FakeRegistry:
        def __init__(self) -> None:
            self.tags: dict[tuple[str, str], str] = {}
            self.aliases: dict[str, str] = {}

        def set_model_version_tag(
            self, name: str, version: str, key: str, value: str
        ) -> None:
            self.tags[(version, key)] = value

        def set_registered_model_alias(
            self, name: str, alias: str, version: str
        ) -> None:
            self.aliases[alias] = version

    registry = FakeRegistry()
    promote_champion(
        registry,
        model_name="iris_classifier",
        model_version="2",
        previous_champion_version="1",
    )

    assert registry.aliases["Champion"] == "2"
    assert registry.tags[("2", "smoke_test_status")] == "passed"
    assert registry.tags[("2", "deployment_status")] == "deployed"
    assert registry.tags[("2", "lifecycle")] == "champion"
    assert registry.tags[("1", "lifecycle")] == "previous_champion"
    assert registry.tags[("1", "deployment_status")] == "superseded"


def test_deployment_job_is_created_or_reset_idempotently() -> None:
    class FakeApiClient:
        def __init__(self, jobs: list[dict[str, object]]) -> None:
            self.jobs = jobs
            self.calls: list[tuple[str, str, dict[str, object]]] = []

        def do(self, method: str, path: str, **kwargs: object) -> dict[str, object]:
            self.calls.append((method, path, kwargs))
            if path.endswith("/list"):
                return {"jobs": self.jobs}
            if path.endswith("/create"):
                return {"job_id": 101}
            return {}

    settings = {"name": "model-deployment", "tasks": []}
    new_client = FakeApiClient([])
    existing_client = FakeApiClient(
        [{"job_id": 77, "settings": {"name": "model-deployment"}}]
    )

    assert ensure_deployment_job(
        new_client, job_name="model-deployment", settings=settings
    ) == ("101", "created")
    assert ensure_deployment_job(
        existing_client, job_name="model-deployment", settings=settings
    ) == ("77", "updated")
    assert existing_client.calls[-1][1].endswith("/reset")


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


def test_local_environment_example_is_self_consistent() -> None:
    repository_root = Path(__file__).parents[1]
    values = {}
    for line in (
        (repository_root / "config" / "local.env.example")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value

    assert values["IRIS_RUNTIME"] == "local"
    assert values["IRIS_REGISTERED_MODEL_NAME"] == "iris_classifier"
    assert values["IRIS_DEPLOYMENT_MODEL_NAME"] == "iris_classifier"
    assert values["IRIS_EXPERIMENT_NAME"] == "iris_mlflow_local"


def test_notebooks_are_clean_and_have_cell_ids() -> None:
    repository_root = Path(__file__).parents[1]
    notebooks = [
        *repository_root.glob("*.ipynb"),
        *(repository_root / "deployment").glob("*.ipynb"),
        *(repository_root / "tools").glob("*.ipynb"),
    ]

    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert all(cell.get("id") for cell in notebook["cells"]), notebook_path
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, notebook_path
                assert cell.get("outputs") == [], notebook_path


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
    assert payload["rollback_available"] is False
    assert payload["rollback_status"] == "not_required"
    assert client.aliases["Champion"] == "3"
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["smoke_test"] == "passed"
    )


def build_runtime_config_for_test() -> object:
    from iris_mlflow_utils.config import build_config

    return build_config(model_slug="random_forest")
