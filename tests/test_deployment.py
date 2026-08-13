"""Tests for evaluation artifacts and promotion gates."""

import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from iris_mlflow_utils import (
    approve_locally,
    build_evaluation_artifacts,
    build_probability_metrics,
    capture_serving_endpoint,
    detect_runtime,
    evaluate_model,
    evaluate_promotion_gate,
    get_delta_table_version,
    get_model_evaluation_metrics,
    load_dataset_for_runtime,
    promote_champion,
    restore_serving_endpoint,
    rollback_serving_endpoint,
    simulate_local_deployment,
    upsert_serving_endpoint,
)
from iris_mlflow_utils.config import DeploymentConfig, build_deployment_config


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
        notebook = json.loads((repository_root / "deployment" / name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert "model_name" in source
        assert "model_version" in source
        assert "IRIS_DATA_PATH" not in source
        assert notebook["metadata"]["application/vnd.databricks.v1+notebook"]["widgets"] == {}


def test_evaluation_notebook_links_run_and_logged_model() -> None:
    repository_root = Path(__file__).parents[1]
    notebook = json.loads(
        (repository_root / "deployment" / "evaluate_model.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "ensure_mlflow_experiment" in source
    assert "model_id=logged_model_id" in source
    assert "evaluation_run_id" in source
    assert "evaluation_model_id" in source


def test_deployment_notebook_rolls_back_endpoint_before_failing() -> None:
    repository_root = Path(__file__).parents[1]
    notebook = json.loads(
        (repository_root / "deployment" / "deploy_model.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "upsert_serving_endpoint" in source
    assert "rollback_serving_endpoint" in source
    assert "rollback_status" in source
    rollback_branch = source[source.index("except Exception as deployment_error") :]
    assert rollback_branch.index("rollback_serving_endpoint") < rollback_branch.rindex("raise")


def test_bundle_defines_governed_deployment_job() -> None:
    repository_root = Path(__file__).parents[1]
    bundle = (repository_root / "databricks.yml").read_text(encoding="utf-8")

    assert "task_key: Approval_Check" in bundle
    assert "task_key: approval_model" not in bundle
    assert "max_retries: 0" in bundle
    assert "max_concurrent_runs: 1" in bundle
    assert "service_principal_name: ${var.production_service_principal}" in bundle
    assert "deployment_job_id: ${resources.jobs.model_deployment.id}" in bundle
    assert "workspace.default.iris_classifier_dev" in bundle
    assert "serving_endpoint_name: iris-classifier-dev" in bundle
    assert "serving_endpoint_name: iris-classifier" in bundle


def test_connector_notebook_does_not_create_or_reset_jobs() -> None:
    repository_root = Path(__file__).parents[1]
    notebook = json.loads(
        (repository_root / "deployment" / "create_deployment_job.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "deployment_job_id" in source
    assert "update_registered_model" in source
    assert "ensure_deployment_job" not in source
    assert "IRIS_DEPLOYMENT_CLUSTER_ID" not in source


def test_bundle_ci_uses_oidc_without_long_lived_secret() -> None:
    repository_root = Path(__file__).parents[1]
    workflow = (repository_root / ".github" / "workflows" / "databricks-bundle.yml").read_text(
        encoding="utf-8"
    )

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
                        "routes": [{"served_model_name": "iris-1", "traffic_percentage": 100}]
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
    assert snapshot is not None
    restore_serving_endpoint(workspace, snapshot)

    assert snapshot.served_entities[0]["entity_version"] == "1"
    assert workspace.serving_endpoints.updated["name"] == "iris-classifier"
    assert workspace.serving_endpoints.updated["traffic_config"] == snapshot.traffic_config


def test_upsert_creates_missing_endpoint_and_rollback_deletes_it() -> None:
    class MissingEndpoint(RuntimeError):
        error_code = "RESOURCE_DOES_NOT_EXIST"

    class FakeServingEndpoints:
        def __init__(self) -> None:
            self.created: dict[str, object] = {}
            self.deleted = ""

        @staticmethod
        def get(name: str) -> object:
            raise MissingEndpoint(name)

        def create(self, **kwargs: object) -> None:
            self.created = kwargs

        def delete(self, name: str) -> None:
            self.deleted = name

    class FakeWorkspace:
        serving_endpoints = FakeServingEndpoints()

    workspace = FakeWorkspace()
    change = upsert_serving_endpoint(
        workspace,
        endpoint_name="iris-classifier-dev",
        model_name="workspace.default.iris_classifier_dev",
        model_version="1",
    )
    rollback_serving_endpoint(workspace, change)

    assert change.created is True
    assert workspace.serving_endpoints.created["name"] == "iris-classifier-dev"
    assert workspace.serving_endpoints.deleted == "iris-classifier-dev"


def test_upsert_updates_existing_endpoint_and_retains_snapshot() -> None:
    class FakeServingEndpoints:
        def __init__(self) -> None:
            self.updated: dict[str, object] = {}

        @staticmethod
        def get(name: str) -> dict[str, object]:
            return {"config": {"served_entities": [{"entity_name": "old", "entity_version": "1"}]}}

        def update_config(self, **kwargs: object) -> None:
            self.updated = kwargs

    class FakeWorkspace:
        serving_endpoints = FakeServingEndpoints()

    workspace = FakeWorkspace()
    change = upsert_serving_endpoint(
        workspace,
        endpoint_name="iris-classifier",
        model_name="workspace.default.iris_classifier",
        model_version="2",
    )

    assert change.created is False
    assert change.snapshot is not None
    assert workspace.serving_endpoints.updated["served_entities"][0]["entity_version"] == "2"  # type: ignore[index]


def test_rollback_restores_an_updated_endpoint() -> None:
    class FakeServingEndpoints:
        def __init__(self) -> None:
            self.updated: dict[str, object] = {}

        @staticmethod
        def get(name: str) -> dict[str, object]:
            return {"config": {"served_entities": [{"entity_name": "old", "entity_version": "1"}]}}

        def update_config(self, **kwargs: object) -> None:
            self.updated = kwargs

    class FakeWorkspace:
        serving_endpoints = FakeServingEndpoints()

    workspace = FakeWorkspace()
    change = upsert_serving_endpoint(
        workspace,
        endpoint_name="iris-classifier",
        model_name="workspace.default.iris_classifier",
        model_version="2",
    )
    rollback_serving_endpoint(workspace, change)
    assert workspace.serving_endpoints.updated["served_entities"] == [
        {"entity_name": "old", "entity_version": "1"}
    ]


def test_wait_for_endpoint_ready_handles_success_failure_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris_mlflow_utils import wait_for_endpoint_ready

    class FakeServingEndpoints:
        def __init__(self, states: list[str]) -> None:
            self.states = states

        def get(self, name: str) -> object:
            ready = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return type(
                "Endpoint",
                (),
                {"state": type("State", (), {"ready": ready, "config_update": None})()},
            )()

    class FakeWorkspace:
        def __init__(self, states: list[str]) -> None:
            self.serving_endpoints = FakeServingEndpoints(states)

    assert wait_for_endpoint_ready(
        FakeWorkspace(["NOT_READY_YET", "READY"]),
        "iris",
        timeout_seconds=1,
        poll_seconds=0,
    )
    with pytest.raises(RuntimeError, match="no está listo"):
        wait_for_endpoint_ready(
            FakeWorkspace(["FAILED"]), "iris", timeout_seconds=1, poll_seconds=0
        )
    times = iter([0.0, 2.0])
    monkeypatch.setattr("iris_mlflow_utils.deployment.time.monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="no quedó READY"):
        wait_for_endpoint_ready(
            FakeWorkspace(["UPDATING"]), "iris", timeout_seconds=1, poll_seconds=0
        )


def test_capture_endpoint_rejects_non_restorable_configuration() -> None:
    class FakeWorkspace:
        serving_endpoints = type(
            "Serving", (), {"get": staticmethod(lambda name: {"config": {"served_entities": []}})}
        )()

    with pytest.raises(RuntimeError, match="no contiene entidades"):
        capture_serving_endpoint(FakeWorkspace(), "empty")


def test_champion_promotion_updates_lifecycle_tags() -> None:
    class FakeRegistry:
        def __init__(self) -> None:
            self.tags: dict[tuple[str, str], str] = {}
            self.aliases: dict[str, str] = {}

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            self.tags[(version, key)] = value

        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
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


def test_deployment_config_comes_from_versioned_toml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_RUNTIME", "databricks")
    config = build_deployment_config()

    assert config.model_name == "workspace.default.iris_classifier"
    assert config.endpoint_name == "iris-classifier"
    assert config.min_test_f1_weighted == 0.90
    assert config.required_approval_tag == "Approval_Check"


def test_delta_version_is_read_from_history() -> None:
    class FakeResult:
        @staticmethod
        def collect() -> list[dict[str, int]]:
            return [{"version": 7}]

    class FakeSpark:
        def sql(self, query: str) -> FakeResult:
            assert query == "DESCRIBE HISTORY workspace.default.iris_features LIMIT 1"
            return FakeResult()

    assert get_delta_table_version(FakeSpark(), "workspace.default.iris_features") == "7"


def test_champion_metrics_come_from_linked_evaluation_run() -> None:
    class FakeClient:
        @staticmethod
        def get_model_version(name: str, version: str) -> object:
            return type("Version", (), {"tags": {"evaluation_run_id": "eval-123"}})()

        @staticmethod
        def get_run(run_id: str) -> object:
            assert run_id == "eval-123"
            return type(
                "Run",
                (),
                {"data": type("Data", (), {"metrics": {"test_f1_weighted": 0.94}})()},
            )()

    metrics, run_id = get_model_evaluation_metrics(
        FakeClient(), model_name="iris_classifier", model_version="2"
    )
    assert run_id == "eval-123"
    assert metrics["test_f1_weighted"] == 0.94


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
        pd.api.types.is_numeric_dtype(dataset.dataframe[column]) for column in feature_columns
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
        (repository_root / "config" / "local.env.example").read_text(encoding="utf-8").splitlines()
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
                compile("".join(cell["source"]), str(notebook_path), "exec")


def test_local_approval_and_deployment_write_manifest(tmp_path: Path) -> None:
    class FakeRegistry:
        def __init__(self) -> None:
            self.tags: dict[str, str] = {}
            self.aliases: dict[str, str] = {}

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            self.tags[key] = value

        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
            self.aliases[alias] = version

    client = FakeRegistry()
    assert (
        approve_locally(client, model_name="iris_classifier", model_version="3")["Approval_Check"]
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
    assert json.loads((tmp_path / "manifest.json").read_text())["smoke_test"] == "passed"


def build_runtime_config_for_test() -> object:
    from iris_mlflow_utils.config import build_config

    return build_config(model_slug="random_forest")
