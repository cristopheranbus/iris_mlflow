"""Contracts for clean, thin, externally configured notebooks."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def test_training_notebooks_use_shared_governed_helpers(
    repository_root: Path, notebook_source: Callable[[Path], str]
) -> None:
    for name in ("random_forest.ipynb", "xgboost.ipynb"):
        source = notebook_source(repository_root / "notebooks" / "training" / name)
        for required in (
            "load_dataset_for_runtime",
            "RUNTIME_MODE = detect_runtime()",
            "config.registered_model_name",
            "config.challenger_alias",
            "mlflow.register_model",
            "metadata/model_identity.json",
            "MODEL_TYPE = config.model_type",
            "MODEL_FRAMEWORK = config.model_framework",
            "build_probability_metrics",
        ):
            assert required in source
        for forbidden in (
            "ensure_feature_table",
            "DATABRICKS_SERVING_ENDPOINT_NAME",
            "DATABRICKS_SERVING_TRAFFIC_PERCENTAGE",
            '"precision_macro"',
        ):
            assert forbidden not in source


def test_deployment_notebooks_keep_dynamic_inputs_and_rollback_order(
    repository_root: Path, notebook_source: Callable[[Path], str]
) -> None:
    deployment = repository_root / "deployment"
    for name in ("evaluate_model.ipynb", "approval.ipynb", "deploy_model.ipynb"):
        source = notebook_source(deployment / name)
        assert "model_name" in source
        assert "model_version" in source
        assert "IRIS_DATA_PATH" not in source

    evaluation = notebook_source(deployment / "evaluate_model.ipynb")
    assert "ensure_mlflow_experiment" in evaluation
    assert "model_id=logged_model_id" in evaluation
    assert "evaluation_run_id" in evaluation

    deploy = notebook_source(deployment / "deploy_model.ipynb")
    rollback = deploy[deploy.index("except Exception as deployment_error") :]
    assert rollback.index("rollback_serving_endpoint") < rollback.rindex("raise")


def test_connector_only_associates_bundle_job(
    repository_root: Path, notebook_source: Callable[[Path], str]
) -> None:
    source = notebook_source(repository_root / "deployment" / "create_deployment_job.ipynb")
    assert "deployment_job_id" in source
    assert "update_registered_model" in source
    assert "ensure_deployment_job" not in source
    assert "IRIS_DEPLOYMENT_CLUSTER_ID" not in source


def test_endpoint_notebook_uses_external_configuration(
    repository_root: Path, notebook_source: Callable[[Path], str]
) -> None:
    source = notebook_source(repository_root / "notebooks" / "serving" / "test_endpoint.ipynb")
    assert "load_file_config" in source
    assert "dbutils.widgets" not in source
    assert "apiToken()" not in source


def test_notebooks_are_clean_compilable_and_have_ids(repository_root: Path) -> None:
    notebooks = [
        *(repository_root / "deployment").glob("*.ipynb"),
        *(repository_root / "notebooks").rglob("*.ipynb"),
    ]
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert all(cell.get("id") for cell in notebook["cells"]), path
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, path
                assert cell.get("outputs") == [], path
                compile("".join(cell["source"]), str(path), "exec")
