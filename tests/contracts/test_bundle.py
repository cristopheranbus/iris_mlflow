"""Structural contracts for the Databricks bundle."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.contract


def test_bundle_declares_governed_isolated_deployment_jobs(repository_root: Path) -> None:
    bundle = yaml.safe_load((repository_root / "databricks.yml").read_text(encoding="utf-8"))
    job = bundle["resources"]["jobs"]["model_deployment"]
    task_keys = [task["task_key"] for task in job["tasks"]]

    assert task_keys == ["evaluate_model", "Approval_Check", "deploy_model"]
    assert job["max_concurrent_runs"] == 1
    assert job["tasks"][1]["max_retries"] == 0
    assert bundle["targets"]["dev"]["variables"]["serving_endpoint_name"] == "iris-classifier-dev"
    assert bundle["targets"]["prod"]["variables"]["serving_endpoint_name"] == "iris-classifier"
    assert bundle["targets"]["prod"]["resources"]["jobs"]["model_deployment"]["permissions"]
    assert bundle["artifacts"]["iris_mlflow_tools"]["path"] == "."
    assert "./src" in bundle["sync"]["paths"]
    assert "./notebooks/training" in bundle["sync"]["paths"]
    monitor = bundle["resources"]["jobs"]["model_monitoring"]
    assert monitor["max_concurrent_runs"] == 1
    assert monitor["tasks"][0]["spark_python_task"]["python_file"].endswith("monitor_endpoint.py")
    assert bundle["targets"]["dev"]["variables"]["promotion_profile"] == "dev"
    assert bundle["targets"]["prod"]["variables"]["promotion_profile"] == "prod"


def test_connector_uses_bundle_managed_job_id(repository_root: Path) -> None:
    bundle = yaml.safe_load((repository_root / "databricks.yml").read_text(encoding="utf-8"))
    connector = bundle["resources"]["jobs"]["connect_deployment_job"]
    parameters = connector["tasks"][0]["notebook_task"]["base_parameters"]
    assert parameters["deployment_job_id"] == "${resources.jobs.model_deployment.id}"
