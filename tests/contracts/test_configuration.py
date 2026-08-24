"""Contracts for checked-in local configuration examples."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def test_local_environment_example_is_self_consistent(repository_root: Path) -> None:
    values: dict[str, str] = {}
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
