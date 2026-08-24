"""Isolation for tests that use the process-global MLflow client."""

from collections.abc import Generator

import mlflow
import pytest


@pytest.fixture(autouse=True)
def restore_mlflow_state() -> Generator[None]:
    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        yield
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        mlflow.set_tracking_uri(previous_tracking_uri)
