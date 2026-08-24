"""Shared isolation fixtures for all test suites."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

_ISOLATED_PREFIXES = ("IRIS_", "MLFLOW_", "DATABRICKS_")


def pytest_sessionstart() -> None:
    """Create the ignored root for all local test state before tmp_path starts."""

    local_root = Path(__file__).parents[1] / ".local"
    local_root.mkdir(exist_ok=True)
    (local_root / "coverage").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def isolate_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent workstation or CI variables from changing test behavior."""

    for name in tuple(os.environ):
        if name.startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MPLBACKEND", "Agg")
