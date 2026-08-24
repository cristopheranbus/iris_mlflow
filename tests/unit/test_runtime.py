"""Unit tests for runtime detection and parameter access."""

import builtins

import pytest

from iris_mlflow_utils.runtime import (
    detect_runtime,
    get_dbutils,
    get_runtime_parameter,
    is_databricks_runtime,
)

pytestmark = pytest.mark.unit


def _install_dbutils(monkeypatch: pytest.MonkeyPatch, dbutils: object | None) -> None:
    shell = type("Shell", (), {"user_ns": {"dbutils": dbutils}})()
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell, raising=False)


def test_detect_runtime_defaults_to_local_without_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(builtins, "get_ipython", raising=False)
    assert detect_runtime() == "local"
    assert is_databricks_runtime() is False
    assert get_dbutils() is None


def test_detect_runtime_prefers_explicit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_RUNTIME", "local")
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    assert detect_runtime() == "local"


def test_detect_runtime_uses_runtime_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    assert detect_runtime() == "databricks"
    assert is_databricks_runtime() is True


def test_detect_runtime_and_get_dbutils_from_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    dbutils = object()
    _install_dbutils(monkeypatch, dbutils)
    assert detect_runtime() == "databricks"
    assert get_dbutils() is dbutils


def test_get_dbutils_handles_empty_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    assert get_dbutils() is None


def test_runtime_parameter_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_VERSION", " 7 ")
    assert get_runtime_parameter("MODEL_VERSION", "1", "databricks") == "7"


def test_runtime_parameter_returns_default_locally() -> None:
    assert get_runtime_parameter("MODEL_VERSION", "1", "local") == "1"


@pytest.mark.parametrize("widget_value, expected", [(" 9 ", "9"), (" ", "1")])
def test_runtime_parameter_reads_widget(
    monkeypatch: pytest.MonkeyPatch,
    widget_value: str,
    expected: str,
) -> None:
    widgets = type("Widgets", (), {"get": lambda self, name: widget_value})()
    dbutils = type("Dbutils", (), {"widgets": widgets})()
    _install_dbutils(monkeypatch, dbutils)
    assert get_runtime_parameter("MODEL_VERSION", "1", "databricks") == expected


def test_runtime_parameter_handles_missing_or_failing_widgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dbutils(monkeypatch, None)
    assert get_runtime_parameter("MODEL_VERSION", "1", "databricks") == "1"

    widgets = type(
        "Widgets", (), {"get": lambda self, name: (_ for _ in ()).throw(KeyError(name))}
    )()
    _install_dbutils(monkeypatch, type("Dbutils", (), {"widgets": widgets})())
    assert get_runtime_parameter("MODEL_VERSION", "1", "databricks") == "1"
