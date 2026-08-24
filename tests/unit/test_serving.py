"""Unit tests for the reusable Databricks serving client."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from iris_mlflow_utils.serving import (
    DatabricksEndpointError,
    build_invocation_url,
    extract_predictions,
    predict,
    predict_dataframe,
    read_configuration,
)

pytestmark = pytest.mark.unit


def test_build_invocation_url_removes_protocol_and_slashes() -> None:
    assert (
        build_invocation_url("https://workspace.example/", "/iris-endpoint/")
        == "https://workspace.example/serving-endpoints/iris-endpoint/invocations"
    )


@pytest.mark.parametrize("host, endpoint", [("", "iris"), ("https://workspace", "")])
def test_build_invocation_url_rejects_empty_parts(host: str, endpoint: str) -> None:
    with pytest.raises(DatabricksEndpointError, match="obligatorios"):
        build_invocation_url(host, endpoint)


def test_read_configuration_reports_missing_names_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "do-not-leak")

    with pytest.raises(DatabricksEndpointError) as captured:
        read_configuration()

    assert "DATABRICKS_HOST" in str(captured.value)
    assert "DATABRICKS_ENDPOINT_NAME" in str(captured.value)
    assert "do-not-leak" not in str(captured.value)


def test_extract_predictions_rejects_invalid_contract() -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"result": []}'

    with pytest.raises(DatabricksEndpointError, match="predictions"):
        extract_predictions(response)


def test_extract_predictions_rejects_non_json() -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b"not-json"

    with pytest.raises(DatabricksEndpointError, match="JSON válido"):
        extract_predictions(response)


def test_predict_uses_dataframe_split_and_returns_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, list[int]]:
            return {"predictions": [0, 1]}

    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret")
    monkeypatch.setenv("DATABRICKS_ENDPOINT_NAME", "iris")
    monkeypatch.setattr("iris_mlflow_utils.serving.requests.post", fake_post)

    assert predict(["feature"], [[1.0], [2.0]]) == [0, 1]
    assert captured["url"] == "https://workspace.example/serving-endpoints/iris/invocations"
    assert captured["json"] == {"dataframe_split": {"columns": ["feature"], "data": [[1.0], [2.0]]}}


@pytest.mark.parametrize("columns, rows", [([], []), (["a", "b"], [[1]]), (["a"], [[1, 2]])])
def test_predict_rejects_incoherent_tabular_input(
    monkeypatch: pytest.MonkeyPatch,
    columns: list[str],
    rows: list[list[object]],
) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret")
    monkeypatch.setenv("DATABRICKS_ENDPOINT_NAME", "iris")

    with pytest.raises(DatabricksEndpointError, match="formato coherente"):
        predict(columns, rows)


def test_predict_translates_network_errors_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_post(url: str, **kwargs: object) -> None:
        raise requests.Timeout("secret-token")

    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret-token")
    monkeypatch.setenv("DATABRICKS_ENDPOINT_NAME", "iris")
    monkeypatch.setattr("iris_mlflow_utils.serving.requests.post", fail_post)

    with pytest.raises(DatabricksEndpointError) as captured:
        predict(["feature"], [[1.0]])

    assert "conectar" in str(captured.value)
    assert "secret-token" not in str(captured.value)


@pytest.mark.parametrize(
    "status_code, expected",
    [
        (400, "firma"),
        (401, "Token"),
        (403, "permisos"),
        (404, "no encontrado"),
        (429, "limitó"),
        (500, "interno"),
        (503, "disponible"),
        (418, "Revisa"),
    ],
)
def test_predict_translates_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected: str,
) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = status_code

    monkeypatch.setenv("DATABRICKS_HOST", "workspace.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "secret")
    monkeypatch.setenv("DATABRICKS_ENDPOINT_NAME", "iris")
    monkeypatch.setattr(
        "iris_mlflow_utils.serving.requests.post", lambda url, **kwargs: FakeResponse()
    )

    with pytest.raises(DatabricksEndpointError, match=expected):
        predict(["feature"], [[1.0]])


def test_predict_dataframe_validates_prediction_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("iris_mlflow_utils.serving.predict", lambda columns, rows: [1])
    dataframe = pd.DataFrame({"feature": [1.0, 2.0]})

    with pytest.raises(DatabricksEndpointError, match="cantidad"):
        from iris_mlflow_utils.serving import predict_dataframe

        predict_dataframe(dataframe)


def test_predict_dataframe_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        predict_dataframe({"feature": [1.0]})  # type: ignore[arg-type]


def test_predict_dataframe_returns_copy_with_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({"feature": [1.0, 2.0]})
    monkeypatch.setattr("iris_mlflow_utils.serving.predict", lambda columns, rows: [0, 1])

    result = predict_dataframe(dataframe)

    assert list(result["prediction"]) == [0, 1]
    assert "prediction" not in dataframe.columns
