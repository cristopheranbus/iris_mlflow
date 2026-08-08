"""Tests for the reusable Databricks serving client."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from iris_mlflow_utils.serving import (
    DatabricksEndpointError,
    build_invocation_url,
    extract_predictions,
    predict,
)


def test_build_invocation_url_removes_protocol_and_slashes() -> None:
    assert (
        build_invocation_url("https://workspace.example/", "/iris-endpoint/")
        == "https://workspace.example/serving-endpoints/iris-endpoint/invocations"
    )


def test_extract_predictions_rejects_invalid_contract() -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"result": []}'

    with pytest.raises(DatabricksEndpointError, match="predictions"):
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


def test_predict_dataframe_validates_prediction_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("iris_mlflow_utils.serving.predict", lambda columns, rows: [1])
    dataframe = pd.DataFrame({"feature": [1.0, 2.0]})

    with pytest.raises(DatabricksEndpointError, match="cantidad"):
        from iris_mlflow_utils.serving import predict_dataframe

        predict_dataframe(dataframe)
