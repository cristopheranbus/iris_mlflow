"""Pruebas unitarias del cliente REST de Databricks."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from databricks_endpoint_client.client import (
    TIMEOUT_SEGUNDOS,
    DatabricksEndpointError,
    consultar_endpoint,
    predecir_dataframe,
)

VARIABLES_DATABRICKS = {
    "DATABRICKS_HOST": "mi-workspace.cloud.databricks.com",
    "DATABRICKS_TOKEN": "token-de-prueba",
    "DATABRICKS_ENDPOINT_NAME": "iris-endpoint",
}
COLUMNAS = ["sepal length (cm)", "sepal width (cm)"]
DATOS = [[5.1, 3.5], [6.4, 3.2]]


def configurar_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configura variables válidas para las pruebas."""
    for nombre, valor in VARIABLES_DATABRICKS.items():
        monkeypatch.setenv(nombre, valor)


def respuesta_mock(status_code: int, payload: object, text: str = "respuesta") -> Mock:
    """Construye una respuesta mínima compatible con requests.Response."""
    respuesta = Mock()
    respuesta.status_code = status_code
    respuesta.text = text
    respuesta.json.return_value = payload
    return respuesta


def test_respuesta_exitosa_envia_dataframe_split_y_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Devuelve predicciones y envía URL, headers, payload y timeout correctos."""
    configurar_entorno(monkeypatch)
    respuesta = respuesta_mock(200, {"predictions": ["setosa", "versicolor"]})

    with patch("databricks_endpoint_client.client.requests.post", return_value=respuesta) as post:
        predicciones = consultar_endpoint(COLUMNAS, DATOS)

    assert predicciones == ["setosa", "versicolor"]
    post.assert_called_once_with(
        "https://mi-workspace.cloud.databricks.com/serving-endpoints/iris-endpoint/invocations",
        headers={
            "Authorization": "Bearer token-de-prueba",
            "Content-Type": "application/json",
        },
        json={"dataframe_split": {"columns": COLUMNAS, "data": DATOS}},
        timeout=TIMEOUT_SEGUNDOS,
    )


@pytest.mark.parametrize(
    ("status_code", "causa"),
    [
        (401, "token"),
        (403, "permisos"),
        (404, "endpoint"),
        (429, "limitó"),
        (503, "disponible"),
    ],
)
def test_errores_http_con_mensaje_explicativo(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    causa: str,
) -> None:
    """Incluye código, texto de Databricks y causa probable."""
    configurar_entorno(monkeypatch)
    respuesta = respuesta_mock(status_code, {"error": "detalle"}, "detalle devuelto")

    with patch("databricks_endpoint_client.client.requests.post", return_value=respuesta):
        with pytest.raises(DatabricksEndpointError) as error:
            consultar_endpoint(COLUMNAS, DATOS)

    mensaje = str(error.value)
    assert f"HTTP {status_code}" in mensaje
    assert "detalle devuelto" in mensaje
    assert causa in mensaje


def test_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convierte un timeout de requests en un error del cliente."""
    configurar_entorno(monkeypatch)

    with patch(
        "databricks_endpoint_client.client.requests.post",
        side_effect=requests.Timeout,
    ):
        with pytest.raises(DatabricksEndpointError, match="Timeout"):
            consultar_endpoint(COLUMNAS, DATOS)


def test_respuesta_sin_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rechaza respuestas JSON que no tienen la clave requerida."""
    configurar_entorno(monkeypatch)
    respuesta = respuesta_mock(200, {"outputs": [1, 2]}, "sin predictions")

    with patch("databricks_endpoint_client.client.requests.post", return_value=respuesta):
        with pytest.raises(DatabricksEndpointError, match="predictions"):
            consultar_endpoint(COLUMNAS, DATOS)


def test_json_invalido(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rechaza una respuesta que no puede interpretarse como JSON."""
    configurar_entorno(monkeypatch)
    respuesta = respuesta_mock(200, None, "<!doctype html>")
    respuesta.json.side_effect = ValueError("JSON inválido")

    with patch("databricks_endpoint_client.client.requests.post", return_value=respuesta):
        with pytest.raises(DatabricksEndpointError, match="JSON inválida"):
            consultar_endpoint(COLUMNAS, DATOS)


def test_cantidad_de_predicciones_diferente_al_numero_de_filas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rechaza un resultado con una predicción por cada fila."""
    configurar_entorno(monkeypatch)
    dataframe = pd.DataFrame(DATOS, columns=COLUMNAS)
    respuesta = respuesta_mock(200, {"predictions": ["setosa"]})

    with patch("databricks_endpoint_client.client.requests.post", return_value=respuesta):
        with pytest.raises(DatabricksEndpointError, match="no coincide"):
            predecir_dataframe(dataframe)


def test_predecir_dataframe_devuelve_copia_con_prediccion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conserva el DataFrame original y agrega predicciones en una copia."""
    configurar_entorno(monkeypatch)
    dataframe = pd.DataFrame(DATOS, columns=COLUMNAS)
    respuesta = respuesta_mock(200, {"predictions": ["setosa", "versicolor"]})

    with patch("databricks_endpoint_client.client.requests.post", return_value=respuesta):
        resultado = predecir_dataframe(dataframe)

    assert "prediccion" not in dataframe.columns
    assert resultado["prediccion"].tolist() == ["setosa", "versicolor"]
    assert resultado is not dataframe


def test_variables_de_entorno_faltantes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Indica todas las variables de entorno que faltan."""
    for nombre in VARIABLES_DATABRICKS:
        monkeypatch.delenv(nombre, raising=False)

    with pytest.raises(DatabricksEndpointError) as error:
        consultar_endpoint(COLUMNAS, DATOS)

    mensaje = str(error.value)
    for nombre in VARIABLES_DATABRICKS:
        assert nombre in mensaje
