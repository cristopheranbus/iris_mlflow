"""Pruebas unitarias del cliente REST sin dependencias externas.

Las pruebas sustituyen ``requests.post`` y aíslan las variables de entorno con
``monkeypatch``. Por eso verifican el contrato de nuestro código sin enviar
datos a Databricks ni requerir un token real. Los nombres de las pruebas
describen el comportamiento observable que debe conservarse al refactorizar.
"""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from databricks_endpoint_client import (
    DatabricksEndpointError,
    consultar_endpoint,
    predecir_dataframe,
)


def configurar_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instala una configuración ficticia y segura para cada caso."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com/")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token-seguro")
    monkeypatch.setenv("DATABRICKS_ENDPOINT_NAME", "/iris-endpoint/")


def respuesta_json(predicciones: list[object]) -> Mock:
    """Construye una respuesta HTTP exitosa con el contrato esperado."""
    respuesta = Mock(spec=requests.Response)
    respuesta.status_code = 200
    respuesta.text = '{"predictions": [...]} '
    respuesta.json.return_value = {"predictions": predicciones}
    return respuesta


def test_consultar_endpoint_envia_dataframe_split_y_devuelve_predicciones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La API construye URL, headers y payload sin alterar las predicciones."""
    configurar_entorno(monkeypatch)
    respuesta = respuesta_json(["setosa", "versicolor"])
    post = Mock(return_value=respuesta)
    monkeypatch.setattr(requests, "post", post)

    predicciones = consultar_endpoint(
        ["sepal_length", "sepal_width"],
        [[5.1, 3.5], [6.4, 3.2]],
    )

    assert predicciones == ["setosa", "versicolor"]
    post.assert_called_once_with(
        "https://workspace.example.com/serving-endpoints/iris-endpoint/invocations",
        headers={
            "Authorization": "Bearer token-seguro",
            "Content-Type": "application/json",
        },
        json={
            "dataframe_split": {
                "columns": ["sepal_length", "sepal_width"],
                "data": [[5.1, 3.5], [6.4, 3.2]],
            }
        },
        timeout=60,
    )


def test_consultar_endpoint_rechaza_variables_faltantes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La ausencia de configuración falla antes de intentar una petición."""
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_ENDPOINT_NAME", raising=False)

    with pytest.raises(DatabricksEndpointError, match="DATABRICKS_HOST"):
        consultar_endpoint(["feature"], [[1.0]])


def test_consultar_endpoint_traduce_error_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un status no exitoso se convierte en la excepción pública del paquete."""
    configurar_entorno(monkeypatch)
    respuesta = Mock(spec=requests.Response)
    respuesta.status_code = 401
    respuesta.text = "unauthorized"
    monkeypatch.setattr(requests, "post", Mock(return_value=respuesta))

    with pytest.raises(DatabricksEndpointError, match="Error HTTP 401"):
        consultar_endpoint(["feature"], [[1.0]])


def test_consultar_endpoint_rechaza_json_sin_predicciones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Una respuesta 2xx también debe cumplir el contrato JSON documentado."""
    configurar_entorno(monkeypatch)
    respuesta = Mock(spec=requests.Response)
    respuesta.status_code = 200
    respuesta.text = "{}"
    respuesta.json.return_value = {}
    monkeypatch.setattr(requests, "post", Mock(return_value=respuesta))

    with pytest.raises(DatabricksEndpointError, match="predictions"):
        consultar_endpoint(["feature"], [[1.0]])


def test_predecir_dataframe_no_modifica_original_y_agrega_prediccion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La adaptación a DataFrame conserva la entrada y añade una nueva columna."""
    configurar_entorno(monkeypatch)
    monkeypatch.setattr(
        requests,
        "post",
        Mock(return_value=respuesta_json(["setosa", "versicolor"])),
    )
    original = pd.DataFrame({"feature": [1.0, 2.0]})

    resultado = predecir_dataframe(original)

    pd.testing.assert_frame_equal(original, pd.DataFrame({"feature": [1.0, 2.0]}))
    pd.testing.assert_frame_equal(
        resultado,
        pd.DataFrame({"feature": [1.0, 2.0], "prediccion": ["setosa", "versicolor"]}),
    )


def test_predecir_dataframe_rechaza_cantidad_incorrecta_de_predicciones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Se rechaza una respuesta que no mantiene una predicción por fila."""
    configurar_entorno(monkeypatch)
    monkeypatch.setattr(requests, "post", Mock(return_value=respuesta_json(["setosa"])))

    with pytest.raises(DatabricksEndpointError, match="no coincide"):
        predecir_dataframe(pd.DataFrame({"feature": [1.0, 2.0]}))
