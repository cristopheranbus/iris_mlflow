"""Small, testable client for Databricks Model Serving endpoints."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests

TIMEOUT_SECONDS = 60
REQUIRED_ENVIRONMENT = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_ENDPOINT_NAME")
HTTP_MESSAGES = {
    400: "Solicitud incompatible con la firma del modelo.",
    401: "Token ausente, inválido o expirado.",
    403: "El token no tiene permisos para invocar el endpoint.",
    404: "Workspace o endpoint no encontrado.",
    429: "Databricks limitó temporalmente las solicitudes.",
    500: "El endpoint devolvió un error interno.",
    503: "El endpoint no está disponible o está iniciando.",
}


class DatabricksEndpointError(RuntimeError):
    """Error controlado de configuración, red, HTTP o contrato de respuesta."""


def read_configuration() -> tuple[str, str, str]:
    """Read endpoint settings without exposing credentials in error messages."""

    values = {name: os.getenv(name, "").strip() for name in REQUIRED_ENVIRONMENT}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise DatabricksEndpointError(f"Faltan variables requeridas: {', '.join(missing)}.")
    return values["DATABRICKS_HOST"], values["DATABRICKS_TOKEN"], values["DATABRICKS_ENDPOINT_NAME"]


def build_invocation_url(host: str, endpoint_name: str) -> str:
    """Build a canonical Databricks serving invocation URL."""

    clean_host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    clean_endpoint = endpoint_name.strip("/")
    if not clean_host or not clean_endpoint:
        raise DatabricksEndpointError("El host y el nombre del endpoint son obligatorios.")
    return f"https://{clean_host}/serving-endpoints/{clean_endpoint}/invocations"


def extract_predictions(response: requests.Response) -> list[Any]:
    """Validate and extract the standard Databricks predictions response."""

    try:
        content = response.json()
    except ValueError as error:
        raise DatabricksEndpointError("La respuesta no es JSON válido.") from error
    predictions = content.get("predictions") if isinstance(content, dict) else None
    if not isinstance(predictions, list):
        raise DatabricksEndpointError("La respuesta no contiene una lista 'predictions'.")
    return predictions


def predict(columns: list[str], rows: list[list[Any]]) -> list[Any]:
    """Invoke an endpoint using the dataframe_split protocol."""

    if not columns or any(len(row) != len(columns) for row in rows):
        raise DatabricksEndpointError("Las columnas y filas no tienen un formato coherente.")
    host, token, endpoint_name = read_configuration()
    payload = {"dataframe_split": {"columns": columns, "data": rows}}
    try:
        response = requests.post(
            build_invocation_url(host, endpoint_name),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise DatabricksEndpointError("No fue posible conectar con Databricks.") from error
    if not 200 <= response.status_code < 300:
        cause = HTTP_MESSAGES.get(response.status_code, "Revisa la respuesta del endpoint.")
        raise DatabricksEndpointError(f"Error HTTP {response.status_code}: {cause}")
    return extract_predictions(response)


def predict_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a dataframe with an endpoint prediction column."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe debe ser un pandas.DataFrame.")
    columns = [str(column) for column in dataframe.columns]
    rows = json.loads(dataframe.to_json(orient="split", date_format="iso"))["data"]
    predictions = predict(columns, rows)
    if len(predictions) != len(dataframe):
        raise DatabricksEndpointError("La cantidad de predicciones no coincide con las filas.")
    result = dataframe.copy()
    result["prediction"] = predictions
    return result
