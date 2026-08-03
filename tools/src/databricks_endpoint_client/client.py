"""Cliente REST para consumir un endpoint de Databricks Model Serving."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests

TIMEOUT_SEGUNDOS = 60
"""Tiempo máximo, en segundos, para esperar la respuesta de Databricks."""

_VARIABLES_REQUERIDAS = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_ENDPOINT_NAME",
)

_CAUSAS_HTTP: dict[int, str] = {
    400: " ".join(
        (
            "La solicitud no tiene el formato esperado o las columnas/datos",
            "no coinciden con el modelo.",
        )
    ),
    401: "El token falta, es inválido, expiró o no fue aceptado por Databricks.",
    403: "El token es válido, pero no tiene permisos para usar este endpoint.",
    404: "El workspace o el endpoint no existe, o su nombre/ruta es incorrecto.",
    429: "Databricks limitó temporalmente las solicitudes por exceso de tráfico o cuota.",
    500: "El servidor encontró un error interno al procesar la predicción.",
    503: "El endpoint no está disponible, está iniciando o no tiene capacidad disponible.",
}


class DatabricksEndpointError(RuntimeError):
    """Error controlado al consultar o interpretar un endpoint de Databricks."""


def _leer_configuracion() -> tuple[str, str, str]:
    """Lee y valida las variables de entorno necesarias para la consulta."""
    valores = {nombre: os.getenv(nombre, "").strip() for nombre in _VARIABLES_REQUERIDAS}
    faltantes = [nombre for nombre, valor in valores.items() if not valor]
    if faltantes:
        raise DatabricksEndpointError(
            "Faltan variables de entorno requeridas: "
            f"{', '.join(faltantes)}. Define DATABRICKS_HOST, "
            "DATABRICKS_TOKEN y DATABRICKS_ENDPOINT_NAME antes de ejecutar el cliente."
        )

    return (
        valores["DATABRICKS_HOST"],
        valores["DATABRICKS_TOKEN"],
        valores["DATABRICKS_ENDPOINT_NAME"],
    )


def _construir_url(host: str, nombre_endpoint: str) -> str:
    """Construye la URL HTTPS de invocación del endpoint."""
    host_limpio = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    endpoint_limpio = nombre_endpoint.strip("/")
    return f"https://{host_limpio}/serving-endpoints/{endpoint_limpio}/invocations"


def _extraer_predicciones(respuesta: requests.Response) -> list[Any]:
    """Convierte la respuesta JSON en una lista de predicciones validada."""
    try:
        contenido: Any = respuesta.json()
    except ValueError as error:
        raise DatabricksEndpointError(
            "Databricks devolvió una respuesta JSON inválida. "
            f"Código HTTP: {respuesta.status_code}. Texto devuelto: {respuesta.text!r}. "
            "Causa probable: el endpoint devolvió HTML, texto plano o un error no estructurado."
        ) from error

    if not isinstance(contenido, dict) or "predictions" not in contenido:
        raise DatabricksEndpointError(
            "La respuesta de Databricks no contiene la clave 'predictions'. "
            f"Código HTTP: {respuesta.status_code}. Texto devuelto: {respuesta.text!r}. "
            "Causa probable: el endpoint usa otro formato de respuesta o devolvió "
            "un detalle de error."
        )

    predicciones = contenido["predictions"]
    if not isinstance(predicciones, list):
        raise DatabricksEndpointError(
            "La clave 'predictions' no contiene una lista. "
            f"Código HTTP: {respuesta.status_code}. Texto devuelto: {respuesta.text!r}. "
            "Causa probable: el modelo está devolviendo un formato incompatible con este cliente."
        )

    return predicciones


def _mensaje_error_http(respuesta: requests.Response) -> str:
    """Genera un mensaje consistente para errores HTTP de Databricks."""
    causa = _CAUSAS_HTTP.get(
        respuesta.status_code,
        "Revisa la respuesta y la configuración del endpoint; el código no está "
        "mapeado explícitamente.",
    )
    return (
        f"Error HTTP {respuesta.status_code} al consultar Databricks. "
        f"Texto devuelto: {respuesta.text!r}. Causa probable: {causa}"
    )


def consultar_endpoint(columnas: list[str], datos: list[list[Any]]) -> list[Any]:
    """Consulta un endpoint de Databricks usando el formato ``dataframe_split``.

    Args:
        columnas: Nombres de las columnas en el mismo orden de cada fila de ``datos``.
        datos: Filas que se enviarán al modelo.

    Returns:
        Lista de predicciones devuelta por Databricks.

    Raises:
        DatabricksEndpointError: Si falta configuración, falla la red, ocurre un
            error HTTP o la respuesta no tiene el formato esperado.
    """
    if not columnas:
        raise DatabricksEndpointError("La lista 'columnas' no puede estar vacía.")
    if any(len(fila) != len(columnas) for fila in datos):
        raise DatabricksEndpointError(
            "Cada fila de 'datos' debe tener exactamente la misma cantidad de valores "
            "que 'columnas'."
        )

    host, token, nombre_endpoint = _leer_configuracion()
    url = _construir_url(host, nombre_endpoint)
    payload = {
        "dataframe_split": {
            "columns": columnas,
            "data": datos,
        }
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        respuesta = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SEGUNDOS,
        )
    except requests.Timeout as error:
        raise DatabricksEndpointError(
            f"Timeout al consultar Databricks después de {TIMEOUT_SEGUNDOS} segundos. "
            "Causa probable: el endpoint está ocupado, iniciando o la predicción tarda demasiado."
        ) from error
    except requests.ConnectionError as error:
        raise DatabricksEndpointError(
            "Error de conexión con Databricks. Causa probable: el host no es accesible, "
            "la red bloquea la conexión o el workspace no está disponible."
        ) from error
    except requests.RequestException as error:
        raise DatabricksEndpointError(
            f"Error de red al consultar Databricks: {error}. "
            "Causa probable: fallo de transporte o configuración de requests."
        ) from error

    if respuesta.status_code in _CAUSAS_HTTP or not 200 <= respuesta.status_code < 300:
        raise DatabricksEndpointError(_mensaje_error_http(respuesta))

    return _extraer_predicciones(respuesta)


def predecir_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Agrega las predicciones de Databricks a una copia del DataFrame.

    Args:
        dataframe: DataFrame cuyas columnas y filas se enviarán al endpoint.

    Returns:
        Copia del DataFrame original con la columna ``prediccion`` al final.

    Raises:
        DatabricksEndpointError: Si la respuesta no tiene una predicción por fila.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("'dataframe' debe ser una instancia de pandas.DataFrame.")

    columnas = [str(columna) for columna in dataframe.columns]
    datos = json.loads(dataframe.to_json(orient="split", date_format="iso"))["data"]
    predicciones = consultar_endpoint(columnas, datos)

    if len(predicciones) != len(dataframe):
        raise DatabricksEndpointError(
            "La cantidad de predicciones no coincide con la cantidad de filas. "
            f"Filas recibidas: {len(dataframe)}. Predicciones recibidas: {len(predicciones)}. "
            "Causa probable: el endpoint descartó filas o devolvió un resultado agregado."
        )

    resultado = dataframe.copy()
    resultado["prediccion"] = predicciones
    return resultado


if __name__ == "__main__":
    columnas_iris = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    datos_iris = [
        [5.1, 3.5, 1.4, 0.2],
        [6.4, 3.2, 4.5, 1.5],
        [6.7, 3.1, 5.6, 2.4],
    ]
    print(consultar_endpoint(columnas_iris, datos_iris))
