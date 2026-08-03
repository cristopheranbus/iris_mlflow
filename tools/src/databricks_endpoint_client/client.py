"""Cliente REST pequeño y explícito para Databricks Model Serving.

El módulo encapsula cuatro responsabilidades que conviene mantener separadas:

1. Leer la configuración desde variables de entorno sin exponer el token.
2. Construir la URL y el payload que espera Databricks.
3. Traducir errores de transporte, HTTP y formato a una excepción del dominio.
4. Ofrecer una API cómoda para trabajar con listas o con ``pandas.DataFrame``.

El cliente usa el formato ``dataframe_split`` porque conserva tanto los nombres
como el orden de las columnas. Esto es importante: un endpoint servido con una
firma de modelo espera las mismas columnas y tipos que recibió durante el
entrenamiento. El cliente no intenta corregir nombres, reordenar features ni
convertir la respuesta en una etiqueta concreta; esas decisiones pertenecen al
contrato del modelo y deben validarse antes de invocar el endpoint.

Seguridad:
    ``DATABRICKS_TOKEN`` solo se usa para construir el header Authorization. No
    se imprime, no se incluye en las excepciones y no se persiste en disco.

Reintentos:
    El módulo no reintenta automáticamente. Un reintento puede duplicar carga
    o retrasar una respuesta cuando el endpoint escala desde cero. La política
    de reintentos debe vivir en la aplicación que conoce el contexto de negocio.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests

TIMEOUT_SEGUNDOS = 60
"""Tiempo máximo, en segundos, para esperar una respuesta del endpoint."""

_VARIABLES_REQUERIDAS = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_ENDPOINT_NAME",
)

# Estas explicaciones convierten códigos HTTP frecuentes en mensajes accionables.
# Para un código no contemplado se conserva una explicación genérica.
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
    """Error controlado al consultar o interpretar un endpoint de Databricks.

    La excepción unifica fallos de configuración, red, HTTP y contrato de
    respuesta. Así, una aplicación consumidora puede capturar un único tipo sin
    depender de los detalles internos de ``requests``.
    """


def _leer_configuracion() -> tuple[str, str, str]:
    """Lee y valida las variables necesarias para invocar el endpoint.

    Se eliminan espacios accidentales al leer cada variable. La validación se
    realiza antes de construir el payload o iniciar una conexión, de modo que
    un error de configuración falla rápido y no se confunde con un error remoto.

    Returns:
        Una tupla ``(host, token, nombre_endpoint)`` ya normalizada en cuanto a
        espacios exteriores.

    Raises:
        DatabricksEndpointError: Si falta una o más variables requeridas.
    """
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
    """Construye la URL HTTPS estándar de invocación.

    ``DATABRICKS_HOST`` puede recibirse con o sin esquema y con una barra final;
    el endpoint puede recibirse con barras laterales. Se limpian únicamente esas
    variantes de formato para evitar URLs duplicadas. El esquema final siempre
    es HTTPS, porque el token viaja en el header Authorization.

    Args:
        host: Dominio del workspace, opcionalmente precedido por ``http://`` o
            ``https://``.
        nombre_endpoint: Nombre lógico del endpoint de Model Serving.

    Returns:
        URL con la ruta ``/serving-endpoints/<nombre>/invocations``.
    """
    host_limpio = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    endpoint_limpio = nombre_endpoint.strip("/")
    return f"https://{host_limpio}/serving-endpoints/{endpoint_limpio}/invocations"


def _extraer_predicciones(respuesta: requests.Response) -> list[Any]:
    """Valida el JSON exitoso y devuelve la lista ``predictions``.

    Un status HTTP 2xx no garantiza que el cuerpo tenga el contrato que espera
    este cliente. Por eso se valida explícitamente que el cuerpo sea un objeto,
    que contenga ``predictions`` y que ese campo sea una lista.

    Args:
        respuesta: Respuesta de ``requests`` cuyo status ya fue considerado
            exitoso por ``consultar_endpoint``.

    Returns:
        Lista de predicciones sin transformar sus valores.

    Raises:
        DatabricksEndpointError: Si el cuerpo no es JSON válido o no cumple el
            contrato de respuesta.
    """
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
    """Genera un mensaje HTTP consistente y útil para diagnóstico.

    El texto de Databricks se conserva porque suele incluir la causa concreta
    (firma incompatible, permisos, endpoint iniciándose, etc.). El token no forma
    parte de ``respuesta.text`` generado por este cliente y nunca se concatena a
    este mensaje.
    """
    causa = _CAUSAS_HTTP.get(
        respuesta.status_code,
        " ".join(
            (
                "Revisa la respuesta y la configuración del endpoint; el código",
                "no está mapeado explícitamente.",
            )
        ),
    )
    return (
        f"Error HTTP {respuesta.status_code} al consultar Databricks. "
        f"Texto devuelto: {respuesta.text!r}. Causa probable: {causa}"
    )


def consultar_endpoint(columnas: list[str], datos: list[list[Any]]) -> list[Any]:
    """Invoca un endpoint usando el formato ``dataframe_split``.

    Args:
        columnas: Nombres de las features en el orden exacto esperado por el
            modelo. El endpoint distingue nombres y orden.
        datos: Filas a enviar. Cada fila debe tener el mismo número de valores
            que ``columnas``. Puede ser una lista vacía si el endpoint acepta un
            lote vacío, aunque normalmente conviene evitar esa llamada.

    Returns:
        Lista de predicciones tal como la entrega Databricks.

    Raises:
        DatabricksEndpointError: Si los argumentos no son coherentes, falta
            configuración, falla la red, ocurre un error HTTP o la respuesta no
            cumple el contrato ``{"predictions": [...]}``.
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

    # ``dataframe_split`` es el contrato de entrada del endpoint: las columnas
    # viajan una sola vez y ``data`` conserva las filas en el mismo orden.
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

    # Se acepta cualquier 2xx. La comprobación adicional contra _CAUSAS_HTTP
    # mantiene el mensaje específico si Databricks devuelve uno de esos códigos.
    if respuesta.status_code in _CAUSAS_HTTP or not 200 <= respuesta.status_code < 300:
        raise DatabricksEndpointError(_mensaje_error_http(respuesta))

    return _extraer_predicciones(respuesta)


def predecir_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Añade predicciones a una copia de un DataFrame.

    El DataFrame original no se modifica. La conversión a JSON ``split`` hace
    que pandas serialice correctamente valores compatibles con JSON, incluyendo
    fechas cuando existen. Después de consultar el endpoint se comprueba que
    haya exactamente una predicción por fila antes de construir el resultado.

    Args:
        dataframe: DataFrame cuyas columnas y filas coinciden con la firma del
            modelo servido.

    Returns:
        Copia del DataFrame original con la columna ``prediccion`` al final.

    Raises:
        TypeError: Si ``dataframe`` no es una instancia de ``pandas.DataFrame``.
        DatabricksEndpointError: Si falla la invocación o el endpoint devuelve
            una cantidad de predicciones distinta a la cantidad de filas.
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
    # Este bloque es una prueba manual mínima. Requiere las tres variables de
    # entorno y un endpoint compatible; no se ejecuta al importar el paquete.
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
