"""Cliente para consumir endpoints de Databricks Model Serving."""

from .client import (
    TIMEOUT_SEGUNDOS,
    DatabricksEndpointError,
    consultar_endpoint,
    predecir_dataframe,
)

__all__ = [
    "TIMEOUT_SEGUNDOS",
    "DatabricksEndpointError",
    "consultar_endpoint",
    "predecir_dataframe",
]
