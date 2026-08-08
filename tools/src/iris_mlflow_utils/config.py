"""Training configuration shared by the local and Databricks notebooks."""

from __future__ import annotations

import builtins
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def is_databricks() -> bool:
    """Return whether the current Python process is running in Databricks."""

    return _get_dbutils() is not None or bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def _get_dbutils() -> Any:
    """Return the Databricks utility object exposed by the notebook shell."""

    get_ipython_fn = getattr(builtins, "get_ipython", None)
    if get_ipython_fn is None:
        return None
    shell = get_ipython_fn()
    if shell is None:
        return None
    return shell.user_ns.get("dbutils")


def get_setting(name: str, default: str, label: str) -> str:
    """Read a setting from env vars, Databricks widgets, or a default value."""

    value = os.getenv(name, "").strip()
    if value:
        return value

    if is_databricks():
        utilities = _get_dbutils()
        if utilities is None:
            return default
        try:
            utilities.widgets.text(name, default, label)
        except Exception:
            pass
        try:
            return utilities.widgets.get(name).strip() or default
        except Exception:
            pass
    return default


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable settings for data preparation, MLflow, and model registration."""

    dataset_path: Path
    experiment_name: str
    artifact_location: str
    tracking_uri: str
    registry_uri: str
    registered_model_name: str
    feature_table: str
    champion_alias: str
    challenger_alias: str
    run_name: str
    dataset_version: str
    project_version: str
    author: str
    purpose: str
    test_size: float = 0.20
    random_state: int = 42
    primary_metric: str = "test_f1_weighted"
    enable_tracing: bool = True
    model_input_example_rows: int = 5
    target_column: str = "Species"
    model_registration_timeout_seconds: int = 300
    model_registration_poll_seconds: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        if self.primary_metric not in {"test_accuracy", "test_f1_weighted"}:
            raise ValueError("primary_metric no está soportada.")
        if not re.fullmatch(
            r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+",
            self.registered_model_name,
        ):
            raise ValueError("registered_model_name debe usar catalog.schema.model.")
        if not re.fullmatch(
            r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+",
            self.feature_table,
        ):
            raise ValueError("feature_table debe usar catalog.schema.table.")
        if not self.champion_alias or not self.challenger_alias:
            raise ValueError("Los aliases Champion y Challenger no pueden estar vacíos.")
        if self.model_input_example_rows < 1:
            raise ValueError("model_input_example_rows debe ser positivo.")
        if self.model_registration_timeout_seconds < 1:
            raise ValueError("model_registration_timeout_seconds debe ser positivo.")
        if self.model_registration_poll_seconds < 1:
            raise ValueError("model_registration_poll_seconds debe ser positivo.")


def build_config(
    *,
    model_slug: str,
    registered_model_name: str,
    run_name: str,
    model_defaults: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Build the common configuration for one algorithm notebook."""

    del model_slug, model_defaults
    databricks = is_databricks()
    default_data_path = (
        "/Volumes/workspace/my_data/my_volumen/Iris.csv" if databricks else "Iris.csv"
    )
    return TrainingConfig(
        dataset_path=Path(get_setting("IRIS_DATA_PATH", default_data_path, "Ruta del dataset")),
        experiment_name=get_setting(
            "IRIS_EXPERIMENT_NAME",
            "/Shared/iris_mlflow" if databricks else "iris_mlflow",
            "Experimento MLflow",
        ),
        artifact_location=get_setting(
            "IRIS_ARTIFACT_LOCATION", "", "Ubicación de artefactos (opcional)"
        ),
        tracking_uri=get_setting("MLFLOW_TRACKING_URI", "", "Tracking URI"),
        registry_uri=get_setting("MLFLOW_REGISTRY_URI", "databricks-uc", "Registry URI"),
        registered_model_name=get_setting(
            "IRIS_REGISTERED_MODEL_NAME",
            registered_model_name,
            "Modelo UC catalog.schema.model",
        ),
        feature_table=get_setting(
            "IRIS_FEATURE_TABLE",
            "workspace.default.iris_features",
            "Tabla UC catalog.schema.table",
        ),
        champion_alias=get_setting(
            "IRIS_CHAMPION_ALIAS", "Champion", "Alias de la versión productiva"
        ),
        challenger_alias=get_setting(
            "IRIS_CHALLENGER_ALIAS", "Challenger", "Alias de la versión candidata"
        ),
        run_name=get_setting("IRIS_RUN_NAME", run_name, "Nombre del run"),
        dataset_version=get_setting("IRIS_DATASET_VERSION", "iris-csv", "Versión del dataset"),
        project_version=get_setting("IRIS_PROJECT_VERSION", "2.0.0", "Versión del proyecto"),
        author=get_setting("IRIS_AUTHOR", "unknown", "Responsable"),
        purpose=get_setting("IRIS_PURPOSE", "baseline-classification", "Propósito"),
        test_size=float(get_setting("IRIS_TEST_SIZE", "0.20", "Proporción de prueba")),
        random_state=int(get_setting("IRIS_RANDOM_STATE", "42", "Semilla")),
        primary_metric=get_setting("IRIS_PRIMARY_METRIC", "test_f1_weighted", "Métrica principal"),
        enable_tracing=get_setting("MLFLOW_ENABLE_TRACING", "true", "Activar tracing").lower()
        in {"1", "true", "yes", "si", "sí"},
        model_input_example_rows=int(
            get_setting("IRIS_MODEL_INPUT_EXAMPLE_ROWS", "5", "Filas del input example")
        ),
        target_column=get_setting("IRIS_TARGET_COLUMN", "Species", "Columna objetivo"),
        model_registration_timeout_seconds=int(
            get_setting(
                "MLFLOW_MODEL_REGISTRATION_TIMEOUT_SECONDS",
                "300",
                "Tiempo máximo de registro",
            )
        ),
        model_registration_poll_seconds=int(
            get_setting(
                "MLFLOW_MODEL_REGISTRATION_POLL_SECONDS",
                "5",
                "Intervalo de consulta del registro",
            )
        ),
    )
