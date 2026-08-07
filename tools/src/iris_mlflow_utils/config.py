"""Configuration shared by the local and Databricks training notebooks."""

from __future__ import annotations

import builtins
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def is_databricks() -> bool:
    """Return whether the current Python process looks like Databricks."""

    return _get_dbutils() is not None or bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def _get_dbutils() -> Any:
    """Find the Databricks utility object exposed by the notebook shell."""

    get_ipython_fn = getattr(builtins, "get_ipython", None)
    if get_ipython_fn is None:
        return None
    shell = get_ipython_fn()
    if shell is None:
        return None
    return shell.user_ns.get("dbutils")


def get_setting(name: str, default: str, label: str) -> str:
    """Read a setting from the environment, Databricks widgets, or a default."""

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
    """Immutable settings shared by both model training flows."""

    dataset_path: Path
    experiment_name: str
    artifact_location: str
    tracking_uri: str
    registry_uri: str
    registered_model_name: str
    run_name: str
    dataset_version: str
    project_version: str
    author: str
    purpose: str
    test_size: float = 0.20
    random_state: int = 42
    primary_metric: str = "test_f1_weighted"
    enable_tracing: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        if self.primary_metric not in {"test_accuracy", "test_f1_weighted"}:
            raise ValueError("primary_metric no está soportada.")


def build_config(
    *,
    model_slug: str,
    registered_model_name: str,
    run_name: str,
    model_defaults: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Build a model-specific configuration from shared environment settings."""

    del model_defaults  # Reserved for future model-specific widget defaults.
    databricks = is_databricks()
    default_data_path = (
        "/Volumes/workspace/my_data/my_volumen/Iris.csv" if databricks else "Iris.csv"
    )
    config = TrainingConfig(
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
            f"IRIS_{model_slug.upper()}_REGISTERED_MODEL",
            registered_model_name,
            "Modelo registrado",
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
    )
    return config
