"""Configuration shared by the local and Databricks training notebooks."""

from __future__ import annotations

import builtins
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runtime import RuntimeMode, detect_runtime, get_runtime_parameter

_MODEL_TYPES = {"random_forest", "xgboost", "neural_network"}
_MODEL_FRAMEWORKS = {"sklearn", "xgboost", "pytorch", "tensorflow"}


def _get_dbutils() -> Any:
    """Return Databricks utilities when available in the notebook namespace."""

    get_ipython_fn = getattr(builtins, "get_ipython", None)
    if get_ipython_fn is None:
        return None
    shell = get_ipython_fn()
    return None if shell is None else shell.user_ns.get("dbutils")


def is_databricks() -> bool:
    """Return whether the current process is running in Databricks."""

    return detect_runtime() == "databricks"


def _find_config_path(path: Path | None = None) -> Path:
    """Resolve the TOML path without requiring a repository-specific install path."""

    candidates = []
    if path is not None:
        candidates.append(path)
    configured = os.getenv("IRIS_CONFIG_PATH", "").strip()
    if configured:
        candidates.append(Path(configured))
    current = Path.cwd().resolve()
    candidates.extend(current / "config" / "training.toml" for _ in [0])
    candidates.extend(parent / "config" / "training.toml" for parent in current.parents)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "No se encontró config/training.toml. Configura IRIS_CONFIG_PATH con la ruta del archivo."
    )


def load_file_config(path: Path | None = None) -> dict[str, Any]:
    """Load the versioned, non-secret TOML configuration."""

    with _find_config_path(path).open("rb") as config_file:
        return tomllib.load(config_file)


def _widget_overrides_enabled() -> bool:
    return os.getenv("IRIS_ENABLE_WIDGET_OVERRIDES", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "si",
        "sí",
    }


def get_setting(name: str, default: str, label: str = "") -> str:
    """Read an environment value and optionally an existing widget override.

    Widgets are never created automatically. They are read only when explicitly
    enabled through IRIS_ENABLE_WIDGET_OVERRIDES.
    """

    value = os.getenv(name, "").strip()
    if value:
        return value
    if not (_widget_overrides_enabled() and is_databricks()):
        return default
    utilities = _get_dbutils()
    if utilities is None:
        return default
    try:
        return utilities.widgets.get(name).strip() or default
    except Exception:
        return default


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable settings for data preparation, MLflow, and model registration."""

    dataset_path: Path | None
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
    feature_table_version: str = ""
    model_type: str = ""
    model_framework: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)
    runtime_mode: RuntimeMode = "databricks"
    auto_approve: bool = False
    deployment_manifest_path: Path = Path("artifacts/local_deployment_manifest.json")

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        if self.primary_metric not in {"test_accuracy", "test_f1_weighted"}:
            raise ValueError("primary_metric no está soportada.")
        if self.runtime_mode == "databricks":
            for value, message in (
                (
                    self.registered_model_name,
                    "registered_model_name debe usar catalog.schema.model.",
                ),
                (self.feature_table, "feature_table debe usar catalog.schema.table."),
            ):
                if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+", value):
                    raise ValueError(message)
        elif not re.fullmatch(r"[A-Za-z0-9_]+", self.registered_model_name):
            raise ValueError(
                "El modelo local debe usar un nombre simple, por ejemplo iris_classifier."
            )
        if not self.champion_alias or not self.challenger_alias:
            raise ValueError("Los aliases Champion y Challenger no pueden estar vacíos.")
        if self.model_input_example_rows < 1:
            raise ValueError("model_input_example_rows debe ser positivo.")
        if self.model_registration_timeout_seconds < 1:
            raise ValueError("model_registration_timeout_seconds debe ser positivo.")
        if self.model_registration_poll_seconds < 1:
            raise ValueError("model_registration_poll_seconds debe ser positivo.")
        if self.model_type and self.model_type not in _MODEL_TYPES:
            raise ValueError(f"model_type no soportado: {self.model_type}.")
        if self.model_framework and self.model_framework not in _MODEL_FRAMEWORKS:
            raise ValueError(f"model_framework no soportado: {self.model_framework}.")


@dataclass(frozen=True)
class DeploymentConfig:
    """Settings for evaluation gates and Model Serving promotion."""

    model_name: str
    endpoint_name: str
    min_test_f1_weighted: float = 0.90
    min_test_accuracy: float = 0.90
    max_metric_regression: float = 0.01
    required_approval_tag: str = "Approval_Check"
    champion_alias: str = "Champion"
    challenger_alias: str = "Challenger"
    smoke_test_rows: int = 1
    serving_timeout_seconds: int = 900
    serving_poll_seconds: int = 15
    notebook_root: str = "/Workspace/Shared/iris_mlflow/deployment"
    job_name: str = "iris-model-deployment"
    runtime_mode: RuntimeMode = "databricks"

    def __post_init__(self) -> None:
        valid_name = (
            re.fullmatch(r"[A-Za-z0-9_]+", self.model_name)
            if self.runtime_mode == "local"
            else re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+", self.model_name)
        )
        if not valid_name:
            raise ValueError(
                "deployment.model_name debe usar catalog.schema.model en Databricks "
                "o un nombre simple en local."
            )
        if not self.endpoint_name.strip():
            raise ValueError("deployment.endpoint_name no puede estar vacío.")
        if not 0 <= self.min_test_f1_weighted <= 1 or not 0 <= self.min_test_accuracy <= 1:
            raise ValueError("Los thresholds de deployment deben estar entre 0 y 1.")
        if self.max_metric_regression < 0:
            raise ValueError("max_metric_regression no puede ser negativo.")


def build_deployment_config() -> DeploymentConfig:
    """Build deployment settings from TOML with environment overrides."""

    file_config = load_file_config()
    common = dict(file_config.get("common", {}))
    deployment = dict(file_config.get("deployment", {}))
    runtime_mode = detect_runtime()
    runtime_config = dict(file_config.get("runtime", {}).get(runtime_mode, {}))

    def setting(name: str, fallback: Any) -> str:
        return get_setting(name, str(fallback), name)

    return DeploymentConfig(
        model_name=setting(
            "IRIS_DEPLOYMENT_MODEL_NAME",
            runtime_config.get(
                "registered_model_name",
                deployment.get("model_name", common.get("registered_model_name", "")),
            ),
        ),
        endpoint_name=setting(
            "IRIS_SERVING_ENDPOINT_NAME", deployment.get("endpoint_name", "iris-classifier")
        ),
        min_test_f1_weighted=float(
            setting("IRIS_MIN_TEST_F1_WEIGHTED", deployment.get("min_test_f1_weighted", 0.90))
        ),
        min_test_accuracy=float(
            setting("IRIS_MIN_TEST_ACCURACY", deployment.get("min_test_accuracy", 0.90))
        ),
        max_metric_regression=float(
            setting("IRIS_MAX_METRIC_REGRESSION", deployment.get("max_metric_regression", 0.01))
        ),
        required_approval_tag=setting(
            "IRIS_REQUIRED_APPROVAL_TAG",
            deployment.get("required_approval_tag", "Approval_Check"),
        ),
        champion_alias=setting(
            "IRIS_DEPLOYMENT_CHAMPION_ALIAS", deployment.get("champion_alias", "Champion")
        ),
        challenger_alias=setting(
            "IRIS_DEPLOYMENT_CHALLENGER_ALIAS", deployment.get("challenger_alias", "Challenger")
        ),
        smoke_test_rows=int(setting("IRIS_SMOKE_TEST_ROWS", deployment.get("smoke_test_rows", 1))),
        serving_timeout_seconds=int(
            setting("IRIS_SERVING_TIMEOUT_SECONDS", deployment.get("serving_timeout_seconds", 900))
        ),
        serving_poll_seconds=int(
            setting("IRIS_SERVING_POLL_SECONDS", deployment.get("serving_poll_seconds", 15))
        ),
        notebook_root=setting(
            "IRIS_DEPLOYMENT_NOTEBOOK_ROOT",
            deployment.get("notebook_root", "/Workspace/Shared/iris_mlflow/deployment"),
        ),
        job_name=setting(
            "IRIS_DEPLOYMENT_JOB_NAME", deployment.get("job_name", "iris-model-deployment")
        ),
        runtime_mode=runtime_mode,
    )


def _file_value(common: dict[str, Any], name: str, fallback: Any) -> Any:
    return common.get(name, fallback)


def build_config(
    *,
    model_slug: str,
    registered_model_name: str | None = None,
    run_name: str | None = None,
    model_defaults: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Build config from TOML, environment, and optional widget overrides."""

    file_config = load_file_config()
    common = dict(file_config.get("common", {}))
    runtime_mode = detect_runtime()
    runtime_config = dict(file_config.get("runtime", {}).get(runtime_mode, {}))
    model = dict(file_config.get("models", {}).get(model_slug, {}))
    params = dict(model.pop("params", {}))
    if model_defaults:
        params.update(model_defaults)
    default_name = registered_model_name or str(common.get("registered_model_name", ""))
    default_run_name = run_name or str(model.get("run_name", model_slug))
    databricks_default_experiment = "/Shared/iris_mlflow" if is_databricks() else "iris_mlflow"

    def setting(name: str, fallback: Any) -> str:
        return get_setting(name, str(fallback), name)

    config_root = _find_config_path().parent.parent
    return TrainingConfig(
        experiment_name=setting(
            "IRIS_EXPERIMENT_NAME",
            runtime_config.get(
                "experiment_name",
                _file_value(common, "experiment_name", databricks_default_experiment),
            ),
        ),
        artifact_location=setting("IRIS_ARTIFACT_LOCATION", ""),
        tracking_uri=setting("MLFLOW_TRACKING_URI", runtime_config.get("tracking_uri", "")),
        registry_uri=setting(
            "MLFLOW_REGISTRY_URI", runtime_config.get("registry_uri", "databricks-uc")
        ),
        registered_model_name=setting(
            "IRIS_REGISTERED_MODEL_NAME", runtime_config.get("registered_model_name", default_name)
        ),
        feature_table=setting(
            "IRIS_FEATURE_TABLE",
            runtime_config.get(
                "feature_table",
                _file_value(common, "feature_table", "workspace.default.iris_features"),
            ),
        ),
        champion_alias=setting(
            "IRIS_CHAMPION_ALIAS", _file_value(common, "champion_alias", "Champion")
        ),
        challenger_alias=setting(
            "IRIS_CHALLENGER_ALIAS", _file_value(common, "challenger_alias", "Challenger")
        ),
        run_name=setting("IRIS_RUN_NAME", default_run_name),
        dataset_version=setting(
            "IRIS_DATASET_VERSION", _file_value(common, "dataset_version", "iris-features-delta")
        ),
        project_version=setting(
            "IRIS_PROJECT_VERSION", _file_value(common, "project_version", "2.0.0")
        ),
        author=setting("IRIS_AUTHOR", _file_value(common, "author", "unknown")),
        purpose=setting("IRIS_PURPOSE", _file_value(common, "purpose", "baseline-classification")),
        test_size=float(setting("IRIS_TEST_SIZE", _file_value(common, "test_size", 0.20))),
        random_state=int(setting("IRIS_RANDOM_STATE", _file_value(common, "random_state", 42))),
        primary_metric=setting(
            "IRIS_PRIMARY_METRIC", _file_value(common, "primary_metric", "test_f1_weighted")
        ),
        enable_tracing=setting(
            "MLFLOW_ENABLE_TRACING", _file_value(common, "enable_tracing", True)
        ).lower()
        in {"1", "true", "yes", "si", "sí"},
        model_input_example_rows=int(
            setting(
                "IRIS_MODEL_INPUT_EXAMPLE_ROWS", _file_value(common, "model_input_example_rows", 5)
            )
        ),
        target_column=setting(
            "IRIS_TARGET_COLUMN", _file_value(common, "target_column", "Species")
        ),
        model_registration_timeout_seconds=int(
            setting(
                "MLFLOW_MODEL_REGISTRATION_TIMEOUT_SECONDS",
                _file_value(common, "model_registration_timeout_seconds", 300),
            )
        ),
        model_registration_poll_seconds=int(
            setting(
                "MLFLOW_MODEL_REGISTRATION_POLL_SECONDS",
                _file_value(common, "model_registration_poll_seconds", 5),
            )
        ),
        feature_table_version=setting(
            "IRIS_FEATURE_TABLE_VERSION", _file_value(common, "feature_table_version", "")
        ),
        model_type=str(model.get("model_type", "")),
        model_framework=str(model.get("model_framework", "")),
        model_params=params,
        runtime_mode=runtime_mode,
        dataset_path=(
            _resolve_project_path(
                get_runtime_parameter(
                    "IRIS_LOCAL_DATASET_PATH",
                    str(runtime_config.get("dataset_path", "")),
                    runtime_mode,
                ),
                config_root,
            )
            if runtime_mode == "local"
            else None
        ),
        auto_approve=str(runtime_config.get("auto_approve", False)).lower() in {"1", "true", "yes"},
        deployment_manifest_path=_resolve_project_path(
            str(
                runtime_config.get(
                    "deployment_manifest_path", "artifacts/local_deployment_manifest.json"
                )
            ),
            config_root,
        ),
    )


def build_runtime_config(*, model_slug: str) -> TrainingConfig:
    """Build a configuration selected by automatic runtime detection."""

    return build_config(model_slug=model_slug)


def _resolve_project_path(value: str, project_root: Path) -> Path:
    """Resolve paths in TOML relative to the repository root."""

    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()
