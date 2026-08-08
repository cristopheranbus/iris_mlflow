"""Model Registry metadata, aliases, descriptions, and verification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mlflow import MlflowClient

from .config import TrainingConfig


def build_registry_client(registry_uri: str = "databricks-uc") -> MlflowClient:
    """Create an MLflow client explicitly bound to the Unity Catalog registry."""

    return MlflowClient(registry_uri=registry_uri)


def _descriptions(config: TrainingConfig, *, version: str, run_id: str) -> tuple[str, str]:
    model_type = config.model_type or "unknown"
    framework = config.model_framework or "unknown"
    model_description = (
        "Clasificador Iris gestionado en Unity Catalog. "
        f"Último algoritmo entrenado: {model_type}. Framework: {framework}. "
        "Las versiones históricas conservan su algoritmo en tags y descripción de versión."
    )
    version_description = (
        f"Versión {version} del clasificador Iris. Algoritmo: {model_type}. "
        f"Framework: {framework}. Dataset: {config.dataset_version}. "
        f"Feature table: {config.feature_table}. "
        f"Feature table version: {config.feature_table_version or 'latest'}. "
        f"Proyecto: {config.project_version}. Run: {run_id}. "
        f"Alias candidato: {config.challenger_alias}."
    )
    return model_description, version_description


def synchronize_model_registry_metadata(
    client: Any,
    *,
    config: TrainingConfig,
    version: str | int,
    run_id: str,
) -> dict[str, Any]:
    """Apply and verify model/version tags, descriptions, and Challenger alias."""

    model_name = config.registered_model_name
    version_string = str(version)
    model_type = config.model_type or "unknown"
    framework = config.model_framework or "unknown"
    version_tags = {
        "model_type": model_type,
        "model_framework": framework,
        "task": "classification",
        "dataset": "iris",
        "feature_source": "unity_catalog",
        "feature_table": config.feature_table,
        "training_stage": "challenger",
    }
    model_tags = {
        "model_type": model_type,
        "model_framework": framework,
        "task": "classification",
        "feature_source": "unity_catalog",
        "latest_version": version_string,
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, version_string, key, value)
    for key, value in model_tags.items():
        client.set_registered_model_tag(model_name, key, value)

    model_description, version_description = _descriptions(
        config, version=version_string, run_id=run_id
    )
    client.update_registered_model(name=model_name, description=model_description)
    client.update_model_version(
        name=model_name,
        version=version_string,
        description=version_description,
    )
    client.set_registered_model_alias(
        name=model_name,
        alias=config.challenger_alias,
        version=version_string,
    )

    registered_model = client.get_registered_model(model_name)
    model_version = client.get_model_version(model_name, version_string)
    alias_version = client.get_model_version_by_alias(model_name, config.challenger_alias)
    observed_version_tags = {key: str(model_version.tags.get(key, "")) for key in version_tags}
    observed_model_tags = {key: str(registered_model.tags.get(key, "")) for key in model_tags}
    version_tags_verified = observed_version_tags == version_tags
    model_tags_verified = observed_model_tags == model_tags
    alias_verified = str(alias_version.version) == version_string
    descriptions_verified = bool(model_version.description and registered_model.description)
    evidence = {
        "registry_uri": client._registry_uri,
        "model_name": model_name,
        "version": version_string,
        "run_id": run_id,
        "model_type": model_type,
        "model_framework": framework,
        "challenger_alias": config.challenger_alias,
        "alias_verified": alias_verified,
        "version_tags_verified": version_tags_verified,
        "registered_model_tags_verified": model_tags_verified,
        "descriptions_verified": descriptions_verified,
        "verified_at": datetime.now(UTC).isoformat(),
        "observed_version_tags": observed_version_tags,
        "observed_model_tags": observed_model_tags,
    }
    if not all((alias_verified, version_tags_verified, model_tags_verified, descriptions_verified)):
        raise RuntimeError(f"La metadata del registry no pudo verificarse: {evidence}")
    return evidence
