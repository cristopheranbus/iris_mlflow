"""MLflow tracking and Unity Catalog registration helpers."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import logging

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature

# CommandContext.extraContext() is not whitelisted on Databricks Serverless;
# MLflow's context registry warns on every run — suppress at WARNING level.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

from .config import TrainingConfig
from .data import SplitBundle
from .evaluation import EvaluationResult, flatten_metrics


@dataclass(frozen=True)
class TrainingRunResult:
    """Identifiers produced by a tracked and registered training run."""

    run_id: str
    model_uri: str
    registered_model_name: str
    registered_model_version: str | None
    metrics: dict[str, float]


def get_or_create_experiment(name: str, artifact_location: str = "") -> str:
    """Return an existing experiment id or create it once."""

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return cast(str, experiment.experiment_id)
    if artifact_location:
        return client.create_experiment(name, artifact_location=artifact_location)
    return client.create_experiment(name)


def _log_confusion_matrix(result: EvaluationResult, classes: tuple[str, ...]) -> None:
    """Create and log a temporary confusion matrix image."""

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(result.confusion_matrix, cmap="Blues")
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicción",
        ylabel="Valor real",
        title="Matriz de confusión",
    )
    for row in range(result.confusion_matrix.shape[0]):
        for column in range(result.confusion_matrix.shape[1]):
            axis.text(column, row, result.confusion_matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "confusion_matrix.png"
        figure.savefig(path, dpi=150)
        mlflow.log_artifact(str(path))
    plt.close(figure)


def log_training_run(
    *,
    model: Any,
    model_type: str,
    model_params: dict[str, Any],
    config: TrainingConfig,
    split: SplitBundle,
    evaluations: dict[str, EvaluationResult],
    feature_columns: tuple[str, ...],
    classes: tuple[str, ...],
) -> TrainingRunResult:
    """Track, serialize, and register a fitted sklearn-compatible model."""

    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(config.registry_uri)
    experiment_id = get_or_create_experiment(config.experiment_name, config.artifact_location)
    mlflow.set_experiment(config.experiment_name)

    metrics = flatten_metrics(evaluations)
    params = {
        **model_params,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "dataset_version": config.dataset_version,
        "feature_columns": json.dumps(feature_columns),
        "train_rows": len(split.x_train),
        "test_rows": len(split.x_test),
        "class_count": len(classes),
    }
    tags = {
        "model_type": model_type,
        "dataset": config.dataset_version,
        "project_version": config.project_version,
        "author": config.author,
        "purpose": config.purpose,
        "primary_metric": config.primary_metric,
    }

    with mlflow.start_run(run_name=config.run_name, experiment_id=experiment_id) as run:
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(evaluations["test"].report, "classification_report.json")
        mlflow.log_dict(
            {str(index): class_name for index, class_name in enumerate(classes)},
            "class_mapping.json",
        )
        _log_confusion_matrix(evaluations["test"], classes)

        signature = infer_signature(split.x_train, model.predict(split.x_train))
        flavor = mlflow.xgboost if model_type == "XGBoost" else mlflow.sklearn
        model_info = flavor.log_model(
            model,
            name="model",
            signature=signature,
            input_example=split.x_train.head(5),
            registered_model_name=config.registered_model_name,
        )
        run_id = run.info.run_id

    version = getattr(model_info, "registered_model_version", None)
    return TrainingRunResult(
        run_id=run_id,
        model_uri=f"runs:/{run_id}/model",
        registered_model_name=config.registered_model_name,
        registered_model_version=str(version) if version is not None else None,
        metrics=metrics,
    )
