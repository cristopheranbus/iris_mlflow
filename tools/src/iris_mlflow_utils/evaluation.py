"""Model evaluation helpers shared by the training notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and diagnostics for one model/data partition."""

    metrics: dict[str, float]
    report: dict[str, Any]
    confusion_matrix: np.ndarray
    predictions: np.ndarray


def evaluate_model(
    model: Any, features: Any, target: np.ndarray, labels: list[int]
) -> EvaluationResult:
    """Evaluate a fitted classifier and return metrics plus diagnostics."""

    predictions = np.asarray(model.predict(features))
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predictions, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }
    return EvaluationResult(
        metrics=metrics,
        report=classification_report(
            target, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        confusion_matrix=confusion_matrix(target, predictions, labels=labels),
        predictions=predictions,
    )


def evaluate_train_test(
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_count: int,
) -> dict[str, EvaluationResult]:
    """Evaluate a fitted classifier consistently on train and test partitions."""

    labels = list(range(class_count))
    return {
        "train": evaluate_model(model, x_train, y_train, labels),
        "test": evaluate_model(model, x_test, y_test, labels),
    }


def flatten_metrics(results: dict[str, EvaluationResult]) -> dict[str, float]:
    """Prefix partition names so metrics are unambiguous in MLflow."""

    return {
        f"{partition}_{name}": value
        for partition, result in results.items()
        for name, value in result.metrics.items()
    }


def build_metrics_summary_table(
    results: dict[str, EvaluationResult],
    *,
    model_type: str,
    dataset_version: str,
    project_version: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Build an MLflow table with one row per partition and metric."""

    rows = [
        {
            "model_type": model_type,
            "partition": partition,
            "metric": metric,
            "value": value,
            "dataset_version": dataset_version,
            "project_version": project_version,
            "run_id": run_id,
        }
        for partition, result in results.items()
        for metric, value in result.metrics.items()
    ]
    return pd.DataFrame(rows)


def build_classification_table(
    results: dict[str, EvaluationResult],
    classes: tuple[str, ...],
    *,
    model_type: str,
    dataset_version: str,
    project_version: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Normalize per-class reports into a table suitable for MLflow."""

    rows: list[dict[str, Any]] = []
    for partition, result in results.items():
        for class_id, class_name in enumerate(classes):
            values = result.report.get(str(class_id), {})
            rows.append(
                {
                    "model_type": model_type,
                    "partition": partition,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": float(values.get("precision", 0.0)),
                    "recall": float(values.get("recall", 0.0)),
                    "f1_score": float(values.get("f1-score", 0.0)),
                    "support": int(values.get("support", 0)),
                    "dataset_version": dataset_version,
                    "project_version": project_version,
                    "run_id": run_id,
                }
            )
    return pd.DataFrame(rows)
