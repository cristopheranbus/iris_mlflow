"""Model evaluation helpers shared by the training notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .data import SplitBundle


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
    model: Any, split: SplitBundle, class_count: int
) -> dict[str, EvaluationResult]:
    """Evaluate a fitted classifier consistently on train and test partitions."""

    labels = list(range(class_count))
    return {
        "train": evaluate_model(model, split.x_train, split.y_train, labels),
        "test": evaluate_model(model, split.x_test, split.y_test, labels),
    }


def flatten_metrics(results: dict[str, EvaluationResult]) -> dict[str, float]:
    """Prefix partition names so metrics are unambiguous in MLflow."""

    return {
        f"{partition}_{name}": value
        for partition, result in results.items()
        for name, value in result.metrics.items()
    }
