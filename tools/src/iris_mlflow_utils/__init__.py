"""Reusable data, evaluation, and MLflow helpers for the Iris notebooks."""

import logging

# CommandContext.extraContext() is not whitelisted on Databricks Serverless;
# MLflow's context registry warns on every span/run start — suppress at WARNING level.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

from .config import TrainingConfig, build_config
from .data import DatasetBundle, load_dataset
from .evaluation import (
    EvaluationResult,
    build_classification_table,
    build_metrics_summary_table,
    evaluate_model,
    evaluate_train_test,
)

__all__ = [
    "DatasetBundle",
    "EvaluationResult",
    "TrainingConfig",
    "build_classification_table",
    "build_config",
    "build_metrics_summary_table",
    "evaluate_model",
    "evaluate_train_test",
    "load_dataset",
]
