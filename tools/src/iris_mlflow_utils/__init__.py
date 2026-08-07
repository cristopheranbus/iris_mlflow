"""Reusable data, evaluation, and MLflow helpers for the Iris notebooks."""

import logging

# CommandContext.extraContext() is not whitelisted on Databricks Serverless;
# MLflow's context registry warns on every span/run start — suppress at WARNING level.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

from .config import TrainingConfig, build_config  # noqa: E402
from .data import DatasetBundle, load_dataset, load_dataset_frame  # noqa: E402
from .evaluation import (  # noqa: E402
    EvaluationResult,
    build_classification_table,
    build_metrics_summary_table,
    evaluate_model,
    evaluate_train_test,
)
from .feature_table import ensure_feature_table  # noqa: E402

__all__ = [
    "DatasetBundle",
    "EvaluationResult",
    "TrainingConfig",
    "build_classification_table",
    "build_config",
    "build_metrics_summary_table",
    "evaluate_model",
    "evaluate_train_test",
    "ensure_feature_table",
    "load_dataset",
    "load_dataset_frame",
]
