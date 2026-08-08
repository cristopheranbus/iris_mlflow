"""Reusable data, evaluation, and MLflow helpers for the Iris notebooks."""

import logging

# CommandContext.extraContext() is not whitelisted on Databricks Serverless;
# MLflow's context registry warns on every span/run start — suppress at WARNING level.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

from .config import TrainingConfig, build_config, load_file_config  # noqa: E402
from .data import (  # noqa: E402
    DatasetBundle,
    load_dataset,
    load_dataset_frame,
    load_dataset_from_spark,
)
from .evaluation import (  # noqa: E402
    EvaluationResult,
    build_classification_table,
    build_metrics_summary_table,
    evaluate_model,
    evaluate_train_test,
)
from .feature_table import ensure_feature_table  # noqa: E402
from .registry import (  # noqa: E402
    build_registry_client,
    synchronize_model_registry_metadata,
)
from .serving import (  # noqa: E402
    DatabricksEndpointError,
    build_invocation_url,
    extract_predictions,
    predict,
    predict_dataframe,
    read_configuration,
)

__all__ = [
    "DatasetBundle",
    "EvaluationResult",
    "TrainingConfig",
    "build_classification_table",
    "build_config",
    "load_file_config",
    "build_metrics_summary_table",
    "evaluate_model",
    "evaluate_train_test",
    "ensure_feature_table",
    "build_registry_client",
    "synchronize_model_registry_metadata",
    "DatabricksEndpointError",
    "build_invocation_url",
    "extract_predictions",
    "load_dataset",
    "load_dataset_frame",
    "load_dataset_from_spark",
    "predict",
    "predict_dataframe",
    "read_configuration",
]
