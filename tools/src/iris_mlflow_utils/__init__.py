"""Reusable data, evaluation, and MLflow helpers for the Iris notebooks."""

import logging

# CommandContext.extraContext() is not whitelisted on Databricks Serverless;
# MLflow's context registry warns on every span/run start — suppress at WARNING level.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

from .config import (  # noqa: E402
    DeploymentConfig,
    TrainingConfig,
    build_config,
    build_deployment_config,
    load_file_config,
)
from .data import (  # noqa: E402
    DatasetBundle,
    load_dataset,
    load_dataset_frame,
    load_dataset_from_spark,
)
from .deployment import (  # noqa: E402
    PromotionDecision,
    evaluate_promotion_gate,
    promote_champion,
    update_serving_endpoint,
    wait_for_endpoint_ready,
)
from .evaluation import (  # noqa: E402
    EvaluationResult,
    build_classification_table,
    build_evaluation_artifacts,
    build_metrics_summary_table,
    build_probability_metrics,
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
    "DeploymentConfig",
    "build_classification_table",
    "build_config",
    "build_deployment_config",
    "load_file_config",
    "build_metrics_summary_table",
    "build_evaluation_artifacts",
    "build_probability_metrics",
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
    "PromotionDecision",
    "evaluate_promotion_gate",
    "update_serving_endpoint",
    "wait_for_endpoint_ready",
    "promote_champion",
    "predict",
    "predict_dataframe",
    "read_configuration",
]
