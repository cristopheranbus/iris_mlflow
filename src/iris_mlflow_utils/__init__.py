"""Reusable data, evaluation, and MLflow helpers for the Iris notebooks."""

import logging

# CommandContext.extraContext() is not whitelisted on Databricks Serverless;
# MLflow's context registry warns on every span/run start — suppress at WARNING level.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

from .config import (  # noqa: E402
    DeploymentConfig,
    PromotionPolicy,
    PromotionRule,
    TrainingConfig,
    build_config,
    build_deployment_config,
    build_runtime_config,
    load_file_config,
    load_promotion_policy,
)
from .data import (  # noqa: E402
    DatasetBundle,
    get_delta_table_version,
    load_dataset,
    load_dataset_for_runtime,
    load_dataset_frame,
    load_dataset_from_spark,
)
from .deployment import (  # noqa: E402
    PromotionDecision,
    RuleResult,
    ServingEndpointChange,
    ServingEndpointSnapshot,
    capture_serving_endpoint,
    evaluate_promotion_gate,
    promote_champion,
    restore_serving_endpoint,
    rollback_serving_endpoint,
    upsert_serving_endpoint,
    wait_for_endpoint_ready,
)
from .evaluation import (  # noqa: E402
    CrossValidationSummary,
    EvaluationResult,
    build_classification_table,
    build_evaluation_artifacts,
    build_metrics_summary_table,
    build_probability_metrics,
    cross_validate_classifier,
    evaluate_model,
    evaluate_train_test,
)
from .feature_table import ensure_feature_table  # noqa: E402
from .local_deployment import (  # noqa: E402
    approve_locally,
    simulate_local_deployment,
    write_manifest,
)
from .monitoring import (  # noqa: E402
    MonitoringAlert,
    MonitoringConfig,
    MonitoringDecision,
    MonitoringSnapshot,
    evaluate_monitoring,
    load_monitoring_config,
)
from .registry import (  # noqa: E402
    build_registry_client,
    ensure_mlflow_experiment,
    get_model_evaluation_metrics,
    synchronize_model_registry_metadata,
)
from .runtime import (  # noqa: E402
    RuntimeMode,
    detect_runtime,
    get_dbutils,
    get_runtime_parameter,
    is_databricks_runtime,
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
    "get_delta_table_version",
    "EvaluationResult",
    "TrainingConfig",
    "DeploymentConfig",
    "PromotionPolicy",
    "PromotionRule",
    "build_classification_table",
    "build_config",
    "build_deployment_config",
    "build_runtime_config",
    "load_file_config",
    "load_promotion_policy",
    "build_metrics_summary_table",
    "build_evaluation_artifacts",
    "build_probability_metrics",
    "evaluate_model",
    "evaluate_train_test",
    "cross_validate_classifier",
    "CrossValidationSummary",
    "ensure_feature_table",
    "build_registry_client",
    "ensure_mlflow_experiment",
    "get_model_evaluation_metrics",
    "synchronize_model_registry_metadata",
    "DatabricksEndpointError",
    "build_invocation_url",
    "extract_predictions",
    "load_dataset",
    "load_dataset_frame",
    "load_dataset_from_spark",
    "load_dataset_for_runtime",
    "PromotionDecision",
    "RuleResult",
    "ServingEndpointSnapshot",
    "ServingEndpointChange",
    "capture_serving_endpoint",
    "restore_serving_endpoint",
    "evaluate_promotion_gate",
    "upsert_serving_endpoint",
    "rollback_serving_endpoint",
    "wait_for_endpoint_ready",
    "promote_champion",
    "approve_locally",
    "simulate_local_deployment",
    "write_manifest",
    "MonitoringAlert",
    "MonitoringConfig",
    "MonitoringDecision",
    "MonitoringSnapshot",
    "evaluate_monitoring",
    "load_monitoring_config",
    "predict",
    "predict_dataframe",
    "read_configuration",
    "RuntimeMode",
    "detect_runtime",
    "is_databricks_runtime",
    "get_runtime_parameter",
    "get_dbutils",
]
