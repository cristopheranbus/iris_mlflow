"""Tests for evaluation artifacts and promotion gates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from iris_mlflow_utils import (
    build_evaluation_artifacts,
    build_probability_metrics,
    evaluate_model,
    evaluate_promotion_gate,
)
from iris_mlflow_utils.config import DeploymentConfig, build_deployment_config


def test_multiclass_artifacts_include_roc_lift_gain_and_confusion(tmp_path: Path) -> None:
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = np.asarray(iris.target)
    model = LogisticRegression(max_iter=500).fit(features, target)
    result = evaluate_model(model, features, target, [0, 1, 2])

    metrics = build_probability_metrics(result, target, [0, 1, 2])
    paths = build_evaluation_artifacts(
        model,
        result,
        labels=[0, 1, 2],
        class_names=tuple(iris.target_names),
        features=features,
        target=target,
        output_dir=tmp_path / "evaluation",
    )

    assert metrics["roc_auc_ovr_macro"] > 0.9
    assert {"confusion_matrix.png", "roc_curve_by_class.png", "lift_curve_by_class.png"}.issubset(paths)
    assert (tmp_path / "evaluation" / "cumulative_gain_data.json").is_file()
    assert (tmp_path / "evaluation" / "feature_importance.json").is_file() is False


def test_promotion_gate_blocks_regression_and_accepts_candidate() -> None:
    config = DeploymentConfig(
        model_name="workspace.default.iris_classifier",
        endpoint_name="iris-classifier",
    )
    accepted = evaluate_promotion_gate(
        {"test_f1_weighted": 0.95, "test_accuracy": 0.96},
        {"test_f1_weighted": 0.94},
        config,
    )
    rejected = evaluate_promotion_gate(
        {"test_f1_weighted": 0.90, "test_accuracy": 0.95},
        {"test_f1_weighted": 0.95},
        config,
    )

    assert accepted.passed is True
    assert rejected.passed is False
    assert rejected.reason == "candidate_regresses_against_champion"


def test_deployment_notebooks_keep_only_dynamic_job_inputs() -> None:
    repository_root = Path(__file__).parents[1]
    for name in ("evaluate_model.ipynb", "approval.ipynb", "deploy_model.ipynb"):
        notebook = json.loads((repository_root / "deployment" / name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert "model_name" in source
        assert "model_version" in source
        assert "IRIS_DATA_PATH" not in source
        assert notebook["metadata"]["application/vnd.databricks.v1+notebook"]["widgets"] == {}


def test_deployment_config_comes_from_versioned_toml() -> None:
    config = build_deployment_config()

    assert config.model_name == "workspace.default.iris_classifier"
    assert config.endpoint_name == "iris-classifier"
    assert config.min_test_f1_weighted == 0.90
    assert config.required_approval_tag == "Approval_Check"
