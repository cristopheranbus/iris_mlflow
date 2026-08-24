"""Unit tests for model evaluation helpers and artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from iris_mlflow_utils.evaluation import (
    build_classification_table,
    build_evaluation_artifacts,
    build_metrics_summary_table,
    build_probability_metrics,
    evaluate_model,
    evaluate_train_test,
    flatten_metrics,
)

pytestmark = pytest.mark.unit


def binary_data() -> tuple[pd.DataFrame, np.ndarray]:
    return pd.DataFrame({"feature": [0.0, 0.1, 1.0, 1.1]}), np.array([0, 0, 1, 1])


def test_evaluate_model_and_train_test_return_uniform_metrics() -> None:
    features, target = binary_data()
    model = LogisticRegression().fit(features, target)
    result = evaluate_model(model, features, target, [0, 1])
    results = evaluate_train_test(model, features, features, target, target, 2)

    assert result.metrics["accuracy"] == 1.0
    assert result.confusion_matrix.shape == (2, 2)
    assert result.probabilities is not None
    assert set(flatten_metrics(results)) == {
        f"{partition}_{metric}" for partition in ("train", "test") for metric in result.metrics
    }


def test_evaluate_model_supports_classifier_without_probabilities() -> None:
    features, target = binary_data()

    class PredictOnly:
        @staticmethod
        def predict(values: pd.DataFrame) -> np.ndarray:
            return target

    result = evaluate_model(PredictOnly(), features, target, [0, 1])
    assert result.probabilities is None
    assert build_probability_metrics(result, target, [0, 1]) == {}


def test_artifacts_without_probabilities_stop_after_confusion_matrices(tmp_path: Path) -> None:
    features, target = binary_data()

    class PredictOnly:
        @staticmethod
        def predict(values: pd.DataFrame) -> np.ndarray:
            return target

    model = PredictOnly()
    result = evaluate_model(model, features, target, [0, 1])
    paths = build_evaluation_artifacts(
        model,
        result,
        labels=[0, 1],
        class_names=("no", "yes"),
        features=features,
        target=target,
        output_dir=tmp_path,
    )
    assert set(paths) == {"confusion_matrix.png", "confusion_matrix_normalized.png"}


def test_evaluation_tables_have_stable_columns_and_metadata() -> None:
    features, target = binary_data()
    result = evaluate_model(LogisticRegression().fit(features, target), features, target, [0, 1])
    metrics = build_metrics_summary_table(
        {"test": result},
        model_type="logistic",
        dataset_version="1",
        project_version="2",
        run_id="run",
    )
    classes = build_classification_table(
        {"test": result},
        ("no", "yes"),
        model_type="logistic",
        dataset_version="1",
        project_version="2",
        run_id="run",
    )
    assert {"partition", "metric", "value", "run_id"}.issubset(metrics.columns)
    assert {"class_id", "class_name", "precision", "f1_score", "run_id"}.issubset(classes.columns)


def test_multiclass_probability_metrics_and_artifacts(tmp_path: Path) -> None:
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
    assert metrics["log_loss"] >= 0
    assert {"confusion_matrix.png", "roc_curve_by_class.png", "lift_curve_macro.png"}.issubset(
        paths
    )
    gain = pd.read_json(tmp_path / "evaluation" / "cumulative_gain_data.json")
    assert np.allclose(gain.groupby("class_id", as_index=False).tail(1)["cumulative_gain"], 1.0)
    assert "feature_importance.json" not in paths


def test_artifacts_include_feature_importance_when_model_exposes_it(tmp_path: Path) -> None:
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = np.asarray(iris.target)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(features, target)
    result = evaluate_model(model, features, target, [0, 1, 2])
    paths = build_evaluation_artifacts(
        model,
        result,
        labels=[0, 1, 2],
        class_names=tuple(iris.target_names),
        features=features,
        target=target,
        output_dir=tmp_path,
    )
    assert {"feature_importance.json", "feature_importance.png"}.issubset(paths)
