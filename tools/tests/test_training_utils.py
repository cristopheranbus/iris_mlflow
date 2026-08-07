"""Unit tests for the reusable notebook training helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow.models import evaluate
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from iris_mlflow_utils import (
    build_classification_table,
    build_metrics_summary_table,
    evaluate_model,
    load_dataset,
    load_dataset_frame,
)


def iris_csv(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6],
            "SepalLengthCm": [5.1, 5.0, 6.4, 6.3, 6.7, 6.8],
            "SepalWidthCm": [3.5, 3.4, 3.2, 3.3, 3.1, 3.0],
            "PetalLengthCm": [1.4, 1.5, 4.5, 4.7, 5.6, 5.5],
            "PetalWidthCm": [0.2, 0.2, 1.5, 1.6, 2.4, 2.1],
            "Species": ["setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica"],
        }
    )
    path = tmp_path / "Iris.csv"
    data.to_csv(path, index=False)
    return path


def test_load_dataset_validates_and_encodes_labels(tmp_path: Path) -> None:
    bundle = load_dataset(iris_csv(tmp_path))

    assert bundle.feature_columns == (
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    )
    assert bundle.classes == ("setosa", "versicolor", "virginica")
    assert np.array_equal(bundle.target, np.array([0, 0, 1, 1, 2, 2]))


def test_load_dataset_rejects_missing_target(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="columna objetivo"):
        load_dataset(path)


def test_load_dataset_frame_matches_file_loader(tmp_path: Path) -> None:
    path = iris_csv(tmp_path)
    from_file = load_dataset(path)
    from_frame = load_dataset_frame(pd.read_csv(path))

    pd.testing.assert_frame_equal(from_file.features, from_frame.features)
    np.testing.assert_array_equal(from_file.target, from_frame.target)


def test_official_train_test_split_is_reproducible(tmp_path: Path) -> None:
    bundle = load_dataset(iris_csv(tmp_path))
    first = train_test_split(
        bundle.features,
        bundle.target,
        test_size=0.50,
        random_state=42,
        stratify=bundle.target,
    )
    second = train_test_split(
        bundle.features,
        bundle.target,
        test_size=0.50,
        random_state=42,
        stratify=bundle.target,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    np.testing.assert_array_equal(first[3], second[3])


def test_evaluate_model_returns_uniform_metrics() -> None:
    features = pd.DataFrame({"feature": [0.0, 0.1, 1.0, 1.1]})
    target = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(random_state=42).fit(features, target)

    result = evaluate_model(model, features, target, labels=[0, 1])

    assert set(result.metrics) == {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    }
    assert result.metrics["accuracy"] == 1.0
    assert result.confusion_matrix.shape == (2, 2)


def test_evaluation_tables_have_stable_columns() -> None:
    features = pd.DataFrame({"feature": [0.0, 0.1, 1.0, 1.1]})
    target = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(random_state=42).fit(features, target)
    result = evaluate_model(model, features, target, labels=[0, 1])
    evaluations = {"test": result}

    metrics_table = build_metrics_summary_table(
        evaluations,
        model_type="RandomForest",
        dataset_version="test",
        project_version="test",
    )
    classes_table = build_classification_table(
        evaluations,
        ("setosa", "versicolor"),
        model_type="RandomForest",
        dataset_version="test",
        project_version="test",
    )

    assert {"partition", "metric", "value"}.issubset(metrics_table.columns)
    assert {"class_id", "class_name", "precision", "f1_score"}.issubset(classes_table.columns)


def test_mlflow_evaluation_returns_metrics() -> None:
    features = pd.DataFrame({"feature": [0.0, 0.1, 1.0, 1.1]})
    target = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(random_state=42).fit(features, target)

    result = evaluate(  # type: ignore[no-untyped-call]
        model=model.predict,
        data=features.assign(target=target),
        targets="target",
        model_type="classifier",
        evaluator_config={"log_model_explainability": False},
    )

    assert "accuracy_score" in result.metrics


def test_notebooks_use_idempotent_unity_catalog_feature_table() -> None:
    repository_root = Path(__file__).parents[2]
    for notebook_name in ("random_forest.ipynb", "xgboost.ipynb"):
        notebook = json.loads((repository_root / notebook_name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert 'FEATURE_TABLE = "workspace.default.iris_features"' in source
        assert "ensure_feature_table" in source
        assert "spark.table(FEATURE_TABLE)" in source
        assert 'registered_model_name="workspace.default.iris_classifier"' in source
