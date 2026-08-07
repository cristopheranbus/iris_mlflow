"""Unit tests for the reusable notebook training helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from iris_mlflow_utils import evaluate_model, load_dataset, split_dataset


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


def test_split_dataset_is_reproducible(tmp_path: Path) -> None:
    bundle = load_dataset(iris_csv(tmp_path))
    first = split_dataset(bundle, test_size=0.50, random_state=42)
    second = split_dataset(bundle, test_size=0.50, random_state=42)

    pd.testing.assert_frame_equal(first.x_train, second.x_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)


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
