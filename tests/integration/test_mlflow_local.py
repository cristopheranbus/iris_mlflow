"""Local integration tests backed by isolated SQLite stores."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import evaluate
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from iris_mlflow_utils.registry import ensure_mlflow_experiment

pytestmark = pytest.mark.integration


def test_ensure_mlflow_experiment_creates_reuses_and_restores(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    experiment_id = ensure_mlflow_experiment(
        "iris_local", tracking_uri=tracking_uri, artifact_location=str(tmp_path / "artifacts")
    )
    assert ensure_mlflow_experiment("iris_local", tracking_uri=tracking_uri) == experiment_id

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    client.delete_experiment(experiment_id)
    assert ensure_mlflow_experiment("iris_local", tracking_uri=tracking_uri) == experiment_id
    assert client.get_experiment(experiment_id).lifecycle_stage == "active"


def test_mlflow_evaluator_logs_classifier_metrics(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'evaluation.db').as_posix()}")
    mlflow.set_experiment("evaluation")
    features = pd.DataFrame({"feature": [0.0, 0.1, 1.0, 1.1]})
    target = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(random_state=42).fit(features, target)

    result = evaluate(  # type: ignore[no-untyped-call]
        model=model.predict,
        data=features.assign(target=target.astype("float64")),
        targets="target",
        model_type="classifier",
        evaluator_config={"log_model_explainability": False},
    )
    assert result.metrics["accuracy_score"] == 1.0
