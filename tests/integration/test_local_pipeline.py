"""Integration test for the real local dataset and deployment manifest."""

import json
from pathlib import Path

import pytest
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from iris_mlflow_utils import evaluate_model, load_dataset
from iris_mlflow_utils.local_deployment import approve_locally, simulate_local_deployment

pytestmark = pytest.mark.integration


class Registry:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}
        self.aliases: dict[str, str] = {}

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags[key] = value

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        raise LookupError(alias)


def test_local_dataset_can_be_evaluated_and_promoted(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    dataset = load_dataset(root / "data" / "local" / "iris_features.csv")
    model = DecisionTreeClassifier(random_state=42).fit(dataset.features, dataset.target)
    evaluation = evaluate_model(
        model, dataset.features, dataset.target, list(range(len(dataset.classes)))
    )
    assert evaluation.metrics["accuracy"] == 1.0

    registry = Registry()
    approve_locally(registry, model_name="iris", model_version="1")
    manifest = tmp_path / "deployment.json"
    simulate_local_deployment(
        registry,
        model_name="iris",
        model_version="1",
        champion_alias="Champion",
        manifest_path=manifest,
        smoke_test_passed=True,
    )
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "validated"
