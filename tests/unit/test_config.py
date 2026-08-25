"""Unit tests for versioned configuration and validation."""

from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import pytest

import iris_mlflow_utils.config as config_module
from iris_mlflow_utils.config import (
    DeploymentConfig,
    PromotionPolicy,
    PromotionRule,
    TrainingConfig,
    build_config,
    build_deployment_config,
    build_runtime_config,
    get_setting,
    is_databricks,
    load_file_config,
    load_promotion_policy,
)

pytestmark = pytest.mark.unit


def training_config(**overrides: Any) -> TrainingConfig:
    base = TrainingConfig(
        dataset_path=None,
        experiment_name="iris_mlflow",
        artifact_location="",
        tracking_uri="",
        registry_uri="databricks-uc",
        registered_model_name="workspace.default.iris_classifier",
        feature_table="workspace.default.iris_features",
        champion_alias="Champion",
        challenger_alias="Challenger",
        run_name="test",
        dataset_version="test",
        project_version="test",
        author="test",
        purpose="test",
    )
    return replace(base, **overrides)


def test_load_file_config_honors_explicit_and_environment_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.toml"
    environment = tmp_path / "environment.toml"
    explicit.write_text('[common]\nauthor = "explicit"\n', encoding="utf-8")
    environment.write_text('[common]\nauthor = "environment"\n', encoding="utf-8")
    monkeypatch.setenv("IRIS_CONFIG_PATH", str(environment))

    assert load_file_config(explicit)["common"]["author"] == "explicit"
    assert load_file_config()["common"]["author"] == "environment"


def test_load_file_config_fails_when_no_candidate_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_root = Path(tmp_path.anchor) / "iris-mlflow-no-config"
    monkeypatch.setattr("iris_mlflow_utils.config.Path.cwd", classmethod(lambda cls: isolated_root))
    with pytest.raises(FileNotFoundError, match="IRIS_CONFIG_PATH"):
        load_file_config()


def test_versioned_promotion_profiles_are_loadable() -> None:
    development = load_promotion_policy(profile="dev")
    production = load_promotion_policy(profile="prod")

    assert development.name == "iris-development"
    assert production.name == "iris-production"
    assert len(production.rules) == 4
    assert production.rules[2].baseline == "champion"


def test_promotion_policy_rejects_unsafe_or_ambiguous_rules() -> None:
    with pytest.raises(ValueError, match="name y metric"):
        PromotionRule("", "metric", ">=", value=1)
    with pytest.raises(ValueError, match="Operador"):
        PromotionRule("rule", "metric", "==", value=1)
    with pytest.raises(ValueError, match="baseline"):
        PromotionRule("rule", "metric", ">=", baseline="candidate")
    with pytest.raises(ValueError, match="requiere value"):
        PromotionRule("rule", "metric", ">=")
    with pytest.raises(ValueError, match="no puede declarar value"):
        PromotionRule("rule", "metric", ">=", value=1, baseline="champion")
    with pytest.raises(ValueError, match="negativo"):
        PromotionRule("rule", "metric", ">=", baseline="champion", allowed_regression=-1)
    rule = PromotionRule("rule", "metric", ">=", value=0.9)
    with pytest.raises(ValueError, match="name y version"):
        PromotionPolicy("", "1", (rule,))
    with pytest.raises(ValueError, match="combination"):
        PromotionPolicy("policy", "1", (rule,), combination="any")
    with pytest.raises(ValueError, match="al menos una"):
        PromotionPolicy("policy", "1", ())
    with pytest.raises(ValueError, match="únicos"):
        PromotionPolicy(
            "policy",
            "1",
            (
                PromotionRule("duplicate", "metric", ">=", value=0.9),
                PromotionRule("duplicate", "other", ">=", value=0.9),
            ),
        )


def test_promotion_policy_path_validation_and_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="dev o prod"):
        load_promotion_policy(profile="staging")
    monkeypatch.setenv("IRIS_PROMOTION_POLICY_PATH", str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError, match="política"):
        load_promotion_policy(profile="dev")
    monkeypatch.delenv("IRIS_PROMOTION_POLICY_PATH")

    policy_path = tmp_path / "policy.toml"
    policy_path.write_text(
        '[policy]\nname="custom"\nversion="1"\n[[rules]]\n'
        'name="score"\nmetric="score"\noperator=">="\nvalue=0.5\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("IRIS_PROMOTION_POLICY_PATH", str(policy_path))
    assert load_promotion_policy(profile="prod").name == "custom"


def test_get_setting_precedence_and_widget_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SETTING", " environment ")
    assert get_setting("SETTING", "default") == "environment"

    monkeypatch.delenv("SETTING")
    assert get_setting("SETTING", "default") == "default"

    monkeypatch.setenv("IRIS_ENABLE_WIDGET_OVERRIDES", "sí")
    monkeypatch.setenv("IRIS_RUNTIME", "databricks")
    widgets = type("Widgets", (), {"get": lambda self, name: " widget "})()
    monkeypatch.setattr(
        "iris_mlflow_utils.config.get_dbutils", lambda: type("D", (), {"widgets": widgets})()
    )
    assert get_setting("SETTING", "default") == "widget"

    monkeypatch.setattr("iris_mlflow_utils.config.get_dbutils", lambda: None)
    assert get_setting("SETTING", "default") == "default"


def test_get_setting_handles_widget_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_ENABLE_WIDGET_OVERRIDES", "true")
    monkeypatch.setenv("IRIS_RUNTIME", "databricks")
    widgets = type(
        "Widgets", (), {"get": lambda self, name: (_ for _ in ()).throw(KeyError(name))}
    )()
    monkeypatch.setattr(
        "iris_mlflow_utils.config.get_dbutils", lambda: type("D", (), {"widgets": widgets})()
    )
    assert get_setting("SETTING", "default") == "default"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"test_size": 0}, "test_size"),
        ({"primary_metric": "loss"}, "primary_metric"),
        ({"feature_table": "invalid"}, "feature_table"),
        ({"champion_alias": ""}, "aliases"),
        ({"model_input_example_rows": 0}, "positivo"),
        ({"model_registration_timeout_seconds": 0}, "positivo"),
        ({"model_registration_poll_seconds": 0}, "positivo"),
        ({"model_type": "unsupported"}, "model_type"),
        ({"model_framework": "unsupported"}, "model_framework"),
    ],
)
def test_training_config_rejects_invalid_values(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        training_config(**changes)


def test_training_config_validates_local_model_name() -> None:
    assert training_config(runtime_mode="local", registered_model_name="iris_classifier")
    with pytest.raises(ValueError, match="modelo local"):
        training_config(runtime_mode="local", registered_model_name="invalid-model")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"model_name": "invalid"}, "catalog.schema.model"),
        ({"endpoint_name": " "}, "vacío"),
        ({"min_test_f1_weighted": -0.1}, "thresholds"),
        ({"min_test_accuracy": 1.1}, "thresholds"),
        ({"max_metric_regression": -0.1}, "negativo"),
        ({"smoke_test_rows": 0}, "smoke_test_rows"),
        ({"serving_timeout_seconds": 0}, "serving_timeout_seconds"),
        ({"serving_poll_seconds": -1}, "serving_poll_seconds"),
    ],
)
def test_deployment_config_rejects_invalid_values(changes: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "model_name": "workspace.default.iris_classifier",
        "endpoint_name": "iris",
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        DeploymentConfig(**values)  # type: ignore[arg-type]


def test_deployment_config_accepts_local_name() -> None:
    assert DeploymentConfig(
        model_name="iris_classifier", endpoint_name="local", runtime_mode="local"
    )


def test_build_configs_resolve_versioned_values_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_RUNTIME", "databricks")
    monkeypatch.setenv("IRIS_REGISTERED_MODEL_NAME", "workspace.default.override")
    config = build_config(model_slug="random_forest", model_defaults={"n_estimators": 12})
    deployment = build_deployment_config()

    assert config.registered_model_name == "workspace.default.override"
    assert config.model_params["n_estimators"] == 12
    assert deployment.endpoint_name == "iris-classifier"
    assert is_databricks() is True
    assert build_runtime_config(model_slug="xgboost").model_type == "xgboost"


def test_local_mlflow_uris_and_paths_are_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_RUNTIME", "local")
    config = build_config(model_slug="random_forest")
    project_root = Path(__file__).parents[2].resolve().as_posix()
    assert config.artifact_location == str(Path(project_root) / ".local/mlflow/artifacts")
    assert config.tracking_uri == f"sqlite:///{project_root}/.local/mlflow/mlflow.db"
    assert config.registry_uri == f"sqlite:///{project_root}/.local/mlflow/mlflow.db"
    assert config.dataset_path == Path(project_root) / "data/local/iris_features.csv"


def test_local_absolute_mlflow_uri_is_preserved(tmp_path: Path) -> None:
    database = (tmp_path / "mlflow.db").resolve()
    uri = config_module._runtime_mlflow_uri(
        f"sqlite:///{database.as_posix()}",
        runtime_mode="local",
        project_root=tmp_path / "ignored",
    )
    assert uri == f"sqlite:///{database.as_posix()}"


def test_project_version_falls_back_when_distribution_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(config_module, "version", missing_distribution)
    assert config_module._installed_project_version() == "2.0.0"
