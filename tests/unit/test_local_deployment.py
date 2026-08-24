"""Unit tests for local approval and deployment simulation."""

import json
from pathlib import Path

import pytest

from iris_mlflow_utils.local_deployment import (
    approve_locally,
    simulate_local_deployment,
    write_manifest,
)

pytestmark = pytest.mark.unit


class Registry:
    def __init__(self, previous: str | None = None) -> None:
        self.previous = previous
        self.tags: dict[tuple[str, str], str] = {}
        self.aliases: dict[str, str] = {}

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags[(version, key)] = value

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        if self.previous is None:
            raise LookupError(alias)
        return type("Version", (), {"version": self.previous})()


def test_write_manifest_creates_parent_directory(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "nested" / "manifest.json", {"status": "ok"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok"}


def test_approve_locally_applies_governance_tags() -> None:
    registry = Registry()
    result = approve_locally(registry, model_name="iris", model_version="2")
    assert result == {"Approval_Check": "Approved", "approval_status": "approved"}
    assert registry.tags[("2", "approval_status")] == "approved"


def test_simulate_local_deployment_rejects_failed_smoke_test(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="smoke test"):
        simulate_local_deployment(
            Registry(),
            model_name="iris",
            model_version="2",
            champion_alias="Champion",
            manifest_path=tmp_path / "manifest.json",
            smoke_test_passed=False,
        )


@pytest.mark.parametrize("previous, rollback", [(None, False), ("1", True), ("2", False)])
def test_simulate_local_deployment_promotes_and_records_manifest(
    tmp_path: Path, previous: str | None, rollback: bool
) -> None:
    registry = Registry(previous)
    path = tmp_path / "manifest.json"
    payload = simulate_local_deployment(
        registry,
        model_name="iris",
        model_version="2",
        champion_alias="Champion",
        manifest_path=path,
        smoke_test_passed=True,
    )
    assert payload["rollback_available"] is rollback
    assert payload["promoted_at"]
    assert registry.aliases["Champion"] == "2"
    assert json.loads(path.read_text(encoding="utf-8"))["smoke_test"] == "passed"
