"""Structural contracts for ordered and descriptive GitHub Actions workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.contract

WORKFLOW_NAMES = {
    "01-code-quality.yml": "01 · Code quality and package validation",
    "02-security-scanning.yml": "02 · Repository security scanning",
    "03-databricks-deployment.yml": "03 · Databricks bundle validation and deployment",
}


def _load(path: Path) -> dict[object, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[object, Any], payload)


def _triggers(workflow: dict[object, Any]) -> dict[str, Any]:
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return cast(dict[str, Any], triggers)


def test_workflow_files_are_exclusively_ordered_and_descriptive(repository_root: Path) -> None:
    workflow_directory = repository_root / ".github" / "workflows"
    paths = sorted(workflow_directory.glob("*.yml"))

    assert [path.name for path in paths] == list(WORKFLOW_NAMES)
    assert {path.name: _load(path)["name"] for path in paths} == WORKFLOW_NAMES
    for legacy in ("ci.yml", "security.yml", "databricks-bundle.yml"):
        assert not (workflow_directory / legacy).exists()


def test_code_quality_workflow_orders_validation_before_build(repository_root: Path) -> None:
    workflow = _load(repository_root / ".github/workflows/01-code-quality.yml")
    jobs = workflow["jobs"]

    assert list(jobs) == ["test-suite", "static-analysis", "dependency-audit", "package-build"]
    assert jobs["package-build"]["needs"] == [
        "test-suite",
        "static-analysis",
        "dependency-audit",
    ]
    matrix = jobs["test-suite"]["strategy"]["matrix"]["include"]
    assert {entry["os"] for entry in matrix} == {"ubuntu-latest", "windows-latest"}
    assert "quality/smoke_wheel.py" in jobs["package-build"]["steps"][-1]["run"]
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert workflow["concurrency"]["group"].startswith("code-quality-")

    triggers = _triggers(workflow)
    assert triggers["pull_request"]["branches"] == ["main", "dev"]
    assert triggers["push"]["branches"] == ["main", "dev"]


def test_security_workflow_scans_complete_history(repository_root: Path) -> None:
    workflow = _load(repository_root / ".github/workflows/02-security-scanning.yml")
    job = workflow["jobs"]["secret-history-scan"]

    assert list(workflow["jobs"]) == ["secret-history-scan"]
    assert job["steps"][0]["with"]["fetch-depth"] == 0
    assert "gitleaks/gitleaks-action" in job["steps"][1]["uses"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True


def test_databricks_workflow_preserves_oidc_and_environment_guards(
    repository_root: Path,
) -> None:
    path = repository_root / ".github/workflows/03-databricks-deployment.yml"
    workflow = _load(path)
    source = path.read_text(encoding="utf-8")
    jobs = workflow["jobs"]

    assert list(jobs) == [
        "deployment-disabled",
        "validate-bundle",
        "deploy-development",
        "deploy-production",
    ]
    assert jobs["deploy-development"]["needs"] == "validate-bundle"
    assert jobs["deploy-production"]["needs"] == "validate-bundle"
    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}
    assert workflow["concurrency"]["cancel-in-progress"] is False

    triggers = _triggers(workflow)
    assert triggers["pull_request"]["branches"] == ["main", "dev"]
    assert triggers["push"]["branches"] == ["dev"]
    assert triggers["workflow_dispatch"]["inputs"]["target"]["options"] == ["dev", "prod"]
    assert "github.ref == 'refs/heads/dev'" in jobs["deploy-development"]["if"]
    assert "github.ref == 'refs/heads/main'" in jobs["deploy-production"]["if"]
    assert "DATABRICKS_AUTH_TYPE: github-oidc" in source
    assert "DATABRICKS_CLIENT_SECRET" not in source
    assert "ops/databricks/preflight.py" in source


def test_all_external_actions_remain_pinned_to_commit_hashes(repository_root: Path) -> None:
    workflow_directory = repository_root / ".github" / "workflows"
    for path in workflow_directory.glob("*.yml"):
        workflow = _load(path)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if action := step.get("uses"):
                    assert re.search(r"@[0-9a-f]{40}$", action), (
                        f"Unpinned action in {path}: {action}"
                    )
