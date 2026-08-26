"""Repository-level consistency checks for documentation and deployment configuration."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).parents[2]


def test_relative_documentation_links_resolve() -> None:
    documents = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    pattern = re.compile(r"\[[^]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)")
    missing: list[str] = []
    for document in documents:
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_repository_ignores_runtime_state_and_local_mlflow() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".local/",
        ".jupyter/",
        ".jupyter-data/",
        ".ipython/",
        "mlflow.db",
        "mlruns/",
        "coverage.json",
    ):
        assert pattern in ignore


def test_repository_separates_package_notebooks_quality_and_operations() -> None:
    expected = (
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "src/iris_mlflow_utils",
        ROOT / "quality/ruff.toml",
        ROOT / "quality/mypy.ini",
        ROOT / "notebooks/training/random_forest.ipynb",
        ROOT / "notebooks/training/xgboost.ipynb",
        ROOT / "notebooks/serving/test_endpoint.ipynb",
        ROOT / "notebooks/serving/endpoint_client.ipynb",
        ROOT / "ops/databricks/preflight.py",
        ROOT / "ops/databricks/bootstrap_permissions.ps1",
    )
    assert all(path.exists() for path in expected)
    assert not (ROOT / "tools/pyproject.toml").exists()
    assert not (ROOT / "tools/src").exists()


def test_python_version_and_default_development_groups_are_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert project["tool"]["uv"]["default-groups"] == ["test", "quality"]
    assert project["tool"]["uv"]["override-dependencies"] == ["cryptography>=50,<51"]


def test_cryptography_lock_uses_the_patched_major_version() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    cryptography = next(package for package in lock["package"] if package["name"] == "cryptography")
    assert cryptography["version"].startswith("50.")
