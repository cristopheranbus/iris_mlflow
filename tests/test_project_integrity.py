"""Repository-level consistency checks for documentation and deployment configuration."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_relative_documentation_links_resolve() -> None:
    documents = [ROOT / "readme.md", *(ROOT / "docs").glob("*.md")]
    pattern = re.compile(r"\[[^]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)")
    missing: list[str] = []
    for document in documents:
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_bundle_isolates_dev_and_prod_and_keeps_deployment_disabled_safe() -> None:
    bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/databricks-bundle.yml").read_text(encoding="utf-8")

    assert "workspace.default.iris_classifier_dev" in bundle
    assert "workspace.default.iris_classifier" in bundle
    assert "iris-classifier-dev" in bundle
    assert "iris-classifier" in bundle
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "DATABRICKS_DEPLOY_ENABLED != 'true'" in workflow
    assert "databricks_preflight.py" in workflow


def test_repository_ignores_runtime_state_and_local_mlflow() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".jupyter/", ".jupyter-data/", ".ipython/", "mlflow.db", "mlruns/"):
        assert pattern in ignore
