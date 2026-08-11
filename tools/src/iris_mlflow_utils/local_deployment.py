"""Local-only promotion and deployment simulation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write an auditable local deployment manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def approve_locally(
    registry_client: Any,
    *,
    model_name: str,
    model_version: str,
    approval_tag: str = "Approval_Check",
) -> dict[str, str]:
    """Apply the same approval tag used by Databricks in local mode."""

    registry_client.set_model_version_tag(model_name, str(model_version), approval_tag, "Approved")
    registry_client.set_model_version_tag(
        model_name, str(model_version), "approval_status", "approved"
    )
    return {approval_tag: "Approved", "approval_status": "approved"}


def simulate_local_deployment(
    registry_client: Any,
    *,
    model_name: str,
    model_version: str,
    champion_alias: str,
    manifest_path: Path,
    smoke_test_passed: bool,
) -> dict[str, Any]:
    """Promote a locally validated model and emit a deployment manifest."""

    if not smoke_test_passed:
        raise RuntimeError("El smoke test local falló; Champion no fue actualizado.")
    registry_client.set_registered_model_alias(
        name=model_name,
        alias=champion_alias,
        version=str(model_version),
    )
    payload = {
        "runtime": "local",
        "model_name": model_name,
        "model_version": str(model_version),
        "status": "validated",
        "deployment_skipped": True,
        "smoke_test": "passed",
        "champion_alias": champion_alias,
    }
    write_manifest(manifest_path, payload)
    return payload
