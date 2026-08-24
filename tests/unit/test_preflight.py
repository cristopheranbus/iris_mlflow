"""Unit tests for the Databricks deployment preflight."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from ops.databricks import preflight

pytestmark = pytest.mark.unit


def test_run_cli_builds_json_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        captured.append(command)
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return SimpleNamespace(returncode=0, stdout='{"active": true}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert preflight.run_cli("current-user", "me") == {"active": True}
    assert captured == [["databricks", "current-user", "me", "--output", "json"]]


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("", "permission denied", "permission denied"),
        ("workspace cancelled or is not active", "", "suspendida o inactiva"),
    ],
)
def test_run_cli_translates_failures(
    monkeypatch: pytest.MonkeyPatch, stdout: str, stderr: str, expected: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr),
    )

    with pytest.raises(RuntimeError, match=expected):
        preflight.run_cli("tables", "get", "catalog.schema.table")


def test_run_cli_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )

    with pytest.raises(json.JSONDecodeError):
        preflight.run_cli("current-user", "me")


@pytest.mark.parametrize("payload", [{}, {"privilege_assignments": []}, []])
def test_require_effective_grants_rejects_missing_assignments(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    monkeypatch.setattr(preflight, "run_cli", lambda *args: payload)

    with pytest.raises(RuntimeError, match="no tiene grants efectivos"):
        preflight.require_effective_grants("TABLE", "catalog.schema.table")


def test_require_effective_grants_accepts_assignments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight,
        "run_cli",
        lambda *args: {"privilege_assignments": [{"principal": "service-principal"}]},
    )

    preflight.require_effective_grants("TABLE", "catalog.schema.table")


def test_main_validates_identity_resources_and_grants(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run_cli(*arguments: str) -> object:
        calls.append(arguments)
        if arguments == ("current-user", "me"):
            return {"application_id": "client-123"}
        if arguments[0] == "grants":
            return {"privilege_assignments": [{"principal": "client-123"}]}
        return {}

    monkeypatch.setattr(preflight, "run_cli", fake_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "--model-name",
            "catalog.schema.model",
            "--feature-table",
            "catalog.schema.features",
            "--expected-client-id",
            "client-123",
        ],
    )

    preflight.main()

    assert calls == [
        ("current-user", "me"),
        ("tables", "get", "catalog.schema.features"),
        ("registered-models", "get", "catalog.schema.model"),
        ("grants", "get-effective", "TABLE", "catalog.schema.features"),
        ("grants", "get-effective", "REGISTERED_MODEL", "catalog.schema.model"),
    ]
    assert json.loads(capsys.readouterr().out) == {
        "status": "passed",
        "model": "catalog.schema.model",
        "table": "catalog.schema.features",
    }


def test_main_stops_when_oidc_identity_does_not_match(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run_cli(*arguments: str) -> object:
        calls.append(arguments)
        return {"application_id": "unexpected-client"}

    monkeypatch.setattr(preflight, "run_cli", fake_run_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight.py",
            "--model-name",
            "catalog.schema.model",
            "--feature-table",
            "catalog.schema.features",
            "--expected-client-id",
            "client-123",
        ],
    )

    with pytest.raises(RuntimeError, match="no coincide"):
        preflight.main()
    assert calls == [("current-user", "me")]
