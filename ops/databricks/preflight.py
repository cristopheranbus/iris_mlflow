"""Fail-fast validation for a Databricks bundle deployment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any


def run_cli(*arguments: str) -> Any:
    """Run the authenticated Databricks CLI and decode its JSON response."""

    command = ["databricks", *arguments, "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "cancelled or is not active" in detail.lower():
            raise RuntimeError("La organización Databricks está suspendida o inactiva.")
        raise RuntimeError(f"Falló {' '.join(command[:-2])}: {detail}")
    return json.loads(result.stdout or "{}")


def require_effective_grants(securable_type: str, full_name: str) -> None:
    """Require at least one effective grant for the active OIDC identity."""

    grants = run_cli("grants", "get-effective", securable_type, full_name)
    assignments = grants.get("privilege_assignments", []) if isinstance(grants, dict) else []
    if not assignments:
        raise RuntimeError(f"La identidad OIDC no tiene grants efectivos sobre {full_name}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--expected-client-id", default=os.getenv("DATABRICKS_CLIENT_ID", ""))
    args = parser.parse_args()

    identity = run_cli("current-user", "me")
    if args.expected_client_id and args.expected_client_id not in json.dumps(identity):
        raise RuntimeError("La identidad autenticada no coincide con DATABRICKS_CLIENT_ID.")
    run_cli("tables", "get", args.feature_table)
    run_cli("registered-models", "get", args.model_name)
    require_effective_grants("TABLE", args.feature_table)
    require_effective_grants("REGISTERED_MODEL", args.model_name)
    print(json.dumps({"status": "passed", "model": args.model_name, "table": args.feature_table}))


if __name__ == "__main__":
    main()
