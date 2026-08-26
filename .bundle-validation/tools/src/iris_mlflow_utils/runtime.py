"""Runtime detection and parameter access shared by notebooks and jobs."""

from __future__ import annotations

import builtins
import os
from typing import Any, Literal

RuntimeMode = Literal["local", "databricks"]


def _get_dbutils() -> Any:
    """Return dbutils when the current process exposes it."""

    get_ipython_fn = getattr(builtins, "get_ipython", None)
    if get_ipython_fn is None:
        return None
    shell = get_ipython_fn()
    return None if shell is None else shell.user_ns.get("dbutils")


def detect_runtime() -> RuntimeMode:
    """Detect Databricks or local execution with an explicit override."""

    configured = os.getenv("IRIS_RUNTIME", "").strip().lower()
    if configured in {"local", "databricks"}:
        return configured  # type: ignore[return-value]
    if _get_dbutils() is not None or os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return "databricks"
    return "local"


def is_databricks_runtime() -> bool:
    """Return whether the active runtime is Databricks."""

    return detect_runtime() == "databricks"


def get_runtime_parameter(
    name: str,
    default: str = "",
    runtime_mode: RuntimeMode | None = None,
) -> str:
    """Read environment parameters and Databricks widgets when applicable."""

    value = os.getenv(name, "").strip()
    if value:
        return value
    mode = runtime_mode or detect_runtime()
    if mode != "databricks":
        return default
    utilities = _get_dbutils()
    if utilities is None:
        return default
    try:
        return utilities.widgets.get(name).strip() or default
    except Exception:
        return default


def get_dbutils() -> Any:
    """Expose dbutils only to Databricks-specific notebook branches."""

    return _get_dbutils()
