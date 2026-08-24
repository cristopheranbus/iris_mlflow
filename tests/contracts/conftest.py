"""Helpers for repository contract tests."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).parents[2]


@pytest.fixture
def notebook_source() -> Callable[[Path], str]:
    def load(path: Path) -> str:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )

    return load
