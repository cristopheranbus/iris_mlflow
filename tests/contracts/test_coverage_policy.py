"""Contract tests for the independent coverage gate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


def run_gate(
    repository_root: Path, tmp_path: Path, totals: dict[str, int]
) -> subprocess.CompletedProcess[str]:
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"totals": totals}), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(repository_root / "quality" / "check_coverage.py"),
            str(report),
            "--min-statements",
            "90",
            "--min-branches",
            "85",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_coverage_gate_accepts_both_thresholds(repository_root: Path, tmp_path: Path) -> None:
    result = run_gate(
        repository_root,
        tmp_path,
        {"covered_lines": 90, "num_statements": 100, "covered_branches": 85, "num_branches": 100},
    )
    assert result.returncode == 0


def test_coverage_gate_reports_each_failed_dimension(repository_root: Path, tmp_path: Path) -> None:
    result = run_gate(
        repository_root,
        tmp_path,
        {"covered_lines": 89, "num_statements": 100, "covered_branches": 84, "num_branches": 100},
    )
    assert result.returncode == 1
    assert "Statement coverage" in result.stdout
    assert "Branch coverage" in result.stdout
