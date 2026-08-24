"""Enforce independent statement and branch coverage thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percentage(covered: int, total: int) -> float:
    return 100.0 if total == 0 else covered * 100.0 / total


def evaluate_coverage(
    totals: dict[str, Any], *, min_statements: float, min_branches: float
) -> tuple[float, float, list[str]]:
    """Return measured percentages and human-readable policy failures."""

    statement_percent = _percentage(int(totals["covered_lines"]), int(totals["num_statements"]))
    branch_percent = _percentage(int(totals["covered_branches"]), int(totals["num_branches"]))
    failures = []
    if statement_percent < min_statements:
        failures.append(
            f"Statement coverage {statement_percent:.2f}% is below {min_statements:.2f}%."
        )
    if branch_percent < min_branches:
        failures.append(f"Branch coverage {branch_percent:.2f}% is below {min_branches:.2f}%.")
    return statement_percent, branch_percent, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-statements", type=float, default=90.0)
    parser.add_argument("--min-branches", type=float, default=85.0)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    statements, branches, failures = evaluate_coverage(
        payload["totals"],
        min_statements=args.min_statements,
        min_branches=args.min_branches,
    )
    print(f"Statement coverage: {statements:.2f}%")
    print(f"Branch coverage: {branches:.2f}%")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
