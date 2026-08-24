"""Install the built wheel in an isolated environment and verify its public import."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path


def _interpreter(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _project_identity(project_file: Path) -> tuple[str, str]:
    payload = tomllib.loads(project_file.read_text(encoding="utf-8"))
    project = payload["project"]
    return str(project["name"]), str(project["version"])


def _find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Se esperaba exactamente un wheel en {dist_dir}; encontrados: {wheels}")
    return wheels[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--python", default="3.12")
    args = parser.parse_args()

    project_name, expected_version = _project_identity(Path("pyproject.toml"))
    wheel = _find_wheel(args.dist_dir)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv no está disponible para crear el entorno aislado.")

    local_root = Path(".local")
    local_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheel-smoke-", dir=local_root) as temporary:
        venv = Path(temporary) / "venv"
        subprocess.run([uv, "venv", "--python", args.python, str(venv)], check=True)
        python = _interpreter(venv)
        subprocess.run([uv, "pip", "install", "--python", str(python), str(wheel)], check=True)
        smoke = (
            "import sys\n"
            "from importlib.metadata import version\n"
            "import iris_mlflow_utils as package\n"
            f"assert sys.version_info[:2] == ({args.python.replace('.', ', ')})\n"
            f"assert version({project_name!r}) == {expected_version!r}\n"
            "assert callable(package.build_config)\n"
            "assert callable(package.evaluate_promotion_gate)\n"
        )
        subprocess.run([str(python), "-c", smoke], check=True)

    print(f"Wheel smoke test passed: {wheel.name} on Python {args.python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
