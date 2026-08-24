# Calidad del código

Esta carpeta centraliza las políticas y automatizaciones de calidad que no forman
parte del paquete distribuible `iris_mlflow_utils`.

- `ruff.toml`: reglas de lint y formato.
- `mypy.ini`: política de tipado estático estricto.
- `check_coverage.py`: gate independiente para cobertura de sentencias y ramas.
- `smoke_wheel.py`: instalación e importación del wheel en un entorno aislado.

Los ejecutables están en el grupo de dependencias `quality` de `pyproject.toml`.
Así comparten el entorno y las dependencias tipadas del paquete sin crear un
segundo proyecto ni otro lockfile.

Desde la raíz del repositorio:

```powershell
uv run --group quality ruff check --config quality/ruff.toml src tests quality ops
uv run --group quality ruff format --check --config quality/ruff.toml src tests quality ops
uv run --group test --group quality mypy --config-file quality/mypy.ini src tests quality ops
uv build
uv run --no-sync python quality/smoke_wheel.py --python 3.12
```
