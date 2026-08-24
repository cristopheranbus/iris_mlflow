# Desarrollo local

El proyecto Python vive en la raíz y utiliza el layout `src`. El código
distribuible está en `src/iris_mlflow_utils`; pruebas, calidad, notebooks y
operación se mantienen fuera del paquete.

## Dependencias por responsabilidad

- `test`: pytest, cobertura y dependencias de los contratos.
- `quality`: Ruff, mypy, stubs y auditoría de dependencias.
- `notebooks`: Jupyter para ejecución interactiva.

Instala solamente lo necesario:

```powershell
uv sync --group test
uv sync --group quality
uv sync --group notebooks
```

Para preparar un entorno de desarrollo completo:

```powershell
uv sync --all-groups
```

Python está fijado en 3.12 mediante `.python-version` y el rango del proyecto.
Los grupos `test` y `quality` se instalan por defecto; `notebooks` continúa
siendo opcional.

## Validaciones

```powershell
uv run --group test pytest
uv run --group quality ruff check --config quality/ruff.toml src tests quality ops
uv run --group quality ruff format --check --config quality/ruff.toml src tests quality ops
uv run --group test --group quality mypy --config-file quality/mypy.ini src tests quality ops
uv build
uv run --no-sync python quality/smoke_wheel.py --python 3.12
```

Los artefactos locales de pruebas, cobertura, MLflow y despliegue se escriben
bajo `.local/`, que no se versiona.

## Límites de carpetas

- `src/`: API y lógica reutilizable incluida en el wheel.
- `tests/`: pruebas unitarias, de integración local y contratos.
- `quality/`: políticas y automatizaciones de calidad.
- `notebooks/`: entrenamiento y clientes interactivos.
- `ops/databricks/`: verificaciones y bootstrap operacional.
- `deployment/`: notebooks productivos administrados por el bundle.
