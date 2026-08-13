# iris_mlflow_utils

Paquete reutilizable para los notebooks Iris.

## Módulos

- `config.py`: carga `config/training.toml`, entorno y overrides opcionales.
- `constants.py`: contrato de columnas y tipos.
- `data.py`: validación y carga directa desde tablas Spark/Delta.
- `evaluation.py`: métricas y tablas de evaluación.
- `feature_table.py`: bootstrap explícito de tablas Unity Catalog.
- `registry.py`: tags, comentarios, aliases y verificación.
- `deployment.py`: gates, create/update de endpoints y rollback transaccional.
- `local_deployment.py`: aprobación y despliegue local simulado.
- `runtime.py`: detección local/Databricks y parámetros dinámicos.
- `serving.py`: cliente REST testeable para Model Serving.

`scripts/databricks_preflight.py` valida identidad y permisos antes del bundle.
`scripts/bootstrap_databricks_permissions.ps1` aplica grants de Unity Catalog.

## Pruebas

Las pruebas viven en `../tests` para mantener una única suite en la raíz.

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

El paquete conserva las APIs públicas usadas por los notebooks y no requiere
Databricks para sus pruebas unitarias.
