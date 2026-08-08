# iris_mlflow_utils

Paquete reutilizable para los notebooks Iris.

## Módulos

- `config.py`: carga `config/training.toml`, entorno y overrides opcionales.
- `constants.py`: contrato de columnas y tipos.
- `data.py`: validación y carga directa desde tablas Spark/Delta.
- `evaluation.py`: métricas y tablas de evaluación.
- `feature_table.py`: bootstrap explícito de tablas Unity Catalog.
- `registry.py`: tags, comentarios, aliases y verificación.
- `serving.py`: cliente REST testeable para Model Serving.

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
