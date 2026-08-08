# Pruebas

Desde `tools`:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

Las pruebas unitarias no requieren Databricks. Las pruebas de integración
locales pueden usar MLflow local. La ejecución real de notebooks sigue siendo
una validación de integración dependiente de cluster, permisos y secretos.
