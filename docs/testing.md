# Pruebas

Desde la raíz del proyecto:

```powershell
uv run --project tools pytest
uv run --project tools ruff check tools/src tests tools/scripts
uv run --project tools ruff format --check tools/src tests tools/scripts
uv run --project tools mypy
uv run --project tools pip-audit
uv build --project tools
```

La suite exige al menos 85% de cobertura. Las pruebas unitarias y las ejecuciones
locales de MLflow no requieren Databricks. La validación cloud usa cómputo serverless
y requiere cuenta activa, OIDC, modelos y permisos; el proyecto no configura clusters.

GitHub ejecuta además un escaneo de secretos sobre el historial completo. Los notebooks
deben conservar `execution_count=null` y no guardar outputs.
