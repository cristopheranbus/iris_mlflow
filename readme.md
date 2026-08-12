# Iris MLflow

Proyecto reproducible para entrenar clasificadores Iris, evaluarlos con MLflow
y promoverlos de forma controlada a Model Serving.

## Flujo

```text
workspace.default.iris_features -> entrenamiento -> Challenger
                                      -> evaluación y artefactos
                                      -> aprobación manual
                                      -> Model Serving -> Champion
```

Los notebooks de entrenamiento leen directamente la tabla Delta
`workspace.default.iris_features`. No leen ni crean CSV durante el flujo
normal.

## Ejecución local y Databricks

Los mismos notebooks detectan el entorno automáticamente. Para probar sin
Spark ni Databricks, usa el CSV de desarrollo y MLflow local:

```powershell
$env:IRIS_RUNTIME = "local"
cd tools
uv run jupyter notebook ..\random_forest.ipynb
```

El modo local usa `data/local/iris_features.csv`, SQLite (`mlflow.db`) y
simula la aprobación, el smoke test y la promoción de `Champion`. El resultado
queda en `artifacts/local_deployment_manifest.json`; no crea un endpoint HTTP.

En Databricks se conserva el flujo productivo con Delta, Unity Catalog,
Deployment Jobs y Model Serving. Puede forzarse explícitamente con
`$env:IRIS_RUNTIME = "databricks"`. La existencia del CSV local nunca cambia
la fuente productiva.

## Componentes

- `random_forest.ipynb`: entrenamiento Random Forest.
- `xgboost.ipynb`: entrenamiento XGBoost.
- `docs/training_notebooks.md`: contrato, ejecución y verificación de ambos notebooks.
- `deployment/evaluate_model.ipynb`: evaluación, gates y artefactos.
- `deployment/approval.ipynb`: aprobación mediante tag de Unity Catalog.
- `deployment/deploy_model.ipynb`: Model Serving, smoke test y Champion.
- `databricks.yml`: infraestructura declarativa, jobs, permisos y targets.
- `deployment/create_deployment_job.ipynb`: conexión del job del bundle con el modelo.
- `tools/src/iris_mlflow_utils`: configuración, evaluación, registry y serving.
- `config/training.toml`: configuración versionada sin secretos.
- `tests/`: pruebas unitarias y de integración local.
- `docs/`: arquitectura, evaluación, operación, release y rollback.

## Operación rápida

1. Ejecutar Random Forest o XGBoost.
2. Revisar la versión `Challenger`.
3. Revisar métricas y artefactos.
4. Aprobar aplicando `Approval_Check=Approved`.
5. Confirmar que el endpoint `iris-classifier` quedó `READY`.
6. Validar que la versión fue promovida a `Champion`.

El job se crea con `databricks bundle deploy`. Después se ejecuta una vez
`connect_deployment_job` para asociarlo al modelo. Producción exige un service
principal y aprobación del environment de GitHub. Consulta
`docs/infrastructure.md`.

## Calidad

Desde `tools` ejecuta:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

La validación real de Unity Catalog y Model Serving requiere cluster, permisos
y secretos configurados. Consulta `docs/` para la operación completa.
