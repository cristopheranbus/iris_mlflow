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

- [`random_forest.ipynb`](random_forest.ipynb): entrenamiento Random Forest.
- [`xgboost.ipynb`](xgboost.ipynb): entrenamiento XGBoost.
- [`deployment/evaluate_model.ipynb`](deployment/evaluate_model.ipynb): evaluación, gates y artefactos.
- [`deployment/approval.ipynb`](deployment/approval.ipynb): aprobación mediante tag de Unity Catalog.
- [`deployment/deploy_model.ipynb`](deployment/deploy_model.ipynb): Model Serving, smoke test y Champion.
- [`deployment/create_deployment_job.ipynb`](deployment/create_deployment_job.ipynb): conexión del Job del bundle con el modelo.
- [`databricks.yml`](databricks.yml): infraestructura declarativa, Jobs, permisos y targets.
- [`tools/src/iris_mlflow_utils`](tools/src/iris_mlflow_utils): configuración, evaluación, registry y serving.
- [`config/training.toml`](config/training.toml): configuración versionada sin secretos.
- [`tests`](tests): pruebas unitarias y de integración local.

## Guía de documentación

| Documento | Contenido |
|---|---|
| [Arquitectura](docs/architecture.md) | Flujo completo, responsabilidades, ramas local/Databricks y lugares de ejecución. |
| [Infraestructura y CI/CD](docs/infrastructure.md) | Bundle, OIDC, GitHub Actions, targets, service principals y suspensión temporal del despliegue. |
| [Notebooks de entrenamiento](docs/training_notebooks.md) | Contrato y ejecución de Random Forest y XGBoost, parámetros, métricas y publicación en MLflow. |
| [Configuración](docs/configuration.md) | `training.toml`, perfiles de runtime, variables de entorno y configuración de deployment. |
| [Ejecución local](docs/local_execution.md) | Instalación, MLflow local, entrenamiento y simulación de aprobación y despliegue. |
| [Evaluación del modelo](docs/model_evaluation.md) | Métricas, matriz de confusión, ROC, Precision-Recall, lift, gain y criterios de aceptación. |
| [Operaciones](docs/operations.md) | Promoción, aprobación, Model Serving, diagnóstico y ejecución diaria. |
| [Checklist de publicación](docs/release-checklist.md) | Controles requeridos antes y después de promover una versión. |
| [Rollback](docs/rollback.md) | Restauración del endpoint y del alias `Champion`. |
| [Pruebas](docs/testing.md) | Suite automatizada, calidad estática y comandos de validación. |

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
[Infraestructura y CI/CD](docs/infrastructure.md).

## Calidad

Desde `tools` ejecuta:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

La validación real de Unity Catalog y Model Serving requiere una cuenta
Databricks activa, permisos y credenciales configuradas. Empieza por la
[guía de arquitectura](docs/architecture.md) y continúa con el
[manual de operaciones](docs/operations.md).
