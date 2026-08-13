# Iris MLflow

Proyecto reproducible para entrenar clasificadores Iris, evaluarlos con MLflow
y promoverlos de forma controlada a Model Serving.

## Flujo

```text
workspace.default.iris_features@version -> entrenamiento -> Challenger
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

### Entrenamiento e inferencia

| Componente | Funcionalidad |
|---|---|
| [`random_forest.ipynb`](random_forest.ipynb) | Entrena Random Forest, registra métricas y artefactos, publica una versión y asigna `Challenger`. |
| [`xgboost.ipynb`](xgboost.ipynb) | Ejecuta el mismo contrato de entrenamiento y registro utilizando XGBoost. |
| [`test_endpoint.ipynb`](test_endpoint.ipynb) | Prueba manualmente el endpoint desplegado con datos Iris y revisa su respuesta. |
| [`tools/databricks_endpoint_client.ipynb`](tools/databricks_endpoint_client.ipynb) | Cliente interactivo reutilizable para invocar un endpoint de Databricks Model Serving. |

### Evaluación, aprobación y despliegue

| Componente | Funcionalidad |
|---|---|
| [`deployment/evaluate_model.ipynb`](deployment/evaluate_model.ipynb) | Carga una versión exacta, genera artefactos, aplica thresholds y la compara contra `Champion`. |
| [`deployment/approval.ipynb`](deployment/approval.ipynb) | Comprueba la aprobación manual `Approval_Check=Approved` y detiene el flujo si falta. |
| [`deployment/deploy_model.ipynb`](deployment/deploy_model.ipynb) | Actualiza Model Serving, espera `READY`, ejecuta el smoke test y promueve `Champion`. |
| [`deployment/create_deployment_job.ipynb`](deployment/create_deployment_job.ipynb) | Conecta el Job administrado por el bundle con el modelo registrado mediante `deployment_job_id`. |

### Librería Python compartida

| Componente | Funcionalidad |
|---|---|
| [`tools/src/iris_mlflow_utils/config.py`](tools/src/iris_mlflow_utils/config.py) | Lee, valida y resuelve la configuración local o Databricks. |
| [`tools/src/iris_mlflow_utils/constants.py`](tools/src/iris_mlflow_utils/constants.py) | Centraliza nombres de columnas, clases y valores compartidos. |
| [`tools/src/iris_mlflow_utils/runtime.py`](tools/src/iris_mlflow_utils/runtime.py) | Detecta el runtime y obtiene parámetros sin depender directamente de `dbutils` en local. |
| [`tools/src/iris_mlflow_utils/data.py`](tools/src/iris_mlflow_utils/data.py) | Carga y valida el dataset correspondiente a cada runtime. |
| [`tools/src/iris_mlflow_utils/feature_table.py`](tools/src/iris_mlflow_utils/feature_table.py) | Contiene utilidades explícitas de preparación de la tabla de features fuera del flujo normal. |
| [`tools/src/iris_mlflow_utils/evaluation.py`](tools/src/iris_mlflow_utils/evaluation.py) | Calcula métricas y genera matrices, ROC, Precision-Recall, lift, gain y demás artefactos. |
| [`tools/src/iris_mlflow_utils/registry.py`](tools/src/iris_mlflow_utils/registry.py) | Gestiona MLflow Registry, versiones, tags, descripciones y aliases. |
| [`tools/src/iris_mlflow_utils/deployment.py`](tools/src/iris_mlflow_utils/deployment.py) | Implementa gates, create/update del endpoint, rollback y promoción posterior al smoke test. |
| [`tools/src/iris_mlflow_utils/serving.py`](tools/src/iris_mlflow_utils/serving.py) | Construye payloads, administra el endpoint y ejecuta verificaciones de inferencia. |
| [`tools/src/iris_mlflow_utils/local_deployment.py`](tools/src/iris_mlflow_utils/local_deployment.py) | Simula aprobación y despliegue local, y genera manifiestos auditables. |
| [`tools/src/iris_mlflow_utils/__init__.py`](tools/src/iris_mlflow_utils/__init__.py) | Expone la interfaz pública del paquete compartido. |

### Configuración, datos y empaquetado

| Componente | Funcionalidad |
|---|---|
| [`config/training.toml`](config/training.toml) | Fuente versionada de parámetros de entrenamiento, runtime, MLflow y deployment. |
| [`config/local.env.example`](config/local.env.example) | Plantilla de variables permitidas para una ejecución local. |
| [`data/local/iris_features.csv`](data/local/iris_features.csv) | Dataset de desarrollo usado exclusivamente en modo local. |
| [`tools/pyproject.toml`](tools/pyproject.toml) | Define el paquete, dependencias y configuración de Pytest, Ruff y MyPy. |
| [`tools/uv.lock`](tools/uv.lock) | Fija versiones reproducibles de las dependencias Python. |

### Infraestructura y automatización

| Componente | Funcionalidad |
|---|---|
| [`databricks.yml`](databricks.yml) | Declara wheel, notebooks, Jobs, environments serverless, permisos y targets `dev`/`prod`. |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Ejecuta tests, lint, formato, tipos y build en GitHub Actions. |
| [`.github/workflows/databricks-bundle.yml`](.github/workflows/databricks-bundle.yml) | Valida el bundle y despliega a Databricks cuando el interruptor operativo está habilitado. |
| [`.github/workflows/security.yml`](.github/workflows/security.yml) | Escanea el historial para impedir la publicación de secretos. |
| [`tools/scripts/databricks_preflight.py`](tools/scripts/databricks_preflight.py) | Verifica cuenta, identidad, tabla, modelo y grants antes del despliegue. |
| [`tools/scripts/bootstrap_databricks_permissions.ps1`](tools/scripts/bootstrap_databricks_permissions.ps1) | Aplica permisos mínimos de Unity Catalog con una identidad administradora. |

### Pruebas automatizadas

| Componente | Funcionalidad |
|---|---|
| [`tests/test_training_utils.py`](tests/test_training_utils.py) | Valida datos, configuración, evaluación y utilidades usadas durante el entrenamiento. |
| [`tests/test_deployment.py`](tests/test_deployment.py) | Valida runtime dual, gates, artefactos y simulación del despliegue local. |
| [`tests/test_registry.py`](tests/test_registry.py) | Comprueba tags, descripciones, versiones y aliases de MLflow Registry. |
| [`tests/test_serving.py`](tests/test_serving.py) | Comprueba payloads, respuestas y promoción posterior al smoke test. |
| [`tests/conftest.py`](tests/conftest.py) | Proporciona fixtures y configuración compartida por la suite. |
| [`tests/fixtures`](tests/fixtures) | Reserva datos pequeños y estables para pruebas que no deben depender del CSV de desarrollo. |
| [`tests/unit`](tests/unit) | Documenta el alcance de las pruebas unitarias. |
| [`tests/integration`](tests/integration) | Documenta las validaciones que requieren servicios o infraestructura externa. |

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
5. Confirmar que `iris-classifier-dev` o `iris-classifier` quedó `READY`.
6. Validar que la versión fue promovida a `Champion`.

El job se crea con `databricks bundle deploy`. Después se ejecuta una vez
`connect_deployment_job` para asociarlo al modelo. Desarrollo y producción usan
modelos, jobs y endpoints separados. Producción exige `main`, ejecución manual,
service principal y aprobación del environment de GitHub. Consulta
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
