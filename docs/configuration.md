# Configuración

## Perfiles de runtime

`config/training.toml` separa la configuración local de la productiva:

```toml
[runtime.local]
dataset_path = "data/local/iris_features.csv"
artifact_location = ".local/mlflow/artifacts"
tracking_uri = "sqlite:///.local/mlflow/mlflow.db"
registry_uri = "sqlite:///.local/mlflow/mlflow.db"
registered_model_name = "iris_classifier"

[runtime.databricks]
feature_table = "workspace.default.iris_features"
registered_model_name = "workspace.default.iris_classifier"
registry_uri = "databricks-uc"
```

La URI local se canoniza a la raíz del proyecto, así que los notebooks y la UI
usan la misma base aunque se ejecuten desde otro directorio.

La precedencia del runtime es `IRIS_RUNTIME`, luego `dbutils`, luego
`DATABRICKS_RUNTIME_VERSION` y finalmente `local`. Variables locales
permitidas: `IRIS_LOCAL_DATASET_PATH`, `MLFLOW_TRACKING_URI`,
`MLFLOW_REGISTRY_URI`, `IRIS_REGISTERED_MODEL_NAME` e
`IRIS_LOCAL_AUTO_APPROVE`. En producción sólo se lee la tabla Delta configurada;
no se debe guardar ningún secreto en TOML ni en el repositorio.

La configuración versionada vive en `config/training.toml`. Las variables de
entorno tienen prioridad sobre TOML. Los widgets sólo se leen cuando se
habilitan explícitamente y no contienen valores estáticos del flujo.

## Deployment

La sección `[deployment]` define:

- `model_name` y `endpoint_name`.
- `min_test_f1_weighted` y `min_test_accuracy`.
- `max_metric_regression` contra Champion.
- `required_approval_tag`.
- Aliases Champion y Challenger.

Los thresholds históricos se conservan temporalmente por compatibilidad. Las
decisiones nuevas usan los perfiles descritos en
[`promotion-policy.md`](promotion-policy.md). El monitoreo operacional usa
[`config/monitoring.toml`](../config/monitoring.toml).
- Timeout y polling de Model Serving.
- El nombre del job y el endpoint por ambiente pertenecen al bundle.

Los parámetros dinámicos son `model_name` y `model_version`. `endpoint_name`
es un valor estático inyectado por el target del bundle para aislar ambientes.

Overrides permitidos:

```text
IRIS_DEPLOYMENT_MODEL_NAME
IRIS_SERVING_ENDPOINT_NAME
IRIS_MIN_TEST_F1_WEIGHTED
IRIS_MIN_TEST_ACCURACY
IRIS_MAX_METRIC_REGRESSION
IRIS_REQUIRED_APPROVAL_TAG
```

La identidad de producción y el grupo operador se inyectan como variables del
bundle (`BUNDLE_VAR_production_service_principal` y
`BUNDLE_VAR_operators_group`). No se fija un cluster: las tareas usan serverless.

Tokens y credenciales deben resolverse con Databricks Secrets o la identidad
administrada del job. Nunca se guardan en TOML, notebooks, tags o mensajes de
error.

En modo local, MLflow usa `sqlite:///.local/mlflow/mlflow.db` para tracking y registry. Esto
evita el backend filesystem legado de MLflow y conserva runs, artefactos,
versiones y aliases en un único almacén local.
