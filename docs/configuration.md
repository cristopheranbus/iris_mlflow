# Configuración

## Perfiles de runtime

`config/training.toml` separa la configuración local de la productiva:

```toml
[runtime]
mode = "auto"

[runtime.local]
dataset_path = "data/local/iris_features.csv"
tracking_uri = "sqlite:///mlflow.db"
registry_uri = "sqlite:///mlflow.db"
registered_model_name = "iris_classifier"

[runtime.databricks]
feature_table = "workspace.default.iris_features"
registered_model_name = "workspace.default.iris_classifier"
registry_uri = "databricks-uc"
```

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
- Timeout y polling de Model Serving.
- Nombre y ruta raíz del Deployment Job.

Los únicos parámetros dinámicos del job son `model_name` y `model_version`.

Overrides permitidos:

```text
IRIS_DEPLOYMENT_MODEL_NAME
IRIS_SERVING_ENDPOINT_NAME
IRIS_MIN_TEST_F1_WEIGHTED
IRIS_MIN_TEST_ACCURACY
IRIS_MAX_METRIC_REGRESSION
IRIS_REQUIRED_APPROVAL_TAG
IRIS_DEPLOYMENT_CLUSTER_ID
IRIS_DEPLOYMENT_SERVICE_PRINCIPAL
IRIS_DEPLOYMENT_NOTEBOOK_ROOT
IRIS_DEPLOYMENT_JOB_NAME
```

Tokens y credenciales deben resolverse con Databricks Secrets o la identidad
administrada del job. Nunca se guardan en TOML, notebooks, tags o mensajes de
error.

En modo local, MLflow usa `sqlite:///mlflow.db` para tracking y registry. Esto
evita el backend filesystem legado de MLflow y conserva runs, artefactos,
versiones y aliases en un único almacén local.
