# Configuración

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
