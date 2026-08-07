# Herramientas reutilizables

`iris_mlflow_utils` contiene únicamente configuración, carga/validación de
datos, métricas auxiliares y constructores de DataFrames para evaluación. El
cliente `databricks_endpoint_client` permanece separado y solo consume
endpoints de Model Serving.

## Estructura

```text
src/iris_mlflow_utils/
    __init__.py
    config.py
    data.py
    evaluation.py
src/databricks_endpoint_client/
    client.py
tests/
```

## Uso en notebooks

El split usa directamente la API oficial de scikit-learn:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    dataset.features,
    dataset.target,
    test_size=config.test_size,
    random_state=config.random_state,
    stratify=dataset.target,
)
```

El tracking y el tracing también quedan visibles en cada notebook mediante
`mlflow.start_run`, `mlflow.start_span`, `mlflow.log_params`,
`mlflow.log_metrics` y `mlflow.log_table`. La evaluación usa
`mlflow.models.evaluate`, y el registro usa `mlflow.sklearn.log_model` o
`mlflow.xgboost.log_model`.

`load_dataset` valida columnas, nulos, tipos numéricos, clases y excluye `Id`.
`evaluate_model` produce métricas, reporte y matriz de confusión para una
partición. `evaluate_train_test` aplica el mismo contrato a train y test.
`build_metrics_summary_table` y `build_classification_table` solo transforman
resultados a DataFrames; las tablas se guardan con `mlflow.log_table` desde el
notebook.

`ensure_feature_table` comprueba o crea una tabla Delta de Unity Catalog con
clave primaria `Id`. Nunca sobrescribe una tabla existente y detiene el flujo
si encuentra columnas faltantes, tipos incompatibles, nulos o claves duplicadas.

## Databricks y MLflow 3

```python
%pip install -e ./tools "mlflow[databricks]>=3.1,<4"
dbutils.library.restartPython()
```

La celda de reinicio es exclusiva de Databricks. El tracking usa el servidor
administrado del workspace salvo que `MLFLOW_TRACKING_URI` se configure
explícitamente. Unity Catalog se configura con `databricks-uc` y nombres de
modelo de tres niveles.

Los notebooks conservan `model_info.model_uri`, `model_info.model_id` y
`model_info.registered_model_version` retornados por MLflow 3. Esa URI oficial
se utiliza para cargar el modelo; nunca se reconstruye manualmente desde el
`run_id`.

## Calidad

Desde esta carpeta:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Las pruebas son locales y no validan permisos, traces ni registro en un
workspace Databricks real. Esa validación requiere ejecutar los notebooks en
un cluster con acceso al experimento y Unity Catalog.

## Seguridad

No guardes tokens ni secretos en código, notebooks, README o Git. Usa
Databricks Secrets o el gestor de credenciales del entorno.

## Parámetros documentados

`TrainingConfig` centraliza los valores que controlan MLflow, Unity Catalog,
la feature table y el despliegue opcional. Los notebooks usan argumentos
nombrados en las APIs y cada argumento no obvio tiene un comentario sobre su
entrada, efecto y salida.

La configuración incluye el modelo UC de tres niveles, la tabla Delta de
features, `Champion`, `Challenger`, el experimento, tracking URI, registry URI,
dataset, split, semilla, tamaño del input example y valores de Serving.
Databricks widgets tienen prioridad después de las variables de entorno y los
valores predeterminados se usan como último recurso.

Cada entrenamiento mueve el alias configurable `Challenger` a la versión READY
recién registrada y guarda tags de algoritmo, framework, dataset, etapa y
training type. El alias `Champion` solo se consulta; la promoción y actualización
de endpoints pertenecen a un flujo posterior de comparación/despliegue.
