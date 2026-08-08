# Iris MLflow

Proyecto reproducible para entrenar clasificadores del dataset Iris, comparar
sus resultados con MLflow y preparar modelos para Databricks Model Serving.

## Componentes

- `random_forest.ipynb`: baseline con `RandomForestClassifier`.
- `xgboost.ipynb`: challenger con `XGBClassifier`.
- `tools/src/iris_mlflow_utils`: carga, validación, evaluación y tracking común.
- `tools/databricks_endpoint_client.ipynb`: cliente REST documentado para serving.
- `test_endpoint.ipynb`: prueba manual de un endpoint ya publicado.

```text
Iris.csv -> carga/validación -> split estratificado -> modelo
                                      |
                         métricas y artefactos MLflow
                                      |
                         registro en Unity Catalog
                                      |
                         publicación posterior en serving
```

Registrar un modelo no crea automáticamente un endpoint.

## Estructura

```text
.
├── random_forest.ipynb
├── xgboost.ipynb
├── test_endpoint.ipynb
├── readme.md
└── tools
    ├── pyproject.toml
    ├── uv.lock
    ├── README.md
    ├── src/iris_mlflow_utils
    ├── databricks_endpoint_client.ipynb
    └── tests
```

El paquete de entrenamiento no comparte responsabilidades con el cliente REST:
uno entrena y registra modelos; el notebook separado solo consulta endpoints.

Antes del entrenamiento, ambos notebooks validan o crean de forma idempotente
la tabla Delta `workspace.default.iris_features` en Unity Catalog. Si existe,
se verifica su esquema, clave `Id`, nulos y duplicados; si no existe, se crea
desde el CSV sin sobrescribir tablas existentes.

## Dataset y contrato de entrada

El CSV debe incluir `Species`, una columna opcional `Id` y columnas numéricas.
El dataset original contiene:

```text
Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
```

`Id` se excluye del entrenamiento. Las features son:

```python
["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
```

Deben conservarse los nombres, el orden y los tipos numéricos. `Species` se
codifica como índice entero; el mapping se registra en `class_mapping.json`.

## Instalación local

Se requiere Python 3.12 o superior y `uv`:

```powershell
cd tools
uv venv --python 3.12
uv sync --dev
```

Las dependencias incluyen pandas, numpy, scikit-learn, matplotlib, MLflow,
XGBoost y requests. El lockfile fija las versiones resueltas.

## Ejecución

1. Coloca `Iris.csv` en la raíz o define `IRIS_DATA_PATH`.
2. Abre `random_forest.ipynb` o `xgboost.ipynb`.
3. Ejecuta las celdas en orden.
4. Revisa métricas, `run_id`, URI del modelo y versión registrada.

La primera celda instala `tools` en modo editable:

```python
%pip install -e ./tools
```

En Databricks, `%pip` puede reiniciar Python; los imports deben ejecutarse
después de esa celda.

## Ejecución en Databricks

El notebook detecta Databricks y permite configurar valores mediante widgets o
variables de entorno. La ruta predeterminada es:

```text
/Volumes/workspace/my_data/my_volumen/Iris.csv
```

Una ruta de volumen general es:

```text
/Volumes/<catalogo>/<esquema>/<volumen>/Iris.csv
```

La identidad necesita permisos para leer el volumen, escribir en el experimento
y registrar modelos en el catálogo y esquema elegidos.

## Configuración

`iris_mlflow_utils.config` lee primero variables de entorno, después widgets de
Databricks y finalmente valores predeterminados.

| Variable | Obligatoria | Predeterminado | Propósito |
|---|---:|---|---|
| `IRIS_DATA_PATH` | No | `Iris.csv` o volumen Databricks | Ruta del CSV |
| `IRIS_EXPERIMENT_NAME` | No | `iris_mlflow` o `/Shared/iris_mlflow` | Experimento |
| `IRIS_ARTIFACT_LOCATION` | No | Vacío | Ubicación explícita de artefactos |
| `MLFLOW_TRACKING_URI` | No | Local o `databricks` | Backend de tracking |
| `MLFLOW_REGISTRY_URI` | No | `databricks-uc` | Registry de modelos |
| `IRIS_DATASET_VERSION` | No | `iris-csv` | Versión declarada del dataset |
| `IRIS_PROJECT_VERSION` | No | `2.0.0` | Versión del código |
| `IRIS_AUTHOR` | No | `unknown` | Responsable del run |
| `IRIS_PURPOSE` | No | `baseline-classification` | Finalidad |
| `IRIS_TEST_SIZE` | No | `0.20` | Proporción de test |
| `IRIS_RANDOM_STATE` | No | `42` | Semilla |
| `IRIS_PRIMARY_METRIC` | No | `test_f1_weighted` | Métrica principal |

Los notebooks comparten un único nombre configurable:

```text
workspace.default.iris_classifier
```

El nombre común del modelo se configura únicamente con
`IRIS_REGISTERED_MODEL_NAME` en formato `catalog.schema.model`.

## MLflow y Unity Catalog

El tracking URI recibe runs, parámetros y métricas. El registry URI registra
modelos; son conceptos diferentes y pueden requerir permisos diferentes.

Cada run registra:

- parámetros del algoritmo, dataset y split;
- versión del código y del dataset;
- métricas train/test de accuracy, precision, recall y F1 weighted;
- tags de algoritmo, autor y propósito;
- `classification_report.json`;
- `confusion_matrix.png`;
- `class_mapping.json`;
- signature e input example;
- modelo serializado bajo el artefacto `model`.

El registry predeterminado es Unity Catalog (`databricks-uc`) y usa nombres de
tres niveles. La URI del modelo dentro del run tiene esta forma:

```text
runs:/<run_id>/model
```

## Modelos

Random Forest combina árboles independientes. `n_estimators` controla cuántos
árboles se entrenan y `max_depth` limita su complejidad.

XGBoost construye árboles secuencialmente. Sus parámetros principales son
`learning_rate`, `n_estimators`, `max_depth`, `subsample` y
`colsample_bytree`. Ambos notebooks comparten exactamente la preparación de
datos, el split, las métricas y el tracking.

## API reutilizable

`tools/src/iris_mlflow_utils` expone:

- `build_config`: configuración local o Databricks.
- `load_dataset`: validación, features, etiquetas y mapping.
- `train_test_split`: se usa directamente desde scikit-learn en ambos notebooks.
- `evaluate_model`: métricas y diagnósticos de una partición.
- `evaluate_train_test`: evaluación uniforme train/test.
- `build_metrics_summary_table` y `build_classification_table`: preparan tablas
  para registrarlas directamente con `mlflow.log_table`.

## Serving y seguridad

El cliente REST usa `DATABRICKS_HOST`, `DATABRICKS_TOKEN` y
`DATABRICKS_ENDPOINT_NAME`. Envía `dataframe_split` y espera
`{"predictions": [...]}`. Nunca guardes tokens en notebooks, README, commits,
historial de shell ni variables persistentes; usa Databricks Secrets.

Antes de publicar un endpoint:

1. revisa la métrica de test;
2. confirma la signature y las cuatro features;
3. carga y prueba el modelo registrado;
4. selecciona la versión aprobada;
5. configura secretos y permisos;
6. valida una predicción real con el cliente REST.

## Pruebas y calidad

Desde `tools` ejecuta:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Las pruebas no llaman a Databricks: usan CSV temporales y mocks. La ejecución
real de notebooks es una validación de integración y requiere dataset, cluster,
permisos y MLflow accesible.

## Problemas frecuentes

| Problema | Qué revisar |
|---|---|
| Dataset no encontrado | `IRIS_DATA_PATH` y permisos del volumen |
| Columnas faltantes | `Species`, `Id` y features numéricas |
| Valores nulos | Limpieza del CSV |
| Error de `%pip` | Cluster, Python y ruta `./tools` |
| Error de XGBoost | Dependencia instalada en el entorno del notebook |
| Error de firma | Nombres, orden y tipos de features |
| Modelo no registrado | URI `databricks-uc` y permisos de Unity Catalog |
| Artefacto no encontrado | `run_id`, nombre `model` y permisos |
| Error con `__pycache__` | No versionar archivos generados de Python |

## Reproducibilidad

Conserva dataset, código, dependencias, semilla, split, hiperparámetros y
`run_id`. Comparar métricas exige que esos elementos sean compatibles. Un run de
MLflow es la evidencia del entrenamiento y no debe confundirse con el endpoint
que eventualmente sirva el modelo.

## MLflow 3 en Databricks

Los notebooks usan `mlflow[databricks]>=3.1,<4` y el tracking server
administrado por Databricks. En un notebook Databricks se instala el paquete y
se reinicia Python antes de importar las utilidades:

```python
%pip install -e ./tools "mlflow[databricks]>=3.1,<4"
dbutils.library.restartPython()
```

La URI de tracking se deja en la configuración nativa de Databricks salvo que
`MLFLOW_TRACKING_URI` se defina explícitamente. El registry permanece en
`databricks-uc` y los modelos usan nombres completos de Unity Catalog.

Cada run produce tracking tradicional y tracing de las etapas principales. Los
traces se pueden revisar desde la pestaña de traces del experimento. Los inputs
y outputs completos del dataset no se envían al trace; solo se registran
metadatos técnicos y resúmenes.

La evaluación usa `mlflow.models.evaluate` y deja estas tablas en el run:

```text
evaluation/metrics_summary.json
evaluation/classification_by_class.json
```

También se registran `classification_report.json` y
`confusion_matrix.png`. La URI utilizada para verificar el modelo es la URI
retornada por `log_model`; no se reconstruye manualmente como `runs:/...`.

Para registrar modelos en Unity Catalog se necesitan `USE CATALOG`, `USE
SCHEMA` y `CREATE MODEL`. Para cargar una versión registrada se necesita
`EXECUTE` sobre el modelo.

## Parametrización y documentación de APIs

Los notebooks reciben desde `TrainingConfig` los nombres de experimento,
modelo, feature table, aliases, dataset, split, input example y configuración
opcional de Serving. Los valores pueden llegar desde widgets Databricks o
variables de entorno; no se escriben credenciales en el código.

Parámetros principales:

| Variable | Uso |
|---|---|
| `IRIS_REGISTERED_MODEL_NAME` | Modelo UC `catalog.schema.model`. |
| `IRIS_FEATURE_TABLE` | Tabla Delta UC `catalog.schema.table`. |
| `IRIS_CHAMPION_ALIAS` | Alias de la versión productiva. |
| `IRIS_CHALLENGER_ALIAS` | Alias reasignado por cada entrenamiento. |
| `MLFLOW_TRACKING_URI` | Tracking server opcional; vacío usa Databricks nativo. |
| `MLFLOW_REGISTRY_URI` | Registry, por defecto `databricks-uc`. |
| `IRIS_MODEL_INPUT_EXAMPLE_ROWS` | Filas usadas para documentar la firma. |

Cada llamada principal a MLflow, Spark y Unity Catalog incluye argumentos
nombrados y comentarios cercanos que explican su propósito. El proceso registra
cada nuevo modelo como `Challenger`, pero no modifica automáticamente
`Champion` ni el endpoint productivo.

Los secretos deben obtenerse desde Databricks Secrets o el entorno de ejecución.
No se registran tokens en parámetros, tags, traces ni mensajes de error.
