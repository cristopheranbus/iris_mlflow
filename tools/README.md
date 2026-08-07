# Herramientas reutilizables

Este proyecto Python contiene dos componentes separados:

```text
tools/src/iris_mlflow_utils/
    config.py       configuración local/Databricks
    data.py         carga, validación y split
    evaluation.py   métricas y diagnósticos
    tracking.py     tracking MLflow y Unity Catalog

tools/src/databricks_endpoint_client/
    client.py       cliente REST para Model Serving
```

`iris_mlflow_utils` es usado por `random_forest.ipynb` y `xgboost.ipynb`.
`databricks_endpoint_client` no entrena ni registra modelos: solo consulta un
endpoint ya publicado.

## Instalación

Se requiere Python 3.12 o superior y `uv`:

```powershell
uv venv --python 3.12
uv sync --dev
```

Las dependencias incluyen pandas, numpy, scikit-learn, matplotlib, MLflow,
XGBoost y requests. `uv.lock` fija las versiones resueltas.

Los notebooks instalan este proyecto en modo editable con:

```python
%pip install -e ./tools
```

En Databricks, ejecuta esa celda antes de los imports porque `%pip` puede
reiniciar Python.

## API de entrenamiento

### Configuración

`build_config` devuelve un `TrainingConfig` inmutable. Lee variables de entorno,
widgets de Databricks y valores predeterminados en ese orden.

### Datos

`load_dataset(path)` valida el archivo, exige un objetivo con al menos dos
clases, rechaza nulos y features no numéricas, excluye `Id` y codifica las
clases. Devuelve un `DatasetBundle` con dataframe, features, target, columnas y
mapping de clases.

`split_dataset(bundle, test_size, random_state)` realiza un split estratificado
y reproducible, preservando los nombres de columnas.

### Evaluación

`evaluate_model` devuelve accuracy, precision, recall, F1 weighted, reporte por
clase, matriz de confusión y predicciones. `evaluate_train_test` aplica el mismo
contrato a train y test para que los notebooks sean comparables.

### MLflow

`log_training_run` configura tracking y registry, reutiliza o crea el
experimento, abre un run, registra parámetros, tags, métricas, reportes, mapping
de clases, matriz de confusión, signature, input example y modelo.

El parámetro `model_type` debe ser `RandomForest` o `XGBoost`; determina el
flavor de MLflow usado para serializar el estimador.

## Contrato de datos

El dataset necesita `Species`, una columna opcional `Id` y al menos una feature
numérica. El dataset Iris habitual utiliza:

```text
SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
```

Las clases se convierten a índices enteros desde cero. El mapping se registra
como `class_mapping.json` para que serving pueda traducir la predicción.

## Cliente REST

El cliente requiere:

```text
DATABRICKS_HOST
DATABRICKS_TOKEN
DATABRICKS_ENDPOINT_NAME
```

Envía `dataframe_split` y espera `{"predictions": [...]}`. Valida configuración,
URL, tamaño de filas, códigos HTTP y cuerpo JSON. No implementa reintentos
automáticos: esa política pertenece a la aplicación consumidora.

## Calidad

Ejecuta desde esta carpeta:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Las pruebas de entrenamiento usan un CSV temporal y un modelo pequeño. Las
pruebas REST reemplazan la red con mocks; ninguna requiere token, endpoint ni
workspace real.

## Seguridad

No guardes tokens en archivos, notebooks, README, historial de shell ni Git.
Usa Databricks Secrets o el gestor de credenciales del entorno. Los errores del
cliente no incluyen intencionalmente el token.
