# Notebooks de entrenamiento

`notebooks/training/random_forest.ipynb` y `notebooks/training/xgboost.ipynb`
son notebooks ejecutables y auditables.
Conservan explícitamente las llamadas nativas del framework y de MLflow; las
utilidades compartidas se usan sólo para configuración, datos, evaluación
homologada y metadatos del registry.

## Responsabilidad

Cada notebook realiza el mismo contrato:

1. Detecta si corre localmente o en Databricks.
2. Carga la configuración versionada.
3. Lee el dataset correspondiente al runtime.
4. Valida el esquema y excluye `Id` del modelo.
5. Construye un split estratificado reproducible.
6. Entrena con la API nativa del algoritmo.
7. Registra parámetros, métricas y artefactos en MLflow.
8. Registra una versión de `iris_classifier`.
9. Agrega tags, descripción y alias `Challenger`.
10. Carga la versión publicada y ejecuta predicciones de verificación.

El entrenamiento no aprueba candidatos, no actualiza Model Serving y no mueve
el alias `Champion`.

## Diferencia entre los notebooks

| Elemento | Random Forest | XGBoost |
|---|---|---|
| Estimador | `RandomForestClassifier` | `XGBClassifier` |
| Registro | `mlflow.sklearn.log_model` | `mlflow.xgboost.log_model` |
| Tag `model_type` | `random_forest` | `xgboost` |
| Framework | `sklearn` | `xgboost` |

La fuente, el split, las métricas, los artefactos y el contrato de registro son
iguales para ambos, por lo que sus resultados son comparables.

## Entradas

En Databricks:

```text
Dataset: workspace.default.iris_features
Registry: databricks-uc
Modelo: workspace.default.iris_classifier
```

En local:

```text
Dataset: data/local/iris_features.csv
Tracking/Registry: sqlite:///.local/mlflow/mlflow.db
Modelo: iris_classifier
```

Las features efectivas son `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm` y
`PetalWidthCm`, todas como `float64`. `Species` es la etiqueta e `Id` se conserva
sólo para trazabilidad.

## Salidas en MLflow

Cada ejecución publica:

- Parámetros del algoritmo, split, semilla, dataset y columnas.
- Métricas de train y test: accuracy, precision, recall y F1 macro/weighted.
- Evaluación multiclass de MLflow.
- Matrices de confusión absoluta y normalizada.
- ROC y Precision-Recall multiclass.
- Lift y cumulative gain.
- Distribución de probabilidades y feature importance.
- Reportes por clase, esquema, mapping y tabla de predicciones.
- Firma de cuatro features y ejemplo de entrada.
- Tags y comentarios del run y de la versión registrada.

## Ejecución local

Desde la raíz del proyecto:

```powershell
$env:IRIS_RUNTIME = "local"
uv run --group notebooks jupyter notebook notebooks/training/random_forest.ipynb
```

Para XGBoost cambia el archivo por `notebooks/training/xgboost.ipynb`. No es necesario definir las
URI de MLflow manualmente: la configuración local las resuelve contra la raíz
del proyecto.

## Ejecución en Databricks

Importa o despliega el proyecto, asocia el notebook a compute compatible y
ejecuta todas las celdas en orden. La primera celda de código instala el proyecto y
reinicia Python; después del reinicio el notebook continúa con configuración y
entrenamiento. El runtime Databricks nunca lee el CSV local.

## Cómo confirmar que terminó

La última celda debe finalizar sin excepción y mostrar:

```text
Experiment ID
Run ID
Model ID
Model URI
Registered version
Trace ID
Evaluation metrics
Sample predictions
```

Después verifica en MLflow:

1. El run está en estado `FINISHED`.
2. Existen parámetros y métricas de train/test.
3. La carpeta `evaluation/` contiene los artefactos esperados.
4. Existe una nueva versión de `iris_classifier`.
5. El alias `Challenger` apunta a esa versión.
6. La versión contiene `model_type`, `model_framework` y descripción.

Si falla una de estas comprobaciones, el notebook levanta una excepción y no se
debe iniciar la promoción.

## Problemas frecuentes

- Runtime o dataset incorrecto: revisar las cuatro líneas de contexto impresas
  antes de entrenar.
- Experimento eliminado: la inicialización intenta restaurarlo automáticamente.
- Artefactos faltantes: revisar la excepción de la última celda y el run.
- Versión sin alias: comprobar permisos de escritura sobre el modelo registrado.
- Base local vacía en la UI: iniciar MLflow desde la raíz usando `.local/mlflow/mlflow.db`.
