# Iris MLflow

Proyecto reproducible para entrenar clasificadores Iris, evaluarlos con MLflow
y registrarlos en Unity Catalog para un posterior despliegue.

## Flujo

```text
workspace.default.iris_features -> validación Spark -> split estratificado
                                      |
                              entrenamiento y evaluación
                                      |
                       registro, tags, comentarios y alias Challenger
```

Los notebooks de entrenamiento leen directamente la tabla Delta
`workspace.default.iris_features`. No leen ni crean CSV durante el flujo
normal. `ensure_feature_table` queda reservado para bootstrap explícito.

## Componentes

- `random_forest.ipynb`: entrenamiento Random Forest.
- `xgboost.ipynb`: entrenamiento XGBoost.
- `test_endpoint.ipynb`: prueba manual de Model Serving.
- `tools/src/iris_mlflow_utils`: configuración, carga Delta, evaluación, registry y serving.
- `config/training.toml`: configuración versionada sin secretos.
- `tests/`: pruebas unitarias y de integración local.
- `docs/`: arquitectura, configuración, pruebas y operación.

## Instalación

Se requiere Python 3.12 o superior y `uv`:

```powershell
cd tools
uv venv --python 3.12
uv sync --dev
```

## Ejecución

1. Verifica que exista `workspace.default.iris_features` en Unity Catalog.
2. Revisa `config/training.toml`.
3. Abre `random_forest.ipynb` o `xgboost.ipynb`.
4. Ejecuta las celdas en orden.
5. Revisa el run, la versión registrada, los tags, comentarios y alias.

En Databricks, la primera celda instala `tools` en modo editable y reinicia
Python antes de importar la librería.

## Configuración

La precedencia de configuración es:

1. variables de entorno;
2. `config/training.toml`;
3. widgets únicamente si `IRIS_ENABLE_WIDGET_OVERRIDES=true`.

Los valores principales son `IRIS_FEATURE_TABLE`,
`IRIS_FEATURE_TABLE_VERSION`, `IRIS_REGISTERED_MODEL_NAME`, aliases, semilla,
split, experimento y métrica principal. `IRIS_DATA_PATH` está deprecated y no
se utiliza en el entrenamiento desde Delta.

## MLflow y Unity Catalog

El registry se configura como `databricks-uc`. Cada versión registra:

- `model_type`: `random_forest`, `xgboost` o `neural_network`;
- framework, tarea, dataset y tabla de features;
- tags de versión y del modelo registrado;
- descripción del modelo y de la versión;
- alias `Challenger` después de alcanzar `READY`;
- evidencia en `metadata/model_registry_verification.json`.

`Champion` no se modifica automáticamente.

## Calidad

Desde `tools` ejecuta:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

Las pruebas locales no requieren Databricks. La validación real de Unity
Catalog y Model Serving requiere cluster, permisos y secretos configurados.

Consulta [docs/](docs/) para más información.
