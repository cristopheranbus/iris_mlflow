# Ejecución local

El modo local permite probar entrenamiento, evaluación y promoción sin Spark,
Unity Catalog ni Model Serving.

## Instalación

La base local canónica es `<raiz-del-proyecto>/.local/mlflow/mlflow.db`; la configuración la
resuelve de forma absoluta para evitar duplicados por directorio de ejecución.

Desde la raíz instala el grupo de notebooks con `uv sync --group notebooks` y ejecútalos
con Jupyter. El dataset de desarrollo está en
`data/local/iris_features.csv` y conserva el mismo contrato de columnas que la
tabla Delta productiva.

## Entrenamiento

```powershell
$env:IRIS_RUNTIME = "local"
uv run --group notebooks jupyter notebook notebooks/training/random_forest.ipynb
```

También puedes ejecutar `notebooks/training/xgboost.ipynb`. El modelo se registra como
`iris_classifier` en SQLite, con tracking y registry bajo `.local/mlflow/`. Se
conservan métricas, artefactos, tags y descripción. El entrenamiento mueve
únicamente `Challenger`; `Champion` se asigna durante el despliegue.

## Promoción simulada

Ejecuta los notebooks de `deployment/` en este orden: evaluación, aprobación y
despliegue. `create_deployment_job.ipynb` es una simulación independiente del
DAG local y no es necesaria para promover el modelo. Con
`IRIS_LOCAL_AUTO_APPROVE=true`, la aprobación
se registra automáticamente después de superar los gates. El despliegue local
no llama APIs de Databricks: ejecuta un smoke test, promueve el alias local y
escribe `.local/deployment/local_deployment_manifest.json`.

## Diferencia con Databricks Connect

Este modo es completamente local. Databricks Connect sólo permite lanzar el
código desde tu máquina mientras Spark y los datos siguen ejecutándose en
Databricks; por tanto no equivale a este flujo local.
