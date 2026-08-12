# Ejecución local

El modo local permite probar entrenamiento, evaluación y promoción sin Spark,
Unity Catalog ni Model Serving.

## Instalación

La base local canónica es `<raiz-del-proyecto>/mlflow.db`; la configuración la
resuelve de forma absoluta para evitar duplicados por directorio de ejecución.

Desde `tools` instala las dependencias con `uv sync` y ejecuta los notebooks
con Jupyter. El dataset de desarrollo está en
`data/local/iris_features.csv` y conserva el mismo contrato de columnas que la
tabla Delta productiva.

## Entrenamiento

```powershell
$env:IRIS_RUNTIME = "local"
cd tools
uv run jupyter notebook ..\random_forest.ipynb
```

También puedes ejecutar `xgboost.ipynb`. El modelo se registra como
`iris_classifier` en SQLite, con tracking y registry en `mlflow.db`. Se
conservan métricas, artefactos, tags y descripción. El entrenamiento mueve
únicamente `Challenger`; `Champion` se asigna durante el despliegue.

## Promoción simulada

Ejecuta los notebooks de `deployment/` en este orden: evaluación, aprobación y
despliegue. `create_deployment_job.ipynb` es una simulación independiente del
DAG local y no es necesaria para promover el modelo. Con
`IRIS_LOCAL_AUTO_APPROVE=true`, la aprobación
se registra automáticamente después de superar los gates. El despliegue local
no llama APIs de Databricks: ejecuta un smoke test, promueve el alias local y
escribe `artifacts/local_deployment_manifest.json`.

## Diferencia con Databricks Connect

Este modo es completamente local. Databricks Connect sólo permite lanzar el
código desde tu máquina mientras Spark y los datos siguen ejecutándose en
Databricks; por tanto no equivale a este flujo local.
