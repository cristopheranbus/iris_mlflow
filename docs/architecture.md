# Arquitectura

Los notebooks son las interfaces de ejecución y se mantienen sin cambios.
`tools/src/iris_mlflow_utils` contiene las utilidades reutilizables:

- `config.py`: configuración local y Databricks.
- `constants.py`: contrato de columnas y tipos del dataset Iris.
- `data.py`: carga, validación y codificación de etiquetas.
- `evaluation.py`: métricas y tablas de evaluación.
- `feature_table.py`: creación y validación idempotente en Unity Catalog.
- `serving.py`: cliente REST testeable para Model Serving.

La separación permite mejorar la librería sin cambiar el contrato que utilizan
los notebooks existentes.
