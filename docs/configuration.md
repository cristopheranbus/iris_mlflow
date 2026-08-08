# Configuración

`TrainingConfig` conserva los nombres y defaults existentes. Las variables de
entorno tienen prioridad sobre widgets Databricks, y los widgets tienen
prioridad sobre los defaults.

La configuración versionada vive en `config/training.toml`. El entrenamiento
lee directamente `workspace.default.iris_features`; `IRIS_DATA_PATH` ya no se
usa en el flujo normal. `IRIS_FEATURE_TABLE_VERSION` permite fijar una versión
Delta para reproducibilidad.

Las variables sensibles, especialmente tokens, deben llegar desde Databricks
Secrets o desde el entorno de ejecución. Nunca deben escribirse en notebooks,
archivos `.env` versionados, tags de MLflow o mensajes de error.
