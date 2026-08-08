# Configuración

`TrainingConfig` conserva los nombres y defaults existentes. Las variables de
entorno tienen prioridad sobre widgets Databricks, y los widgets tienen
prioridad sobre los defaults.

Las variables sensibles, especialmente tokens, deben llegar desde Databricks
Secrets o desde el entorno de ejecución. Nunca deben escribirse en notebooks,
archivos `.env` versionados, tags de MLflow o mensajes de error.
