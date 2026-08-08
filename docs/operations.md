# Operación

El entrenamiento registra una nueva versión como `Challenger`. La promoción
a `Champion` y la actualización de un endpoint son acciones posteriores y
explícitas.

Los entrenamientos leen `workspace.default.iris_features` directamente. Si la
tabla no existe, el proceso falla y no crea una tabla desde CSV de forma
automática.

Antes de promover un modelo se deben revisar la métrica principal, la
signature, las cuatro features, los artefactos de evaluación y la capacidad de
cargar el modelo desde la URI retornada por MLflow.
