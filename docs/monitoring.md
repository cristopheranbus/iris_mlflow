# Monitoreo de producción

`config/monitoring.toml` define la ventana, el mínimo de observaciones y los
umbrales. El job `model_monitoring` consulta una Unity AI Gateway inference table
con el esquema vigente (`event_time`, `invocation_id`, `status_code`,
`latency_ms`, `logging_error_codes`).

El monitoreo es deliberadamente **alert-only**: nunca mueve el alias Champion ni
ejecuta rollback. El workflow `04 · Production model monitoring` corre cada seis
horas, abre o actualiza un Issue `ml-monitoring` cuando el job falla y cierra el
incidente después de la recuperación.

Antes de habilitarlo en GitHub:

1. Habilitar Unity AI Gateway inference tables para el endpoint.
2. Conceder al service principal `USE CATALOG`, `USE SCHEMA` y `SELECT` sobre la
   tabla payload.
3. Confirmar el nombre en `inference_table_name` para cada target del bundle.
4. Establecer `DATABRICKS_DEPLOY_ENABLED=true` en el repositorio.
5. Ejecutar manualmente el workflow contra `dev` y simular una alerta.

Sin etiquetas reales se monitorean disponibilidad, errores, latencia y errores
de captura. `prediction_drift` ya forma parte del contrato Python y se activará
cuando exista una tabla procesada con predicciones comparables. Accuracy y F1 de
producción no deben calcularse hasta disponer de ground truth.
