# Operación

## Publicar una versión

1. Verificar `workspace.default.iris_features`.
2. Revisar `config/training.toml`.
3. Ejecutar un notebook de entrenamiento.
4. Confirmar la nueva versión `Challenger` y sus tags.
5. Revisar métricas y artefactos de `evaluation/`.
6. Esperar el Deployment Job.
7. Usar el botón **Approve** en la versión del modelo; esto aplica
   `Approval_Check=Approved`.
8. Confirmar `iris-classifier-dev` en desarrollo o `iris-classifier` en producción.
9. Confirmar alias `Champion` y Activity Log.

## Crear o reconectar el job

Validar y desplegar `databricks.yml`; no crear el job manualmente desde un
notebook. Luego ejecutar el recurso `connect_deployment_job`, que recibe el ID
real mediante referencia del bundle y lo registra como `deployment_job_id` del
modelo. El job usa `model_name` y `model_version`, concurrencia `1`, serverless y
cero reintentos en aprobación. Consulta `docs/infrastructure.md`.

## Diagnóstico

- Si no hay trigger, revisar `deployment_job_id`, permisos y modelo de tres
  niveles.
- Si falla el preflight, no desplegar: corregir cuenta, identidad o grants.
- Si falla evaluación, revisar `evaluation_decision` y los artefactos.
- Si falla aprobación, confirmar el tag exacto `Approval_Check=Approved`.
- Si falla serving, revisar permisos del service principal, endpoint y logs.
- Si falla el smoke test, no promover `Champion`.
- Si el endpoint ya fue actualizado y falla `READY` o el smoke test, comprobar
  `rollback_status=restored`. Un valor `failed` requiere intervención inmediata.
- En el primer despliegue fallido, comprobar
  `rollback_status=deleted_new_endpoint`; el endpoint incompleto se elimina.

## Despliegue manual

Se puede ejecutar el job desde Jobs o desde la página de la versión, indicando
`model_name` y `model_version`. La versión debe superar evaluación y aprobación.

## Prueba local

Configura `IRIS_RUNTIME=local`, ejecuta un notebook de entrenamiento y luego,
en orden, `evaluate_model.ipynb`, `approval.ipynb` y `deploy_model.ipynb`.
`create_deployment_job.ipynb` sólo genera el manifiesto del DAG local. Con
`IRIS_LOCAL_AUTO_APPROVE=true` la aprobación
se registra automáticamente después de superar los gates. El despliegue local
es un smoke test y genera un manifiesto, no un endpoint.

La evaluación se registra en el mismo experimento que el entrenamiento. La
versión conserva `feature_table_version`, `evaluation_run_id`,
`comparison_run_id` y `evaluation_model_id` para enlazar dataset, métricas,
comparación y artefactos con el modelo evaluado.

Una promoción exitosa registra `smoke_test_status=passed`,
`deployment_status=deployed` y `lifecycle=champion`. La versión Champion anterior
queda como `lifecycle=previous_champion` y `deployment_status=superseded`.

En Databricks configura `IRIS_RUNTIME=databricks` y usa los parámetros
`model_name` y `model_version`; ese es el único flujo productivo y actualiza
Model Serving.

## Rollback

Seguir `docs/rollback.md`. El rollback restaura primero el endpoint y luego el
alias `Champion`; nunca se debe borrar la versión anterior.
