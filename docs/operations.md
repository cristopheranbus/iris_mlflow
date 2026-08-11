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
8. Confirmar endpoint `iris-classifier` en estado `READY`.
9. Confirmar alias `Champion` y Activity Log.

## Crear o reconectar el job

Ejecutar `deployment/create_deployment_job.ipynb` como administrador. Configurar
`IRIS_DEPLOYMENT_CLUSTER_ID`, `IRIS_DEPLOYMENT_NOTEBOOK_ROOT` y el service
principal de producción. El job usa `model_name` y `model_version` como
parámetros de nivel de job y limita la concurrencia a una ejecución.

## Diagnóstico

- Si no hay trigger, revisar `deployment_job_id`, permisos y modelo de tres
  niveles.
- Si falla evaluación, revisar `evaluation_decision` y los artefactos.
- Si falla aprobación, confirmar el tag exacto `Approval_Check=Approved`.
- Si falla serving, revisar permisos del service principal, endpoint y logs.
- Si falla el smoke test, no promover `Champion`.

## Despliegue manual

Se puede ejecutar el job desde Jobs o desde la página de la versión, indicando
`model_name` y `model_version`. La versión debe superar evaluación y aprobación.

## Prueba local

Configura `IRIS_RUNTIME=local`, ejecuta un notebook de entrenamiento y luego,
en orden, `evaluate_model.ipynb`, `approval.ipynb`, `deploy_model.ipynb` y
`create_deployment_job.ipynb`. Con `IRIS_LOCAL_AUTO_APPROVE=true` la aprobación
se registra automáticamente después de superar los gates. El despliegue local
es un smoke test y genera un manifiesto, no un endpoint.

En Databricks configura `IRIS_RUNTIME=databricks` y usa los parámetros
`model_name` y `model_version`; ese es el único flujo productivo y actualiza
Model Serving.

## Rollback

Seguir `docs/rollback.md`. El rollback restaura primero el endpoint y luego el
alias `Champion`; nunca se debe borrar la versión anterior.
