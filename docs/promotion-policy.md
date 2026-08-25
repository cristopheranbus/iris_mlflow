# Política de promoción

La promoción se gobierna mediante `config/promotion.dev.toml` y
`config/promotion.prod.toml`. Selecciona el perfil con
`IRIS_PROMOTION_PROFILE=dev|prod`; Databricks Bundles lo configura por target.

Cada regla declara `name`, `metric`, `operator` y un valor absoluto o
`baseline = "champion"` con `allowed_regression`. Los operadores permitidos son
`>=`, `<=`, `>` y `<`; no se ejecutan expresiones desde la configuración.

Una regla `required = false` es informativa. Una métrica ausente en una regla
obligatoria rechaza el candidato. La comparación relativa se omite únicamente
cuando todavía no existe Champion. La decisión completa, incluida la versión de
la política y el resultado de cada regla, se registra como metadata del modelo.

Para cambiar un umbral:

1. Modificar solamente el perfil correspondiente.
2. Abrir un pull request y esperar todos los checks.
3. Revisar el impacto en Challenger y Champion sobre el mismo holdout.
4. Aprobar y desplegar. El alias Champion cambia sólo después del smoke test.

Mientras exista un único mantenedor, CODEOWNERS usa `@cristopheranbus` y GitHub
no exige una aprobación imposible del propio autor. Para agregar revisores,
añadir sus usuarios en `.github/CODEOWNERS` y en `databricks-production`; después
activar revisión obligatoria, aprobación del último push y `prevent_self_review`.
