# Infraestructura declarativa y CI/CD

`databricks.yml` es la fuente de verdad de los Deployment Jobs. El bundle construye el
wheel, publica notebooks y configura el DAG serverless
`evaluate_model -> Approval_Check -> deploy_model`.

## Aislamiento de ambientes

| Target | Modelo | Endpoint | Job |
|---|---|---|---|
| `dev` | `workspace.default.iris_classifier_dev` | `iris-classifier-dev` | `model-deployment-dev` |
| `prod` | `workspace.default.iris_classifier` | `iris-classifier` | `model-deployment` |

La tabla `workspace.default.iris_features` es compartida y de sólo lectura. Cada
modelo conserva un `deployment_job_id` distinto. Producción sólo se ejecuta
manualmente desde `main`.

## Suspensión operativa

`DATABRICKS_DEPLOY_ENABLED=false` evita autenticación y llamadas cloud. El workflow
informa que el despliegue está suspendido y los controles locales siguen activos. Para
reanudar:

1. Confirmar que la organización Databricks está activa.
2. Ejecutar el bootstrap de permisos con una identidad administradora.
3. Ejecutar el preflight para el ambiente.
4. Cambiar la variable a `true`.
5. Desplegar `dev`; habilitar `prod` sólo después de la prueba completa.

## OIDC y GitHub Environments

Crear `databricks-dev` y `databricks-production`. Cada uno define
`DATABRICKS_HOST` y su propio `DATABRICKS_CLIENT_ID`; producción también define
`DATABRICKS_OPERATORS_GROUP`. No se almacena client secret.

La política federada debe copiar exactamente el claim `sub` emitido por GitHub:

```text
repo:<owner>/<repository>:environment:databricks-dev
repo:<owner>/<repository>:environment:databricks-production
```

Los identificadores reales permanecen en GitHub y Databricks. El environment de
producción debe aceptar exclusivamente `main` y requerir aprobación. El propietario
actual puede ser aprobador inicial, pero un segundo mantenedor es necesario para una
separación real de funciones.

## Permisos y preflight

Con una sesión administradora:

```powershell
.\ops\databricks\bootstrap_permissions.ps1 `
  -Principal "<application-id>" `
  -ModelName "workspace.default.iris_classifier_dev"
```

El script aplica permisos idempotentes de Unity Catalog. Después del primer bundle
deploy se completa `CAN_MANAGE_RUN` sobre el job y `CAN_MANAGE` sobre el endpoint
mediante la UI o API de permisos.

El workflow ejecuta `ops/databricks/preflight.py` antes de validar el bundle.
Comprueba identidad OIDC, cuenta activa, tabla, modelo y grants efectivos. Falla antes
de modificar infraestructura.

## Despliegue manual

```powershell
databricks bundle validate -t dev --profile dev
databricks bundle deploy -t dev --profile dev
databricks bundle run -t dev connect_deployment_job --profile dev
```

El primer comando consulta Databricks. El segundo crea o actualiza los Jobs y archivos.
El tercero escribe `deployment_job_id` en el modelo correspondiente; desde entonces una
nueva versión dispara automáticamente su Deployment Job.

## Diagnóstico

- `Organization ... cancelled or is not active`: mantener el interruptor en `false` y
  regularizar suscripción o facturación.
- Error OIDC: contrastar environment, issuer, audience y claim `sub` real.
- Grants vacíos: ejecutar el bootstrap con una identidad administradora.
- Modelo inexistente: entrenar y registrar una versión en el ambiente antes de conectar
  el Deployment Job.
