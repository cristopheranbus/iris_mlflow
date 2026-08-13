# Infraestructura declarativa

`databricks.yml` es la fuente de verdad del Deployment Job. Define el wheel de
utilidades, los notebooks, el DAG, los parámetros, la concurrencia, los
permisos y los targets `dev` y `prod`.

## Recursos

- `model_deployment`: ejecuta `evaluate_model` → `Approval_Check` →
  `deploy_model`. Usa cómputo serverless al no fijar un cluster.
- `connect_deployment_job`: conecta una vez el ID del job anterior con el
  modelo registrado. No crea ni modifica la definición del job.
- `iris_mlflow_tools`: construye el wheel de `tools/` y lo instala mediante un
  environment serverless compartido por las tareas.

El job admite sólo estos parámetros dinámicos:

```text
model_name
model_version
```

La concurrencia máxima es `1` y `Approval_Check` tiene cero reintentos para
preservar la pausa de aprobación manual.

## Targets

`dev` usa el nombre `model-deployment-dev` y la identidad que despliega el
bundle. `prod` usa `model-deployment`, ejecuta con el service principal indicado
en `production_service_principal` y entrega `CAN_MANAGE_RUN` al grupo de
operadores configurado. Sus archivos se despliegan en la carpeta privada de la
identidad que realiza el despliegue, no en `/Workspace/Shared`.

Antes de producción reemplaza obligatoriamente el UUID placeholder:

```powershell
$env:BUNDLE_VAR_production_service_principal = "<application-id>"
$env:BUNDLE_VAR_operators_group = "ml-model-operators"
databricks bundle validate -t prod --profile dev
```

## Ciclo manual

```powershell
databricks bundle validate -t dev --profile dev
databricks bundle deploy -t dev --profile dev
databricks bundle run -t dev connect_deployment_job --profile dev
```

El último comando escribe `deployment_job_id` en
`workspace.default.iris_classifier`. Desde ese momento, una nueva versión del
modelo puede disparar el Deployment Job asociado.

## CI/CD

`.github/workflows/databricks-bundle.yml` valida pull requests, despliega `dev`
al actualizar la rama `dev` y reserva `prod` para una ejecución manual protegida
por el environment `databricks-production`.

La secuencia de un despliegue habilitado es:

1. Autenticación sin secretos mediante OIDC y un service principal.
2. Validación de `databricks.yml` contra el target seleccionado.
3. Construcción del wheel `iris_mlflow_tools`.
4. Carga de notebooks, configuración y wheel a la carpeta privada del bundle.
5. Creación o actualización de `model_deployment` y
   `connect_deployment_job` mediante Jobs API.
6. Ejecución de `connect_deployment_job` para asociar el Job administrado con
   `workspace.default.iris_classifier`.

El runner temporal de GitHub sólo construye y orquesta. Las tareas
`evaluate_model`, `Approval_Check` y `deploy_model` corren posteriormente en
Databricks serverless con el wheel del proyecto instalado.

Los despliegues tienen un interruptor operativo a nivel de repositorio:

```text
DATABRICKS_DEPLOY_ENABLED=false
```

Con `false`, GitHub mantiene activos los tests y `Validate bundle`, pero marca
los jobs de despliegue como omitidos. Para reanudar `dev` y `prod`, cambia la
variable a `true` en **Settings → Secrets and variables → Actions →
Variables** y vuelve a ejecutar el workflow. El valor debe escribirse en
minúsculas.

La autenticación usa OIDC. GitHub entrega un token temporal y Databricks lo
intercambia por OAuth; no se almacena `DATABRICKS_CLIENT_SECRET`.

### 1. Crear los environments en GitHub

En el repositorio abre **Settings → Environments → New environment** y crea
exactamente:

```text
databricks-dev
databricks-production
```

En `databricks-dev`, agrega estas **Environment variables**:

```text
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_CLIENT_ID=<application-id-del-service-principal-dev>
```

En `databricks-production`, agrega:

```text
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_CLIENT_ID=<application-id-del-service-principal-prod>
DATABRICKS_OPERATORS_GROUP=<grupo-operador-de-databricks>
```

No agregues estas variables como secrets: no contienen credenciales. El
workflow usa `vars.DATABRICKS_HOST` y `vars.DATABRICKS_CLIENT_ID`.

### 2. Proteger producción

En `databricks-production` configura:

- Required reviewers: al menos una persona o equipo responsable.
- Prevent self-review: habilitado, cuando el plan de GitHub lo permita.
- Deployment branches: sólo `main`.

`databricks-dev` puede quedar sin aprobación manual para permitir validaciones y
despliegues desde la rama `dev`.

### 3. Crear las políticas de federación en Databricks

Usa un service principal diferente por ambiente. Ambos deben estar asignados al
workspace. Un account admin debe crear una política para cada environment con:

```text
Issuer: https://token.actions.githubusercontent.com
Subject claim: sub
Audience: valor exacto del claim `aud` emitido por GitHub
```

Subjects exactos para este repositorio:

```text
repo:cristopheranbus@6538839/iris_mlflow@1318684974:environment:databricks-dev
repo:cristopheranbus@6538839/iris_mlflow@1318684974:environment:databricks-production
```

Ejemplo con un perfil de administrador de cuenta:

```powershell
databricks account service-principal-federation-policy create `
  <service-principal-numeric-id> `
  --profile account-admin `
  --json '{
    "oidc_policy": {
      "issuer": "https://token.actions.githubusercontent.com",
      "audiences": ["https://dbc-f2dbc696-258a.cloud.databricks.com/oidc/v1/token"],
      "subject": "repo:cristopheranbus@6538839/iris_mlflow@1318684974:environment:databricks-dev"
    }
  }'
```

Repite el comando para producción, cambiando el service principal y el subject.
El `DATABRICKS_CLIENT_ID` de GitHub es el **application ID**; el argumento del
comando anterior es el **ID numérico interno** del service principal.

### 4. Otorgar permisos mínimos

El principal de despliegue necesita crear o actualizar Jobs y archivos del
bundle. El principal que ejecuta producción necesita además acceso a:

- `USE CATALOG` y `USE SCHEMA`.
- Lectura de `workspace.default.iris_features`.
- Lectura y administración de versiones/aliases del modelo registrado.
- Creación o actualización del endpoint `iris-classifier`.
- Ejecución del Deployment Job.

No concedas permisos de account admin al workflow.

### 5. Probar la configuración

En GitHub abre **Actions → Databricks bundle → Run workflow**, selecciona
`dev` y ejecuta: la ejecución manual valida el bundle. Para desplegar y conectar
el job de desarrollo, sube o integra el cambio en la rama `dev`; ese evento
ejecuta además `Deploy development` y `Connect deployment job to model`.

Para producción ejecuta manualmente el workflow desde `main`, selecciona
`prod` y aprueba el environment cuando GitHub lo solicite.

Si aparece un error de OIDC, comprueba primero que el nombre del environment y
el subject sean idénticos, incluida la capitalización. El workflow ya solicita
`id-token: write` y usa `DATABRICKS_AUTH_TYPE=github-oidc`.

En organizaciones con identificadores administrados, GitHub puede incluir IDs
numericos en el claim `sub`. Por eso la politica debe copiar el valor real del
token y no reconstruirlo a partir del nombre visible del repositorio.

### Diagnostico de despliegue

Si la validacion OIDC pasa pero Jobs responde `Organization ... has been
cancelled or is not active yet`, la federacion ya funciona. Verifica el estado
de suscripcion o facturacion de la cuenta Databricks antes de ampliar permisos.
Los principals de CI deben conservar como minimo `workspace-access`; no
necesitan privilegios de account admin.

#### Estado de la cuenta de desarrollo (12 de agosto de 2026)

La autenticacion OIDC y `databricks bundle validate` estan operativos. El
despliegue esta bloqueado en la cuenta, tanto con el service principal como con
el usuario administrador, por esta respuesta de Jobs API:

```text
Organization 7474648951266353 has been cancelled or is not active yet.
```

La cuenta fue creada el 27 de julio de 2026 y el trial de Databricks dura 14
dias. Para habilitar nuevamente la creacion de Jobs:

1. Abrir `https://accounts.cloud.databricks.com` con un account admin.
2. Ir a **Settings -> Subscription & billing**.
3. Agregar una tarjeta o vincular AWS Marketplace para pasar a pago por uso.
4. En **Settings -> Feature enablement**, aceptar los terminos de serverless si
   aparece la solicitud.
5. En GitHub, abrir la ejecucion fallida y seleccionar **Re-run failed jobs**.

Mientras la cuenta permanezca inactiva, conserva la variable de repositorio
`DATABRICKS_DEPLOY_ENABLED=false`. Una vez regularizada, cambiala a `true`
antes de repetir la ejecucion.

Si el plan aparece como `Cancelled` y no solamente como trial vencido,
Databricks no permite reactivarlo: se debe contactar al equipo de cuenta o
crear una cuenta nueva y actualizar `DATABRICKS_HOST`, las identidades y las
politicas OIDC de ambos environments.
