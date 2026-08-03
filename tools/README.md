# Cliente REST para Databricks Model Serving

Este paquete contiene el cliente Python que consume el endpoint de clasificación de Iris publicado en Databricks Model Serving. Convierte una entrada tabular en una petición REST y la respuesta en una lista de predicciones o en un `pandas.DataFrame`.

El cliente no entrena modelos, no registra runs de MLflow, no crea endpoints y no administra secretos. Solo invoca un endpoint ya publicado y traduce sus errores a `DatabricksEndpointError`.

## Estructura

```text
tools/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/databricks_endpoint_client/
│   ├── __init__.py
│   └── client.py
└── tests/test_client.py
```

El notebook de prueba para ejecutar desde un cluster de Databricks está en la raíz del proyecto: [`../test_endpoint.ipynb`](../test_endpoint.ipynb).

## Requisitos e instalación

- Python 3.12 o superior.
- `uv` instalado.
- Un endpoint activo de Databricks Model Serving.
- Un token o secreto con permiso `CAN_QUERY`.
- La firma de entrada del modelo: nombres, orden y tipos de columnas.

Desde esta carpeta (`tools`):

```powershell
uv venv --python 3.12
uv sync --dev
```

`uv.lock` fija las versiones resueltas. Si se actualiza una dependencia, regenera el lockfile y revisa el diff antes de confirmarlo.

Dependencias de ejecución: `requests` para HTTP y `pandas` para la adaptación de DataFrames. Dependencias de desarrollo: `pytest`, `ruff`, `mypy`, `pandas-stubs` y `types-requests`.

## Configuración segura

En PowerShell, configura las variables solo para la sesión actual:

```powershell
$env:DATABRICKS_HOST = "mi-workspace.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "dapi-..."
$env:DATABRICKS_ENDPOINT_NAME = "iris-random-forest"
```

| Variable | Obligatoria | Descripción |
|---|---:|---|
| `DATABRICKS_HOST` | Sí | Dominio del workspace; puede incluir `https://` y una barra final. |
| `DATABRICKS_TOKEN` | Sí | Credencial enviada como `Bearer token`. |
| `DATABRICKS_ENDPOINT_NAME` | Sí | Nombre lógico del endpoint de Model Serving. |

El cliente normaliza el host y construye siempre:

```text
https://<workspace>/serving-endpoints/<endpoint-name>/invocations
```

Para automatización, usa el gestor de secretos de la plataforma. No guardes el token en el README, notebooks, archivos Python, historial de Git ni variables persistentes de usuario. El cliente no imprime el token ni lo incluye intencionalmente en excepciones.

## Contrato de entrada y salida

`consultar_endpoint` envía el formato `dataframe_split`:

```json
{
  "dataframe_split": {
    "columns": ["SepalLengthCm", "SepalWidthCm"],
    "data": [[5.1, 3.5]]
  }
}
```

Reglas del contrato:

1. `columns` debe tener al menos un nombre.
2. Cada fila de `data` debe tener tantos valores como columnas.
3. Los nombres deben coincidir exactamente con la firma del modelo.
4. El orden de las columnas debe ser el esperado por el modelo.
5. Los tipos deben ser compatibles con la firma publicada.
6. La respuesta debe ser JSON con una lista llamada `predictions`.

El cliente valida las reglas estructurales locales, pero no puede descubrir la firma del modelo ni corregir columnas equivocadas. Un HTTP 400 suele indicar una incompatibilidad con ella.

## API pública

### `consultar_endpoint(columnas, datos)`

Es la API de bajo nivel. Devuelve las predicciones sin transformarlas:

```python
from databricks_endpoint_client import consultar_endpoint

columnas = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
datos = [[5.1, 3.5, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5]]

predicciones = consultar_endpoint(columnas, datos)
print(predicciones)
```

### `predecir_dataframe(dataframe)`

Es la API recomendada cuando la aplicación ya trabaja con pandas:

```python
import pandas as pd
from databricks_endpoint_client import predecir_dataframe

entrada = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
)
resultado = predecir_dataframe(entrada)
```

La función convierte el DataFrame a `dataframe_split`, consulta el endpoint, comprueba que haya una predicción por fila y devuelve una copia con `prediccion`. El DataFrame original no se modifica.

## Errores y diagnóstico

Los fallos propios del cliente se exponen como `DatabricksEndpointError`, salvo el `TypeError` explícito de `predecir_dataframe` para entradas que no son DataFrames.

| Error | Causa probable | Qué revisar |
|---|---|---|
| Variables faltantes | Entorno incompleto | Las tres variables y la sesión de PowerShell. |
| 400 | Payload o firma incompatible | Nombres, orden, cantidad y tipos de columnas. |
| 401 | Token inválido o expirado | Token, host y origen de la credencial. |
| 403 | Falta autorización | Permiso `CAN_QUERY` de la identidad. |
| 404 | Host o endpoint incorrecto | Workspace, nombre exacto y URL generada. |
| 429 | Límite de frecuencia o cuota | Volumen y política de reintentos de la aplicación. |
| 500 | Error interno | Logs del endpoint y estado del modelo. |
| 503 | Servicio iniciando o no disponible | Estado del endpoint y scale-to-zero. |
| Timeout | Respuesta mayor a 60 segundos | Arranque en frío, lote y latencia. |
| JSON sin `predictions` | Contrato diferente | Respuesta real y versión del modelo. |
| Predicciones incompletas | Filas descartadas o agregadas | Logs y correspondencia fila-predicción. |

El cliente no reintenta automáticamente. Si se necesitan reintentos, deben vivir en la aplicación con backoff, límite de intentos, observabilidad y una decisión explícita sobre si la operación es segura de repetir.

## Pruebas y calidad

Las pruebas no llaman a Databricks: sustituyen `requests.post` con mocks y configuran variables de entorno temporales. Se pueden ejecutar sin token, red ni endpoint activo:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

La suite cubre payload y URL exitosos, variables faltantes, errores HTTP, JSON sin `predictions`, no mutación del DataFrame y cantidad incorrecta de predicciones. Toda nueva rama de errores debe incluir una prueba que documente el contrato esperado.

## Prueba manual contra Databricks

Con las variables configuradas, ejecuta desde `tools`:

```powershell
uv run python .\src\databricks_endpoint_client\client.py
```

También puedes importar [`../test_endpoint.ipynb`](../test_endpoint.ipynb) en Databricks. El notebook obtiene el token desde Databricks Secrets y no lo imprime. Completa los widgets `DATABRICKS_HOST`, `DATABRICKS_ENDPOINT_NAME`, `DATABRICKS_SECRET_SCOPE` y `DATABRICKS_SECRET_KEY`.

La firma Iris documentada originalmente usa cuatro columnas numéricas:

```python
["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
```

Antes de una ejecución real, confirma que el endpoint exista, que el modelo activo conserve esa firma y que la identidad tenga `CAN_QUERY`. Los nombres, estados y permisos de un workspace pueden cambiar.

## Límites conocidos

- No crea ni actualiza endpoints.
- No registra métricas ni runs en MLflow.
- No descubre automáticamente la firma del modelo.
- No convierte etiquetas, probabilidades ni formatos de respuesta alternativos.
- No implementa reintentos, circuit breaker ni rate limiting.
- Usa un timeout fijo de 60 segundos.

Estas decisiones mantienen la librería predecible. Si el proyecto necesita producción, conviene añadir esas capacidades en una capa de servicio con requisitos, métricas y pruebas de integración separados.

## Hooks de Git

Los hooks de Git son scripts locales que se ejecutan automáticamente en puntos
concretos del flujo de trabajo. Actualmente este repositorio no tiene hooks
activos; la carpeta `.git/hooks` fue limpiada de los archivos de ejemplo de Git.

La propuesta para el cliente es:

### `pre-commit`

Se ejecuta antes de crear un commit y debe permanecer rápido. Comprueba lint y
formato sin hacer llamadas externas:

```powershell
uv run ruff check .
uv run ruff format --check .
```

Si se prefiere que el hook corrija automáticamente el formato, puede usar:

```powershell
uv run ruff check . --fix
uv run ruff format .
```

Después de una corrección automática conviene revisar `git diff` antes de
confirmar el commit.

### `pre-push`

Se ejecuta antes de enviar commits al remoto. Como las pruebas pueden tardar
más que Ruff, aquí se ejecutan Pytest y Mypy:

```powershell
uv run pytest
uv run mypy
```

Si cualquiera falla, Git cancela el push. El commit local no se elimina: se
corrige el problema y se vuelve a intentar.

### GitHub Actions

El workflow en [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) repite
Ruff, formato, Pytest y Mypy en cada Pull Request hacia `main`. Esa repetición
es necesaria porque los hooks son locales, pueden no estar instalados y pueden
omitirse con `--no-verify`; GitHub Actions es el control centralizado que debe
proteger la rama principal.

```text
commit local ──► pre-commit: Ruff
push local   ──► pre-push: Pytest + Mypy
Pull Request ──► GitHub Actions: todas las validaciones
```
