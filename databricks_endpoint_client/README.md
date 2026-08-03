# Cliente REST para Databricks Model Serving

Cliente Python para consumir un endpoint de Databricks Model Serving mediante REST API. Envía observaciones usando el formato `dataframe_split` y devuelve la lista de predicciones. También incluye una función para consultar directamente un `pandas.DataFrame`.

## Requisitos

- Python 3.12 o superior.
- Un endpoint activo de Databricks Model Serving.
- Un token con permisos para invocar el endpoint.
- `uv` instalado.

El token nunca se escribe en el código ni se imprime en los mensajes de error.

## Crear el entorno con uv

Desde la carpeta `databricks_endpoint_client`:

```powershell
uv venv --python 3.12
uv sync --dev
```

`uv sync --dev` instala las dependencias de ejecución y las herramientas de desarrollo declaradas en `pyproject.toml`: `requests`, `pandas`, `pytest`, `ruff`, `mypy`, `pandas-stubs` y `types-requests`.

## Variables de entorno en PowerShell

```powershell
$env:DATABRICKS_HOST = "mi-workspace.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "dapi-..."
$env:DATABRICKS_ENDPOINT_NAME = "iris-endpoint"
```

`DATABRICKS_HOST` puede incluir `https://`; el cliente normaliza el valor y construye siempre una URL HTTPS con este formato:

```text
https://<workspace>/serving-endpoints/<endpoint-name>/invocations
```

Las variables solo duran mientras esté abierta la sesión de PowerShell. Para una ejecución automatizada, utiliza el gestor de secretos de tu entorno; no guardes el token en Git, en el README ni en archivos `.py`.

## Ejecutar el script

Con las variables definidas:

```powershell
uv run python .\src\databricks_endpoint_client\client.py
```

El ejemplo ejecutable envía estas cuatro columnas y tres observaciones Iris:

```python
columnas = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
```

El endpoint debe aceptar exactamente los nombres y el orden de columnas que espera el modelo.

## Notebook para probarlo en Databricks

El notebook [`notebooks/probar_endpoint_databricks.ipynb`](notebooks/probar_endpoint_databricks.ipynb) permite probar el endpoint directamente desde un cluster de Databricks.

1. Importa el notebook en tu workspace.
2. Crea o identifica un secret scope y guarda allí el token de Databricks.
3. Ejecuta la primera celda y completa los widgets `DATABRICKS_HOST`, `DATABRICKS_ENDPOINT_NAME`, `DATABRICKS_SECRET_SCOPE` y `DATABRICKS_SECRET_KEY`.
4. Ejecuta las celdas en orden.
5. Revisa el código HTTP, las predicciones y el DataFrame final.

El notebook nunca imprime el token. El host se normaliza y la URL final queda como `https://<workspace>/serving-endpoints/<endpoint-name>/invocations`. Si el modelo usa otras columnas, cambia únicamente las listas `columnas` y `datos` por valores que coincidan con la firma del endpoint.

### Valores confirmados para el endpoint Iris

Para el endpoint actual, usa estos valores:

```text
DATABRICKS_HOST=https://dbc-f2dbc696-258a.cloud.databricks.com
DATABRICKS_ENDPOINT_NAME=iris-random-forest
```

El endpoint está en estado `READY`, usa el modelo `iris_random_forest-1` y tiene scale-to-zero. La primera invocación puede tardar aproximadamente entre 15 y 30 segundos mientras el servicio inicia; el notebook utiliza un timeout de 60 segundos.

El token debe pertenecer a una identidad con permiso `CAN_QUERY` sobre el endpoint. Los nombres `DATABRICKS_SECRET_SCOPE` y `DATABRICKS_SECRET_KEY` deben completarse con los valores reales del workspace.

La firma de entrada confirmada es:

```python
["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
```

Las cuatro columnas deben enviarse como valores numéricos `double`/`float64`. El endpoint no espera `Id` ni `Species`.

## Usar el cliente desde otro módulo

```python
import pandas as pd

from databricks_endpoint_client import consultar_endpoint, predecir_dataframe

columnas = ["sepal length (cm)", "sepal width (cm)"]
datos = [[5.1, 3.5], [6.4, 3.2]]
predicciones = consultar_endpoint(columnas, datos)

dataframe = pd.DataFrame(datos, columns=columnas)
resultado = predecir_dataframe(dataframe)
print(predicciones)
print(resultado)
```

`predecir_dataframe` convierte el DataFrame a `dataframe_split`, consulta el endpoint, devuelve una copia y agrega la columna `prediccion`. El DataFrame original no se modifica. Si Databricks devuelve una cantidad de predicciones diferente a la cantidad de filas, se lanza un error antes de devolver el resultado.

## Ejecutar pruebas

Las pruebas no llaman a Databricks: simulan `requests.post` con `unittest.mock`, por lo que no necesitan token ni conexión real.

```powershell
uv run pytest
```

Las pruebas cubren respuestas exitosas, errores HTTP 401, 403, 404, 429 y 503, timeout, JSON inválido, respuesta sin `predictions`, cantidad incorrecta de predicciones y variables faltantes.

## Ruff y Mypy

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Formato enviado

El cliente envía JSON con esta estructura:

```json
{
  "dataframe_split": {
    "columns": ["columna_1", "columna_2"],
    "data": [["valor_1", "valor_2"]]
  }
}
```

Los headers enviados son:

```python
{
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json",
}
```

El timeout es de 60 segundos.

## Errores frecuentes de Databricks

- **400**: el JSON no coincide con la firma del modelo, faltan columnas, el orden es incorrecto o hay tipos incompatibles.
- **401**: el token falta, expiró, es incorrecto o se configuró en otra sesión de PowerShell.
- **403**: el token funciona, pero su identidad no tiene permiso para invocar el endpoint.
- **404**: el host, el nombre del endpoint o la ruta del workspace son incorrectos.
- **429**: Databricks aplicó un límite de frecuencia o el endpoint está recibiendo demasiadas solicitudes. Implementa reintentos con backoff si tu aplicación lo necesita.
- **500**: el servicio encontró un error interno; revisa los logs del endpoint y su modelo.
- **503**: el endpoint está iniciando, no está disponible o no tiene capacidad. Espera y vuelve a intentar según la política de tu aplicación.
- **Timeout**: la respuesta tardó más de 60 segundos; revisa la carga, el tamaño del lote y el estado del endpoint.
- **JSON sin `predictions`**: la respuesta del endpoint usa otro formato o contiene un error estructurado distinto al esperado. Revisa la firma y la documentación del modelo.

Todos los errores generados por el cliente incluyen, cuando existe, el código HTTP, el texto devuelto por Databricks y una causa probable. El token no se incluye en esos mensajes.
