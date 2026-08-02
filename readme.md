# Iris MLflow

Proyecto de clasificación del dataset Iris usando `RandomForestClassifier` y MLflow. El notebook [`iris.ipynb`](iris.ipynb) contiene el flujo completo: configuración, validación de datos, entrenamiento, evaluación y registro del modelo.

## Objetivo

Entrenar un modelo reproducible que clasifique la especie de una flor Iris y dejar evidencia de cada entrenamiento en MLflow.

## Requisitos

Instala las dependencias en el entorno donde ejecutarás el notebook:

```bash
pip install pandas numpy scikit-learn matplotlib mlflow jupyter
```

En Databricks, usa un cluster o entorno que incluya esas librerías. La versión exacta de las dependencias debe fijarse en el entorno de ejecución para obtener resultados reproducibles.

## Ejecución rápida

1. Coloca `Iris.csv` en la carpeta actual o define `IRIS_DATA_PATH` con su ruta completa.
2. Abre `iris.ipynb` en Jupyter o Databricks.
3. Ejecuta las celdas en orden.
4. Revisa el `run_id`, las métricas y la URI del modelo que se imprimen al final.

El dataset debe tener las columnas `Id`, `Species` y una o más columnas numéricas predictoras. El notebook detiene la ejecución con un mensaje explícito si faltan columnas, hay nulos o no existe el archivo.

## Configuración

La configuración está concentrada al comienzo de `iris.ipynb`. Cada valor tiene una responsabilidad específica:

- `IRIS_DATA_PATH`: ruta del archivo `Iris.csv`. Es la entrada de datos del proyecto y se utiliza únicamente para leer el dataset. Puede ser una ruta local, una ruta montada en Databricks o una ruta absoluta. No es la ubicación donde se guardan modelos ni reportes.
- `MLFLOW_TRACKING_URI`: dirección de la instalación de MLflow que recibe los experimentos, runs, parámetros y métricas. Ejemplos: `databricks`, `http://servidor:5000` o una configuración local. No es una ruta del dataset.
- `IRIS_EXPERIMENT_NAME`: nombre del experimento de MLflow. El experimento es el contenedor lógico que agrupa múltiples entrenamientos.
- `IRIS_ARTIFACT_LOCATION`: ubicación donde MLflow guarda archivos producidos por los runs, como el modelo, el reporte JSON y la matriz de confusión. Es distinta de `MLFLOW_TRACKING_URI`.
- `IRIS_MODEL_NAME`: nombre descriptivo del entrenamiento o modelo dentro del run. No despliega el modelo automáticamente.
- `IRIS_DATASET_VERSION`: identificador manual del dataset. Debe cambiarse cuando se reemplaza, limpia o regenera el CSV.
- `IRIS_PROJECT_VERSION`: versión del código o notebook que produjo el resultado.
- `IRIS_AUTHOR`: responsable de la ejecución, guardado como metadato.
- `IRIS_PURPOSE`: motivo de la ejecución, por ejemplo `baseline-classification`.

Los parámetros del algoritmo también están centralizados: `TEST_SIZE` controla cuánto se reserva para prueba, `RANDOM_STATE` permite repetir el experimento, `N_ESTIMATORS` define cuántos árboles usa Random Forest, `MAX_DEPTH` limita la profundidad de los árboles y `PRIMARY_METRIC` define qué métrica se usará como referencia principal.

### Configuración recomendada en Databricks

El notebook detecta automáticamente si está ejecutándose en Databricks. En ese caso:

- Crea widgets para las variables de configuración, de modo que las rutas puedan cambiarse desde la interfaz sin editar el código.
- Usa `databricks` como tracking URI para conectarse al servidor administrado de MLflow del workspace.
- Usa `/Shared/iris_mlflow` como experimento predeterminado. Si se necesita otro lugar, define `IRIS_EXPERIMENT_NAME` con una ruta de workspace que el usuario pueda escribir.
- Usa por defecto `/Volumes/workspace/my_data/my_volumen/Iris.csv` como ruta del dataset, basada en la ubicación original del proyecto. Si el catálogo, esquema o volumen son distintos, cambia el widget `IRIS_DATA_PATH`.
- Deja vacía la ubicación de artefactos para que Databricks use el almacenamiento administrado por MLflow. Solo define `IRIS_ARTIFACT_LOCATION` si el usuario tiene permisos de escritura en la ubicación elegida.

Para un volumen de Unity Catalog, la ruta de lectura debe tener la forma `/Volumes/<catalogo>/<esquema>/<volumen>/Iris.csv`. Para una ubicación de artefactos de MLflow en un volumen, la URI debe tener la forma `dbfs:/Volumes/<catalogo>/<esquema>/<volumen>/<directorio>`, y el usuario necesita permisos sobre el experimento y el volumen.

El notebook también funciona fuera de Databricks: en ese caso usa `Iris.csv` en la carpeta actual, el tracking URI predeterminado de MLflow y el almacenamiento de artefactos configurado localmente.

Los valores del modelo están centralizados en el notebook:

| Parámetro | Valor predeterminado | Propósito |
|---|---:|---|
| `TEST_SIZE` | `0.20` | Porción reservada para prueba |
| `RANDOM_STATE` | `42` | Reproducibilidad |
| `N_ESTIMATORS` | `100` | Número de árboles |
| `MAX_DEPTH` | `4` | Profundidad máxima |
| `PRIMARY_METRIC` | `test_f1_weighted` | Métrica principal |

Las rutas y metadatos pueden configurarse mediante variables de entorno:

```bash
set IRIS_DATA_PATH=/ruta/a/Iris.csv
set MLFLOW_TRACKING_URI=databricks
set IRIS_EXPERIMENT_NAME=/Shared/iris_mlflow
set IRIS_ARTIFACT_LOCATION=dbfs:/ruta/de/artefactos
set IRIS_MODEL_NAME=iris-random-forest
set IRIS_DATASET_VERSION=iris-csv-v1
set IRIS_PROJECT_VERSION=1.0.0
set IRIS_AUTHOR=equipo-ml
set IRIS_PURPOSE=baseline-classification
```

En PowerShell usa `$env:NOMBRE = "valor"`. En Databricks también pueden usarse widgets o la configuración del entorno. Si no se define `MLFLOW_TRACKING_URI`, MLflow usa su configuración predeterminada. Si no se define la ubicación de artefactos, se utiliza la configuración del backend de MLflow.

## MLflow explicado para principiantes

MLflow guarda qué modelo se entrenó, con qué configuración, qué resultados obtuvo y qué archivos generó.

### Tracking URI

El tracking URI indica a qué instalación o servidor se conecta MLflow para guardar ejecuciones. En local puede apuntar a una carpeta o a un servidor; en Databricks normalmente se usa el MLflow integrado. Es distinto de la ubicación de artefactos: el tracking URI conecta con MLflow, mientras que la ubicación de artefactos guarda archivos como el modelo, reportes e imágenes.

### Experimento

Un experimento agrupa runs relacionados. El notebook busca el experimento por nombre y lo crea solamente si no existe. Cambiar los hiperparámetros no reemplaza el experimento: genera otro run dentro del mismo grupo.

### Run

Un run es un entrenamiento concreto y tiene un `run_id` único. Dentro del bloque `mlflow.start_run(...)` se registran todos los datos de ese entrenamiento. El `run_id` permite recuperar sus métricas, parámetros, artefactos y modelo.

### Parámetros, métricas y tags

- **Parámetros**: configuración usada, por ejemplo `n_estimators`, `max_depth`, `test_size` y `random_state`. Responden: “¿con qué configuración se obtuvo este resultado?”.
- **Métricas**: resultados medidos después del entrenamiento. Se registran accuracy, precision, recall y F1 para entrenamiento y prueba.
- **Tags**: etiquetas de contexto, como el tipo de modelo, la versión del proyecto, el dataset o el propósito. No son métricas ni configuración del algoritmo.

### Artefactos y modelo

El notebook registra como artefactos el reporte de clasificación y la matriz de confusión. También guarda el Random Forest en la ruta `modelo` mediante `mlflow.sklearn.log_model()`.

Registrar el modelo no significa desplegarlo en producción. Significa guardar una copia reproducible que después puede cargarse con una URI como `runs:/<run_id>/modelo`. La firma del modelo describe las columnas y tipos de entrada esperados; `input_example` muestra una entrada válida pequeña.

| Concepto | Significado |
|---|---|
| Experimento | Agrupa entrenamientos relacionados |
| Run | Un entrenamiento específico |
| Parámetro | Configuración usada |
| Métrica | Resultado obtenido |
| Tag | Contexto descriptivo |
| Artefacto | Archivo generado |
| Modelo | Objeto entrenado que puede reutilizarse |

## Qué revisar al terminar una ejecución

El notebook imprime:

- Nombre e identificador del experimento.
- `run_id`.
- Métrica principal.
- URI del modelo.
- Ubicación de artefactos.

En la interfaz de MLflow, compara runs del mismo experimento revisando primero la misma versión del dataset y luego la métrica principal. No compares métricas de experimentos distintos sin verificar que usan los mismos datos y criterios.

## Métricas

Las métricas de entrenamiento muestran cómo se comportó el modelo con datos que ya vio. Las métricas de prueba muestran cómo funciona con datos reservados y son las principales para comparar modelos.

Se registran:

- `train_accuracy` y `test_accuracy`.
- `train_precision_weighted` y `test_precision_weighted`.
- `train_recall_weighted` y `test_recall_weighted`.
- `train_f1_weighted` y `test_f1_weighted`.
- Reporte detallado por clase.
- Matriz de confusión.

## Problemas frecuentes

| Problema | Solución |
|---|---|
| No existe el dataset | Define `IRIS_DATA_PATH` con una ruta válida |
| Faltan columnas | Verifica que existan `Id`, `Species` y columnas numéricas |
| Hay valores nulos | Limpia el dataset antes de ejecutar |
| MLflow no conecta | Revisa `MLFLOW_TRACKING_URI` y las credenciales del entorno |
| No se puede guardar el modelo | Revisa permisos y `IRIS_ARTIFACT_LOCATION` |
| Aparece otro run al repetir | Es normal: cada ejecución se conserva para comparar resultados |
| Los resultados no son comparables | Verifica dataset, versión, semilla y configuración |
| No se puede cargar el modelo | Comprueba que la URI tenga el `run_id` correcto y que el artefacto exista |

## Estructura actual

```text
.
├── iris.ipynb
└── readme.md
```

El notebook continúa siendo el punto principal de ejecución y los archivos existentes mantienen sus formatos.
