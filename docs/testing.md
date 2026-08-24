# Estrategia y catálogo de pruebas

Este documento concentra la estrategia completa de pruebas. La suite está
dividida en tres tipos con responsabilidades distintas:

| Suite | Propósito | Dependencias reales |
|---|---|---|
| Unitarias | Validar una unidad de lógica y todos sus límites de forma rápida y aislada. | Ninguna red, Databricks ni MLflow persistente. |
| Integración | Comprobar que varios componentes locales funcionan juntos. | MLflow con SQLite, filesystem temporal y dataset local. |
| Contratos | Impedir que cambie accidentalmente la estructura operativa del repositorio. | Archivos versionados; no ejecuta infraestructura. |

La regla para ubicar una prueba nueva es:

- Si todas las dependencias externas están sustituidas por dobles, pertenece a `unit`.
- Si usa implementaciones reales locales como MLflow, SQLite o el CSV canónico, pertenece a `integration`.
- Si inspecciona notebooks, YAML, Markdown, CI o convenciones del repositorio, pertenece a `contracts`.

## Ejecución completa

Desde la raíz del proyecto:

```powershell
uv run --group test pytest --cov=iris_mlflow_utils --cov-branch --cov-report=term-missing --cov-report=json:.local/coverage/coverage.json --cov-fail-under=85
uv run --group test python quality/check_coverage.py .local/coverage/coverage.json --min-statements 90 --min-branches 85
uv run --group quality ruff check --config quality/ruff.toml src tests quality ops
uv run --group quality ruff format --check --config quality/ruff.toml src tests quality ops
uv run --group test --group quality mypy --config-file quality/mypy.ini src tests quality ops
uv run --group quality pip-audit --ignore-vuln PYSEC-2026-3552
uv build
uv run --no-sync python quality/smoke_wheel.py --python 3.12
```

La suite mantiene un piso combinado de 85% y exige por separado al menos 90% de
sentencias y 85% de ramas. Ninguna prueba automática requiere Databricks.

## Ejecución por tipo

```powershell
uv run --group test pytest -m unit
uv run --group test pytest -m integration
uv run --group test pytest -m contract
```

Los marcadores están registrados en `pyproject.toml`. Cada módulo declara
su marcador a nivel de archivo, por lo que una prueba no depende de su nombre para
ser seleccionada.

## Pruebas unitarias

Las pruebas unitarias validan funciones y decisiones aisladas. Utilizan objetos
simulados para Spark, Databricks SDK, Registry y HTTP; no abren conexiones de red
ni escriben en un backend real de MLflow.

### Para qué sirven

- Detectar rápidamente errores de lógica y validación.
- Cubrir límites, excepciones y caminos de rollback difíciles de reproducir en cloud.
- Comprobar contratos internos de payloads y respuestas.
- Permitir refactors con retroalimentación rápida y determinista.

### Detalle

| Archivo | Qué valida | Riesgo que reduce |
|---|---|---|
| [`test_config.py`](../tests/unit/test_config.py) | Precedencia entre TOML, entorno y widgets; nombres, rangos, aliases, rutas y configuración por runtime. | Configuración inválida o distinta entre local y Databricks. |
| [`test_data.py`](../tests/unit/test_data.py) | Carga CSV/Spark, contrato de features, nulos, duplicados, tipos, versiones Delta y selección por runtime. | Entrenar con datos incompatibles o con un snapshot incorrecto. |
| [`test_deployment.py`](../tests/unit/test_deployment.py) | Gates de calidad, métricas no finitas, create/update, snapshot, rollback y estados del endpoint. | Promover un modelo degradado o dejar serving en un estado inconsistente. |
| [`test_evaluation.py`](../tests/unit/test_evaluation.py) | Métricas, probabilidades, tablas, ROC, lift, cumulative gain y feature importance. | Publicar evidencia de evaluación incompleta o con formato inestable. |
| [`test_feature_table.py`](../tests/unit/test_feature_table.py) | Creación idempotente, esquema Spark, nulos, duplicados, carreras y clave primaria. | Sobrescribir o aceptar una tabla Unity Catalog inválida. |
| [`test_local_deployment.py`](../tests/unit/test_local_deployment.py) | Aprobación local, promoción, rollback disponible y escritura del manifiesto. | Que la simulación local difiera del ciclo de gobierno esperado. |
| [`test_preflight.py`](../tests/unit/test_preflight.py) | Comandos del CLI de Databricks, errores de cuenta, grants, identidad OIDC y validación completa previa al bundle. | Iniciar un despliegue con una cuenta, identidad o permisos incorrectos. |
| [`test_registry.py`](../tests/unit/test_registry.py) | Tags, descripciones, aliases, verificación y métricas enlazadas a runs. | Perder trazabilidad entre modelo, evaluación y estado de promoción. |
| [`test_runtime.py`](../tests/unit/test_runtime.py) | Detección local/Databricks y precedencia de parámetros de entorno y widgets. | Ejecutar una rama incorrecta o ignorar parámetros del Job. |
| [`test_serving.py`](../tests/unit/test_serving.py) | URL, configuración, payload, errores HTTP/red, JSON, DataFrames y protección de secretos. | Fallos silenciosos del cliente o exposición accidental de credenciales. |

Esta suite debe permanecer rápida, no requerir credenciales y poder ejecutarse
sin acceso a internet. Las nuevas dependencias externas deben sustituirse en el
límite del módulo, no mediante simulaciones de detalles internos de terceros.

Agregar una prueba unitaria cuando se incorpora una regla, validación, rama de
error, transformación o interacción que pueda representarse con entradas y
dobles controlados.

## Pruebas de integración local

Las pruebas de integración ejercitan componentes reales que pueden ejecutarse
sin infraestructura cloud. Usan MLflow con un backend SQLite temporal, el
filesystem temporal de pytest y el dataset local versionado.

### Para qué sirven

- Detectar incompatibilidades entre el proyecto y una versión real de MLflow.
- Comprobar persistencia y restauración de experimentos.
- Verificar que datos, evaluación, aprobación y manifiesto funcionan juntos.
- Mantener un flujo representativo sin coste ni credenciales de Databricks.

### Detalle

| Archivo | Qué valida | Riesgo que reduce |
|---|---|---|
| [`test_mlflow_local.py`](../tests/integration/test_mlflow_local.py) | Creación, reutilización y restauración de experimentos; evaluación real de un clasificador con MLflow y SQLite. | Que los dobles unitarios oculten cambios incompatibles de MLflow. |
| [`test_local_pipeline.py`](../tests/integration/test_local_pipeline.py) | Carga del CSV canónico, entrenamiento, evaluación, aprobación, promoción y manifiesto local. | Que módulos correctos por separado fallen cuando se conectan en el flujo local. |

El fixture [`conftest.py`](../tests/integration/conftest.py) restaura el tracking
URI y finaliza cualquier run abierto para impedir contaminación entre pruebas.
Las bases SQLite, runs y manifiestos deben crearse bajo `tmp_path`. Esta suite no
puede usar tokens, endpoints, tablas Unity Catalog ni recursos reales de
Databricks.

Agregar una integración cuando el comportamiento solo sea confiable usando una
implementación local real o conectando varios módulos. Las pruebas cloud deben
permanecer en un proceso manual o workflow protegido.

## Pruebas de contratos

Las pruebas de contratos inspeccionan los artefactos versionados que conectan el
código con CI, Databricks y la operación del proyecto. No ejecutan notebooks ni
crean infraestructura.

### Para qué sirven

- Detectar cambios accidentales en el DAG y en la separación dev/prod.
- Mantener notebooks limpios, compilables y centrados en utilidades compartidas.
- Evitar secretos persistentes o mecanismos de autenticación prohibidos.
- Verificar enlaces, configuración de ejemplo y la propia política de cobertura.

### Detalle

| Archivo | Qué valida | Riesgo que reduce |
|---|---|---|
| [`test_bundle.py`](../tests/contracts/test_bundle.py) | Estructura YAML del bundle, tareas, dependencias, targets, permisos y conexión del Deployment Job. | Desplegar un DAG incorrecto o mezclar recursos de desarrollo y producción. |
| [`test_configuration.py`](../tests/contracts/test_configuration.py) | Coherencia de `local.env.example` y nombres usados en el flujo local. | Entregar instrucciones locales que no coincidan con la configuración real. |
| [`test_coverage_policy.py`](../tests/contracts/test_coverage_policy.py) | Aceptación y rechazo independiente de los umbrales de sentencias y ramas. | Que una métrica alta oculte una caída de la otra. |
| [`test_notebooks.py`](../tests/contracts/test_notebooks.py) | Helpers obligatorios, APIs prohibidas, inputs dinámicos, orden del rollback, outputs vacíos y compilación. | Duplicar lógica, persistir estado o perder salvaguardas de despliegue. |
| [`test_repository.py`](../tests/contracts/test_repository.py) | Enlaces relativos e inclusión de estados generados en `.gitignore`. | Documentación rota o artefactos locales versionados accidentalmente. |
| [`test_workflows.py`](../tests/contracts/test_workflows.py) | Orden, nombres, jobs, triggers, permisos, concurrencia y acciones fijadas de GitHub Actions. | Perder checks, protecciones o condiciones de despliegue al reorganizar CI/CD. |

Los helpers de [`conftest.py`](../tests/contracts/conftest.py) centralizan la raíz
del repositorio y la extracción del código fuente de notebooks.

YAML y JSON deben analizarse estructuralmente cuando la estructura sea la regla.
Las búsquedas de texto se reservan para prohibiciones de seguridad o presencia
de llamadas concretas. Una prueba no debe fallar únicamente por cambios de
espacios o formato.

Agregar un contrato cuando una modificación válida de Python todavía podría
romper el empaquetado, la operación, el workflow o la forma de un notebook. No
usar esta suite para sustituir pruebas funcionales del código.

## Criterios globales

- La suite completa debe ser independiente del orden de ejecución.
- Los tests no deben usar credenciales locales ni depender de variables heredadas.
- `tests/conftest.py` limpia las variables `IRIS_*`, `MLFLOW_*` y `DATABRICKS_*`.
- Los artefactos se escriben en `tmp_path` o bajo `.local/`, que está ignorado.
- Un error debe probarse por su tipo y por una parte estable del mensaje, no por el texto completo.
- La cobertura debe priorizar decisiones y caminos de fallo; no se agregan assertions triviales solo para aumentar el porcentaje.

## Validaciones fuera de pytest

Ruff comprueba estilo y formato; mypy valida tipos; `pip-audit` revisa
dependencias y `uv build` confirma que el paquete continúa siendo distribuible.
Las políticas de Ruff, mypy y cobertura están centralizadas en `quality/`.

La validación cloud usa cómputo serverless y requiere cuenta activa, OIDC,
modelos y permisos; el proyecto no configura clusters.

GitHub ejecuta además un escaneo de secretos sobre el historial completo. Los notebooks
deben conservar `execution_count=null` y no guardar outputs.
