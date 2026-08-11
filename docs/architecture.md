# Arquitectura

El ciclo separa entrenamiento, evaluación, aprobación y despliegue. Los
notebooks de entrenamiento registran versiones como `Challenger`; los
notebooks de `deployment/` gestionan la promoción posterior.

```mermaid
flowchart LR
    A{Runtime} -->|Databricks| B[Tabla Delta iris_features]
    A -->|Local| L[CSV de desarrollo]
    B --> C[Entrenamiento Spark MLflow UC]
    L --> M[Entrenamiento local MLflow SQLite]
    C --> N[Challenger]
    M --> N
    N --> D[Evaluación y artefactos]
    D --> E{Gates de calidad}
    E -->|fallo| F[Conservar Champion]
    E -->|ok| G[Aprobación]
    G -->|Databricks| H[Model Serving]
    G -->|Local| I[Despliegue simulado]
    H --> J[Smoke test]
    I --> J
    J -->|fallo| F
    J -->|éxito| K[Alias Champion]
```

La rama local comparte evaluación, gates, aliases y artefactos, pero reemplaza
Unity Catalog y Model Serving por SQLite y un manifiesto auditable. Sólo la
rama Databricks es productiva.

Responsabilidades:

- Entrenamiento: datos, modelo, evaluación inicial y registro Challenger.
- Evaluation: métricas, artefactos y gates contra Champion.
- Approval: validación del tag de aprobación de Unity Catalog.
- Deployment: actualización del endpoint, smoke test y alias Champion.
- Create job: creación y conexión del Deployment Job.

MLflow Tracking conserva runs, métricas y artefactos. Unity Catalog conserva el
modelo, sus versiones, tags, descripciones y aliases. Lakeflow Jobs orquesta
las tareas. Model Serving expone exactamente la versión aprobada.

Las utilidades se encuentran en `tools/src/iris_mlflow_utils`: `config.py`,
`data.py`, `evaluation.py`, `deployment.py`, `registry.py` y `serving.py`.
