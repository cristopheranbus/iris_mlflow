# Artefactos y evaluación del modelo

La evaluación usa el conjunto de test y registra métricas, tablas y gráficos
en MLflow bajo `evaluation/`.

- La matriz de confusión muestra errores absolutos; la normalizada muestra
  proporciones por clase.
- ROC se calcula one-vs-rest para cada especie y como agregados macro/micro.
- Precision-Recall es útil para revisar el equilibrio entre falsos positivos y
  cobertura por clase.
- Lift compara la concentración de positivos frente a una selección aleatoria.
- Cumulative gain muestra cuántos positivos se capturan al ordenar por score.
- Feature importance muestra la contribución relativa de las cuatro features
  para Random Forest y XGBoost.
- Las métricas macro pesan todas las clases por igual; weighted pondera por
  soporte.

El gate inicial exige `test_f1_weighted >= 0.90`,
`test_accuracy >= 0.90` y una degradación máxima de `0.01` frente a Champion.
Estos valores pertenecen ahora a `config/promotion.prod.toml` y pueden cambiarse
sin modificar Python. El entrenamiento registra además cross-validation
estratificada repetida, intervalos de confianza normales, Brier score, error de
calibración y linaje del dataset mediante `mlflow.log_input()`.

Los artefactos de evaluación se generan con la misma utilidad en modo local y
Databricks: matriz de confusión, ROC, Precision-Recall, lift, cumulative gain,
probabilidades, importancia de variables, predicciones y esquemas.

Las métricas canónicas comunes a Random Forest y XGBoost incluyen accuracy,
precision, recall y F1 macro/weighted, además de log loss, ROC-AUC OvR y average
precision. Las métricas adicionales del evaluador de cada framework no forman
parte del contrato usado por los gates.

El run de evaluación se crea en el experimento configurado y su identificador
queda guardado como `evaluation_run_id` en la versión. Cuando MLflow expone un
`model_id`, las métricas también se registran contra ese Logged Model para que
sean visibles desde la versión registrada.
