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

Los artefactos de evaluación se generan con la misma utilidad en modo local y
Databricks: matriz de confusión, ROC, Precision-Recall, lift, cumulative gain,
probabilidades, importancia de variables, predicciones y esquemas.
