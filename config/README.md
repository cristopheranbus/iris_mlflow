# Configuración

La configuración de entrenamiento conserva las variables que ya consumen los
notebooks. La precedencia es: variables de entorno, widgets de Databricks y
valores predeterminados.

Los archivos de esta carpeta son documentación y ejemplos. No deben contener
tokens, secretos ni credenciales reales.

- `training.toml`: runtime, modelos y serving.
- `promotion.dev.toml` y `promotion.prod.toml`: gates declarativos de Champion.
- `monitoring.toml`: ventana y thresholds de observabilidad alert-only.
