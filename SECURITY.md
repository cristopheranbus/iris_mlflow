# Seguridad

No publiques vulnerabilidades ni credenciales en issues. Repórtalas mediante la opción
**Security advisories** del repositorio. Incluye impacto, pasos de reproducción y versión afectada.

Las credenciales de Databricks se administran con OIDC; nunca deben almacenarse en archivos,
notebooks, variables versionadas ni artefactos de MLflow.

## Excepciones temporales

`PYSEC-2026-3552` se ignora temporalmente porque MLflow 3.15 restringe la dependencia
`cryptography` por debajo de la versión corregida. La excepción debe retirarse cuando
MLflow publique una versión compatible; cualquier otro advisory continúa bloqueando CI.
