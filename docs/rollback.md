# Rollback

1. Identificar la versión anterior en el Activity Log o en el historial del
   endpoint.
2. Ejecutar una actualización de Model Serving apuntando a esa versión exacta.
3. Esperar estado `READY`.
4. Ejecutar el smoke test.
5. Mover el alias `Champion` a la versión restaurada.
6. Registrar causa, versión retirada, versión restaurada y resultado de la
   inferencia.

No se eliminan versiones de Unity Catalog durante un rollback. Si el endpoint
no queda listo, se conserva la configuración anterior y se escala el incidente.
