# Rollback

El notebook de despliegue captura las entidades servidas y la configuración de
tráfico antes de modificar el endpoint. Si falla la espera de `READY` o el smoke
test, restaura ese snapshot, vuelve a esperar `READY` y deja `Champion` sin
cambios.

Tags de evidencia en la versión candidata:

- `deployment_status=failed`.
- `smoke_test_status=failed`.
- `rollback_status=restored` si la restauración terminó correctamente.
- `rollback_status=failed` si también falló la restauración.

Ante `rollback_status=failed`:

1. Detener nuevas promociones.
2. Identificar la versión anterior en el alias `Champion` y en el historial del endpoint.
3. Actualizar Model Serving a la versión exacta o configuración anterior.
4. Esperar estado `READY` y ejecutar el smoke test.
5. Confirmar que `Champion` sigue apuntando a la versión estable.
6. Registrar causa, versión retirada, versión restaurada y respuesta de inferencia.

No se eliminan versiones de Unity Catalog durante un rollback. La promoción del
alias ocurre sólo después del smoke test, por lo que un rollback normal no
requiere mover `Champion`.
