# Contribuir

1. Crea una rama desde `dev`.
2. Instala dependencias con `uv sync --locked --all-groups`.
3. Ejecuta pytest, Ruff, mypy y el build desde la raíz según `docs/development.md`.
4. No incluyas salidas de notebooks, bases MLflow, sesiones Jupyter ni secretos.
5. Abre un pull request hacia `dev`; la promoción a `main` representa una versión productiva.
