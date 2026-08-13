# Contribuir

1. Crea una rama desde `dev`.
2. Instala dependencias con `uv sync --project tools --locked --dev`.
3. Ejecuta `uv run --project tools pytest`, Ruff, MyPy y el build.
4. No incluyas salidas de notebooks, bases MLflow, sesiones Jupyter ni secretos.
5. Abre un pull request hacia `dev`; la promoción a `main` representa una versión productiva.
