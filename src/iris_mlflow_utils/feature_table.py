"""Idempotent Unity Catalog feature-table preparation for Databricks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .constants import FEATURE_TABLE_COLUMNS, FEATURE_TABLE_TYPES

_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")


def ensure_feature_table(
    spark: Any,
    *,
    table_name: str,
    dataset_path: Path,
) -> bool:
    """Create or validate a Unity Catalog Delta feature table.

    Returns whether the table was created. Existing tables are never
    overwritten. Invalid schemas, null keys, duplicate keys, or missing
    primary-key constraints raise a descriptive error.
    """

    _validate_table_name(table_name)
    table_exists = spark.catalog.tableExists(table_name)  # Nombre UC catalog.schema.table.
    if table_exists:
        feature_df = spark.table(table_name)  # Lee la tabla validada sin modificarla.
        _validate_feature_dataframe(feature_df, table_name)
        _ensure_primary_key(spark, table_name)
        return False

    source_df = (
        spark.read.option("header", True)  # La primera fila contiene los nombres de columnas.
        .option("inferSchema", False)  # Los tipos se fuerzan explícitamente después.
        .csv(str(dataset_path))  # Ruta del CSV en el volumen Databricks.
    )
    missing = set(FEATURE_TABLE_COLUMNS) - set(source_df.columns)
    if missing:
        raise ValueError(
            f"El dataset '{dataset_path}' no contiene columnas requeridas: {sorted(missing)}."
        )
    source_df = source_df.select(*FEATURE_TABLE_COLUMNS)
    feature_df = source_df.select(
        source_df["Id"].cast("bigint").alias("Id"),
        source_df["SepalLengthCm"].cast("double").alias("SepalLengthCm"),
        source_df["SepalWidthCm"].cast("double").alias("SepalWidthCm"),
        source_df["PetalLengthCm"].cast("double").alias("PetalLengthCm"),
        source_df["PetalWidthCm"].cast("double").alias("PetalWidthCm"),
        source_df["Species"].cast("string").alias("Species"),
    )
    _validate_feature_dataframe(feature_df, table_name)
    try:
        (
            feature_df.write.format("delta")  # Formato requerido para la tabla de features.
            .mode("error")  # Evita sobrescribir una tabla creada en paralelo.
            .saveAsTable(table_name)  # Nombre completo de la tabla en Unity Catalog.
        )
    except Exception as exc:
        if "TABLE_OR_VIEW_ALREADY_EXISTS" not in str(exc):
            raise
        # La tabla fue creada en paralelo; valida sin sobrescribir.
        feature_df = spark.table(table_name)
        _validate_feature_dataframe(feature_df, table_name)
        _ensure_primary_key(spark, table_name)
        return False
    _ensure_primary_key(spark, table_name)
    return True


def _validate_table_name(table_name: str) -> None:
    if not _TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(
            "La tabla de features debe usar un nombre de tres niveles catalog.schema.table."
        )


def _validate_feature_dataframe(dataframe: Any, table_name: str) -> None:
    columns = set(dataframe.columns)
    missing = set(FEATURE_TABLE_COLUMNS) - columns
    if missing:
        raise ValueError(f"La tabla {table_name} no contiene columnas: {sorted(missing)}")

    schema = {field.name: field.dataType.simpleString() for field in dataframe.schema}
    for column, expected_type in FEATURE_TABLE_TYPES.items():
        actual_type = schema.get(column)
        if actual_type != expected_type:
            raise TypeError(
                f"La columna {table_name}.{column} debe ser {expected_type}; "
                f"se encontró {actual_type}."
            )

    if dataframe.filter(dataframe["Id"].isNull()).limit(1).count():
        raise ValueError(f"La tabla {table_name} contiene Id nulo.")
    numeric_columns = [
        column for column in FEATURE_TABLE_COLUMNS if column not in {"Id", "Species"}
    ]
    if (
        dataframe.filter(
            " OR ".join(f"{column} IS NULL OR isnan({column})" for column in numeric_columns)
        )
        .limit(1)
        .count()
    ):
        raise ValueError(f"La tabla {table_name} contiene features numéricas nulas o inválidas.")
    if dataframe.groupBy("Id").count().filter("count > 1").limit(1).count():
        raise ValueError(f"La tabla {table_name} contiene Id duplicado.")
    if dataframe.filter(dataframe["Species"].isNull()).limit(1).count():
        raise ValueError(f"La tabla {table_name} contiene Species nulo.")


def _ensure_primary_key(spark: Any, table_name: str) -> None:
    ddl = "\n".join(
        row[0]
        for row in spark.sql(  # Consulta DDL para verificar la clave primaria existente.
            f"SHOW CREATE TABLE {table_name}"
        ).collect()
    )
    if "PRIMARY KEY" in ddl.upper():
        return
    try:
        spark.sql(  # La clave debe ser no nula antes de declarar la PK.
            f"ALTER TABLE {table_name} ALTER COLUMN Id SET NOT NULL"
        )
        spark.sql(  # Restricción informativa requerida por Feature Engineering en UC.
            f"ALTER TABLE {table_name} ADD CONSTRAINT iris_features_pk PRIMARY KEY (Id)"
        )
    except Exception as error:
        raise RuntimeError(
            f"La tabla {table_name} no tiene una clave primaria sobre Id y "
            "no fue posible agregarla. Verifica permisos y propiedad de la tabla."
        ) from error
