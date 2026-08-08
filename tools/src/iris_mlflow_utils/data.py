"""Dataset loading, validation, encoding, and reproducible splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]

from .constants import FEATURE_COLUMNS


@dataclass(frozen=True)
class DatasetBundle:
    """Validated data ready for a classification split."""

    dataframe: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray
    feature_columns: tuple[str, ...]
    classes: tuple[str, ...]
    target_column: str
    id_column: str | None


def load_dataset_from_spark(
    spark: Any,
    *,
    table_name: str,
    table_version: str = "",
    target_column: str = "Species",
    id_column: str = "Id",
) -> DatasetBundle:
    """Load the canonical dataset directly from a Unity Catalog Delta table.

    The table is validated before conversion to pandas. ``table_version`` can
    pin a Delta snapshot for reproducible training.
    """

    if not table_name or table_name.count(".") != 2:
        raise ValueError("table_name debe usar catalog.schema.table.")
    catalog = getattr(spark, "catalog", None)
    if catalog is not None and not catalog.tableExists(table_name):
        raise FileNotFoundError(f"No existe la tabla Delta '{table_name}'.")
    reader = getattr(spark, "read", None)
    if table_version:
        if reader is None:
            raise RuntimeError("La sesión Spark no expone spark.read para leer una versión Delta.")
        spark_dataframe = reader.option("versionAsOf", int(table_version)).table(table_name)
    else:
        spark_dataframe = spark.table(table_name)
    expected_columns = list(FEATURE_COLUMNS) + [target_column]
    if id_column in getattr(spark_dataframe, "columns", []):
        expected_columns.insert(0, id_column)
    missing = [column for column in expected_columns if column not in spark_dataframe.columns]
    if missing:
        raise ValueError(f"La tabla '{table_name}' no contiene columnas: {missing}.")
    return load_dataset_frame(
        spark_dataframe.select(*expected_columns).toPandas(),
        target_column=target_column,
        id_column=id_column,
    )


def load_dataset(
    path: Path,
    *,
    target_column: str = "Species",
    id_column: str = "Id",
) -> DatasetBundle:
    """Load and validate the Iris CSV, encoding labels consistently."""

    if not path.is_file():
        raise FileNotFoundError(f"No existe el dataset en '{path}'.")

    dataframe = pd.read_csv(path)
    return load_dataset_frame(
        dataframe,
        target_column=target_column,
        id_column=id_column,
    )


def load_dataset_frame(
    dataframe: pd.DataFrame,
    *,
    target_column: str = "Species",
    id_column: str = "Id",
) -> DatasetBundle:
    """Validate a pandas dataset already loaded from a table or file."""

    dataframe = dataframe.copy()
    if dataframe.empty:
        raise ValueError("El dataset está vacío.")
    if target_column not in dataframe.columns:
        raise ValueError(f"Falta la columna objetivo '{target_column}'.")
    if dataframe[target_column].nunique() < 2:
        raise ValueError("El objetivo debe contener al menos dos clases.")
    if dataframe.isna().any().any():
        null_columns = dataframe.columns[dataframe.isna().any()].tolist()
        raise ValueError(f"Hay valores nulos en: {null_columns}.")
    if id_column in dataframe.columns and dataframe[id_column].duplicated().any():
        raise ValueError(f"La columna '{id_column}' contiene valores duplicados.")

    feature_columns = [
        column for column in dataframe.columns if column not in {id_column, target_column}
    ]
    if not feature_columns:
        raise ValueError("No se encontraron columnas predictoras.")
    if tuple(feature_columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Las columnas predictoras deben coincidir con el contrato Iris: "
            f"{list(FEATURE_COLUMNS)}. Se encontraron: {feature_columns}."
        )
    non_numeric = dataframe[feature_columns].select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise TypeError(f"Las columnas predictoras deben ser numéricas: {non_numeric}.")
    if not np.isfinite(dataframe[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Las columnas predictoras no pueden contener valores infinitos.")

    encoder = LabelEncoder()
    target = encoder.fit_transform(dataframe[target_column].astype(str))
    return DatasetBundle(
        dataframe=dataframe,
        features=dataframe[feature_columns].copy(),
        target=target,
        feature_columns=tuple(feature_columns),
        classes=tuple(str(value) for value in encoder.classes_),
        target_column=target_column,
        id_column=id_column if id_column in dataframe.columns else None,
    )
