"""Dataset loading, validation, encoding, and reproducible splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-untyped]


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

    feature_columns = [
        column for column in dataframe.columns if column not in {id_column, target_column}
    ]
    if not feature_columns:
        raise ValueError("No se encontraron columnas predictoras.")
    non_numeric = dataframe[feature_columns].select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise TypeError(f"Las columnas predictoras deben ser numéricas: {non_numeric}.")

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
