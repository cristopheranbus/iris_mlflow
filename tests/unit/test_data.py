"""Unit tests for dataset loading and validation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from iris_mlflow_utils.data import (
    get_delta_table_version,
    load_dataset,
    load_dataset_for_runtime,
    load_dataset_frame,
    load_dataset_from_spark,
)

pytestmark = pytest.mark.unit


def iris_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6],
            "SepalLengthCm": [5.1, 5.0, 6.4, 6.3, 6.7, 6.8],
            "SepalWidthCm": [3.5, 3.4, 3.2, 3.3, 3.1, 3.0],
            "PetalLengthCm": [1.4, 1.5, 4.5, 4.7, 5.6, 5.5],
            "PetalWidthCm": [0.2, 0.2, 1.5, 1.6, 2.4, 2.1],
            "Species": ["setosa", "setosa", "versicolor", "versicolor", "virginica", "virginica"],
        }
    )


def test_load_dataset_validates_and_encodes_labels(tmp_path: Path) -> None:
    path = tmp_path / "iris.csv"
    iris_frame().to_csv(path, index=False)
    bundle = load_dataset(path)
    assert bundle.classes == ("setosa", "versicolor", "virginica")
    assert np.array_equal(bundle.target, np.array([0, 0, 1, 1, 2, 2]))
    assert bundle.id_column == "Id"


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No existe"):
        load_dataset(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    "mutator, error, message",
    [
        (lambda frame: frame.iloc[0:0], ValueError, "vacío"),
        (lambda frame: frame.drop(columns="Species"), ValueError, "objetivo"),
        (lambda frame: frame.assign(Species="setosa"), ValueError, "dos clases"),
        (lambda frame: frame.assign(SepalLengthCm=np.nan), ValueError, "nulos"),
        (lambda frame: frame.assign(Id=[1, 1, 3, 4, 5, 6]), ValueError, "duplicados"),
        (lambda frame: frame.drop(columns=["SepalWidthCm"]), ValueError, "contrato Iris"),
        (lambda frame: frame.assign(SepalLengthCm="bad"), TypeError, "numéricas"),
        (lambda frame: frame.assign(SepalLengthCm=np.inf), ValueError, "infinitos"),
    ],
)
def test_load_dataset_frame_rejects_invalid_data(
    mutator: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        load_dataset_frame(mutator(iris_frame()))  # type: ignore[operator]


def test_load_dataset_frame_rejects_missing_predictors() -> None:
    frame = pd.DataFrame({"Species": ["setosa", "versicolor"]})
    with pytest.raises(ValueError, match="predictoras"):
        load_dataset_frame(frame)


def test_load_dataset_frame_accepts_dataset_without_id() -> None:
    bundle = load_dataset_frame(iris_frame().drop(columns="Id"))
    assert bundle.id_column is None


@pytest.mark.parametrize(
    "spark, name, message",
    [(None, "a.b.c", "sesión Spark"), (object(), "bad", "catalog.schema.table")],
)
def test_get_delta_table_version_validates_input(spark: object, name: str, message: str) -> None:
    with pytest.raises((RuntimeError, ValueError), match=message):
        get_delta_table_version(spark, name)


def test_get_delta_table_version_handles_empty_and_malformed_history() -> None:
    empty = SimpleNamespace(sql=lambda query: SimpleNamespace(collect=lambda: []))
    with pytest.raises(RuntimeError, match="historial"):
        get_delta_table_version(empty, "workspace.default.features")

    malformed = SimpleNamespace(sql=lambda query: SimpleNamespace(collect=lambda: [{"other": 1}]))
    with pytest.raises(RuntimeError, match="determinar"):
        get_delta_table_version(malformed, "workspace.default.features")


def test_get_delta_table_version_returns_latest_version() -> None:
    spark = SimpleNamespace(sql=lambda query: SimpleNamespace(collect=lambda: [{"version": 7}]))
    assert get_delta_table_version(spark, "workspace.default.features") == "7"


class FakeSparkFrame:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.columns = list(frame.columns)

    def select(self, *columns: str) -> "FakeSparkFrame":
        return FakeSparkFrame(self.frame[list(columns)])

    def toPandas(self) -> pd.DataFrame:
        return self.frame


def test_load_dataset_from_spark_reads_latest_and_exact_version() -> None:
    frame = iris_frame()

    class Reader:
        def option(self, name: str, value: int) -> "Reader":
            assert (name, value) == ("versionAsOf", 3)
            return self

        def table(self, name: str) -> FakeSparkFrame:
            return FakeSparkFrame(frame)

    spark = SimpleNamespace(
        catalog=SimpleNamespace(tableExists=lambda name: True),
        read=Reader(),
        table=lambda name: FakeSparkFrame(frame),
    )
    assert len(load_dataset_from_spark(spark, table_name="w.d.iris").dataframe) == 6
    assert (
        len(load_dataset_from_spark(spark, table_name="w.d.iris", table_version="3").dataframe) == 6
    )


def test_load_dataset_from_spark_rejects_missing_table_reader_and_columns() -> None:
    with pytest.raises(ValueError, match="catalog.schema.table"):
        load_dataset_from_spark(object(), table_name="invalid")

    missing_table = SimpleNamespace(catalog=SimpleNamespace(tableExists=lambda name: False))
    with pytest.raises(FileNotFoundError, match="No existe"):
        load_dataset_from_spark(missing_table, table_name="w.d.iris")

    no_reader = SimpleNamespace(catalog=SimpleNamespace(tableExists=lambda name: True), read=None)
    with pytest.raises(RuntimeError, match="spark.read"):
        load_dataset_from_spark(no_reader, table_name="w.d.iris", table_version="3")

    incomplete = FakeSparkFrame(iris_frame().drop(columns="PetalWidthCm"))
    spark = SimpleNamespace(
        catalog=SimpleNamespace(tableExists=lambda name: True), table=lambda name: incomplete
    )
    with pytest.raises(ValueError, match="no contiene columnas"):
        load_dataset_from_spark(spark, table_name="w.d.iris")


def test_load_dataset_from_spark_accepts_table_without_optional_id() -> None:
    frame = iris_frame().drop(columns="Id")
    spark = SimpleNamespace(
        catalog=SimpleNamespace(tableExists=lambda name: True),
        table=lambda name: FakeSparkFrame(frame),
    )
    bundle = load_dataset_from_spark(spark, table_name="w.d.iris")
    assert bundle.id_column is None


def test_runtime_loader_routes_local_and_databricks(tmp_path: Path) -> None:
    path = tmp_path / "iris.csv"
    iris_frame().to_csv(path, index=False)
    config = SimpleNamespace(
        dataset_path=path,
        target_column="Species",
        feature_table="w.d.iris",
        feature_table_version="",
    )
    assert len(load_dataset_for_runtime("local", spark=None, config=config).dataframe) == 6

    with pytest.raises(RuntimeError, match="Spark activa"):
        load_dataset_for_runtime("databricks", spark=None, config=config)

    spark = SimpleNamespace(
        catalog=SimpleNamespace(tableExists=lambda name: True),
        table=lambda name: FakeSparkFrame(iris_frame()),
    )
    assert (
        len(
            load_dataset_for_runtime(
                "databricks", spark=spark, config=config, table_version=""
            ).dataframe
        )
        == 6
    )

    config.dataset_path = None
    with pytest.raises(ValueError, match="dataset_path"):
        load_dataset_for_runtime("local", spark=None, config=config)
