"""Unit tests for idempotent Unity Catalog feature-table preparation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_mlflow_utils.constants import FEATURE_TABLE_COLUMNS, FEATURE_TABLE_TYPES
from iris_mlflow_utils.feature_table import (
    _ensure_primary_key,
    _validate_feature_dataframe,
    _validate_table_name,
    ensure_feature_table,
)

pytestmark = pytest.mark.unit


class Query:
    def __init__(self, count: int) -> None:
        self.value = count

    def limit(self, value: int) -> "Query":
        return self

    def count(self) -> int:
        return self.value

    def filter(self, expression: object) -> "Query":
        return self


class Column:
    def cast(self, data_type: str) -> "Column":
        return self

    def alias(self, name: str) -> "Column":
        return self

    def isNull(self) -> str:  # noqa: N802 - mirrors Spark API
        return "is-null"


class Writer:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.saved = ""

    def format(self, name: str) -> "Writer":
        assert name == "delta"
        return self

    def mode(self, name: str) -> "Writer":
        assert name == "error"
        return self

    def saveAsTable(self, name: str) -> None:  # noqa: N802 - mirrors Spark API
        if self.failure:
            raise self.failure
        self.saved = name


class Frame:
    def __init__(
        self,
        *,
        columns: tuple[str, ...] = FEATURE_TABLE_COLUMNS,
        types: dict[str, str] | None = None,
        validation_counts: list[int] | None = None,
        duplicate_count: int = 0,
        writer: Writer | None = None,
    ) -> None:
        self.columns = list(columns)
        observed_types = types or FEATURE_TABLE_TYPES
        self.schema = [
            SimpleNamespace(
                name=name, dataType=SimpleNamespace(simpleString=lambda value=value: value)
            )
            for name, value in observed_types.items()
        ]
        self.validation_counts = iter(validation_counts or [0, 0, 0])
        self.duplicate_count = duplicate_count
        self.write = writer or Writer()

    def __getitem__(self, name: str) -> Column:
        return Column()

    def select(self, *columns: object) -> "Frame":
        return self

    def filter(self, expression: object) -> Query:
        return Query(next(self.validation_counts))

    def groupBy(self, name: str) -> "Grouped":  # noqa: N802 - mirrors Spark API
        return Grouped(self.duplicate_count)


class Grouped:
    def __init__(self, duplicate_count: int) -> None:
        self.duplicate_count = duplicate_count

    def count(self) -> Query:
        return Query(self.duplicate_count)


class Reader:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame

    def option(self, name: str, value: object) -> "Reader":
        return self

    def csv(self, path: str) -> Frame:
        return self.frame


class Spark:
    def __init__(
        self, frame: Frame, *, exists: bool, ddl: str = "", alter_error: bool = False
    ) -> None:
        self.frame = frame
        self.catalog = SimpleNamespace(tableExists=lambda name: exists)
        self.read = Reader(frame)
        self.ddl = ddl
        self.alter_error = alter_error
        self.statements: list[str] = []

    def table(self, name: str) -> Frame:
        return self.frame

    def sql(self, statement: str) -> SimpleNamespace:
        self.statements.append(statement)
        if statement.startswith("SHOW CREATE"):
            return SimpleNamespace(collect=lambda: [[self.ddl]])
        if self.alter_error:
            raise PermissionError(statement)
        return SimpleNamespace(collect=lambda: [])


def test_validate_table_name_requires_three_levels() -> None:
    _validate_table_name("workspace.default.iris")
    with pytest.raises(ValueError, match="tres niveles"):
        _validate_table_name("invalid")


@pytest.mark.parametrize(
    "frame, error, message",
    [
        (Frame(columns=tuple(FEATURE_TABLE_COLUMNS[:-1])), ValueError, "no contiene columnas"),
        (Frame(types={**FEATURE_TABLE_TYPES, "Id": "string"}), TypeError, "debe ser bigint"),
        (Frame(validation_counts=[1, 0, 0]), ValueError, "Id nulo"),
        (Frame(validation_counts=[0, 1, 0]), ValueError, "numéricas"),
        (Frame(duplicate_count=1), ValueError, "Id duplicado"),
        (Frame(validation_counts=[0, 0, 1]), ValueError, "Species nulo"),
    ],
)
def test_validate_feature_dataframe_rejects_invalid_contract(
    frame: Frame, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _validate_feature_dataframe(frame, "workspace.default.iris")


def test_ensure_primary_key_skips_existing_constraint() -> None:
    spark = Spark(Frame(), exists=True, ddl="PRIMARY KEY (Id)")
    _ensure_primary_key(spark, "workspace.default.iris")
    assert len(spark.statements) == 1


def test_ensure_primary_key_adds_constraint_or_reports_permission_error() -> None:
    spark = Spark(Frame(), exists=True)
    _ensure_primary_key(spark, "workspace.default.iris")
    assert any("SET NOT NULL" in statement for statement in spark.statements)
    assert any("ADD CONSTRAINT" in statement for statement in spark.statements)

    denied = Spark(Frame(), exists=True, alter_error=True)
    with pytest.raises(RuntimeError, match="permisos"):
        _ensure_primary_key(denied, "workspace.default.iris")


def test_ensure_feature_table_validates_existing_table() -> None:
    assert (
        ensure_feature_table(
            Spark(Frame(), exists=True, ddl="PRIMARY KEY (Id)"),
            table_name="workspace.default.iris",
            dataset_path=Path("unused.csv"),
        )
        is False
    )


def test_ensure_feature_table_creates_new_table() -> None:
    frame = Frame()
    spark = Spark(frame, exists=False)
    assert (
        ensure_feature_table(
            spark, table_name="workspace.default.iris", dataset_path=Path("iris.csv")
        )
        is True
    )
    assert frame.write.saved == "workspace.default.iris"


def test_ensure_feature_table_rejects_missing_source_columns() -> None:
    frame = Frame(columns=tuple(FEATURE_TABLE_COLUMNS[:-1]))
    with pytest.raises(ValueError, match="columnas requeridas"):
        ensure_feature_table(
            Spark(frame, exists=False),
            table_name="workspace.default.iris",
            dataset_path=Path("iris.csv"),
        )


def test_ensure_feature_table_handles_parallel_creation() -> None:
    frame = Frame(
        writer=Writer(RuntimeError("TABLE_OR_VIEW_ALREADY_EXISTS")),
        validation_counts=[0, 0, 0, 0, 0, 0],
    )
    spark = Spark(frame, exists=False, ddl="PRIMARY KEY (Id)")
    assert (
        ensure_feature_table(
            spark, table_name="workspace.default.iris", dataset_path=Path("iris.csv")
        )
        is False
    )


def test_ensure_feature_table_propagates_unexpected_write_error() -> None:
    frame = Frame(writer=Writer(RuntimeError("storage unavailable")))
    with pytest.raises(RuntimeError, match="storage unavailable"):
        ensure_feature_table(
            Spark(frame, exists=False),
            table_name="workspace.default.iris",
            dataset_path=Path("iris.csv"),
        )
