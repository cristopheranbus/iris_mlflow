"""Canonical data-contract constants shared by loaders and feature tables."""

from __future__ import annotations

FEATURE_COLUMNS = (
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
)
FEATURE_TABLE_COLUMNS = ("Id", *FEATURE_COLUMNS, "Species")
FEATURE_TABLE_TYPES = {
    "Id": "bigint",
    "SepalLengthCm": "double",
    "SepalWidthCm": "double",
    "PetalLengthCm": "double",
    "PetalWidthCm": "double",
    "Species": "string",
}
