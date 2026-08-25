"""Aggregate a Unity AI Gateway inference table and emit an alert-only decision."""

from __future__ import annotations

import argparse
import json
import re

from iris_mlflow_utils.monitoring import (
    MonitoringSnapshot,
    evaluate_monitoring,
    load_monitoring_config,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--inference-table", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+", arguments.inference_table):
        raise ValueError("--inference-table debe usar catalog.schema.table.")
    from pyspark.sql import SparkSession  # type: ignore[import-not-found]
    from pyspark.sql import functions as functions

    config = load_monitoring_config()
    spark = SparkSession.builder.getOrCreate()
    source = spark.table(arguments.inference_table).where(
        functions.col("event_time")
        >= functions.current_timestamp() - functions.expr(f"INTERVAL {config.window_hours} HOURS")
    )
    aggregate = source.agg(
        functions.countDistinct("invocation_id").alias("observations"),
        functions.avg(
            functions.when(functions.col("status_code").between(200, 399), 0.0).otherwise(1.0)
        ).alias("error_rate"),
        functions.percentile_approx("latency_ms", 0.95).alias("p95_latency_ms"),
        functions.sum(
            functions.when(functions.size("logging_error_codes") > 0, 1).otherwise(0)
        ).alias("logging_errors"),
    ).first()
    observations = int(aggregate["observations"] or 0)
    snapshot = MonitoringSnapshot(
        endpoint_name=arguments.endpoint_name,
        observations=observations,
        error_rate=float(aggregate["error_rate"] or 0.0),
        p95_latency_ms=float(aggregate["p95_latency_ms"] or 0.0),
        logging_errors=int(aggregate["logging_errors"] or 0),
    )
    decision = evaluate_monitoring(snapshot, config)
    print("IRIS_MONITOR_DECISION=" + json.dumps(decision.as_dict(), separators=(",", ":")))
    return 2 if decision.alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
