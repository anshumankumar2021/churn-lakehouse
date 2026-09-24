"""Data-quality gates, run after every silver batch. Results are appended to a Delta table (ops/data_quality).

Critical checks stop the pipeline (the batch is already committed, but nothing downstream is built from it and
the run is marked failed); warnings are recorded for trend monitoring.
"""
from __future__ import annotations

import datetime as dt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lakehouse.config import TABLES


class DataQualityError(RuntimeError):
    pass


def evaluate(stats: dict, silver_stats: dict) -> list[dict]:
    """Pure: turn batch counts into check results (unit-tested without Spark)."""
    rows_in = max(stats["rows_in"], 1)
    accounted = stats["quarantined"] + stats["duplicates_in_batch"] + stats["already_in_silver"] + stats["inserted"]
    day_rows = max(silver_stats["rows_on_days"], 1)     # silver rows on the batch's days (new or already there)
    checks = [
        ("every input row is accounted for", "critical", accounted, stats["rows_in"], accounted == stats["rows_in"]),
        ("line_id unique in silver", "critical", silver_stats["duplicate_line_ids"], 0, silver_stats["duplicate_line_ids"] == 0),
        ("no nulls in required silver columns", "critical", silver_stats["null_required"], 0, silver_stats["null_required"] == 0),
        ("quarantined share of batch", "warning", round(stats["quarantined"] / rows_in, 4), 0.01, stats["quarantined"] / rows_in <= 0.01),
        ("anonymous (no customer) share", "warning", round(silver_stats["anonymous_on_days"] / day_rows, 4), 0.40,
         silver_stats["anonymous_on_days"] / day_rows <= 0.40),
        ("return lines share", "warning", round(silver_stats["returns_on_days"] / day_rows, 4), 0.05,
         silver_stats["returns_on_days"] / day_rows <= 0.05),
    ]
    return [{"check": c, "severity": s, "value": float(v), "threshold": float(t), "passed": bool(p)} for c, s, v, t, p in checks]


def silver_stats(spark: SparkSession, months: list[str], days: list[str]) -> dict:
    s = spark.read.format("delta").load(str(TABLES["silver"])).where(F.col("invoice_month").isin(months))
    row = s.agg(
        (F.count("*") - F.countDistinct("line_id")).alias("duplicate_line_ids"),
        F.sum(F.when(F.col("invoice_ts").isNull() | F.col("stock_code").isNull() | F.col("quantity").isNull(), 1).otherwise(0)).alias("null_required"),
        F.sum(F.when(F.col("invoice_date").isin(days), 1).otherwise(0)).alias("rows_on_days"),
        F.sum(F.when(F.col("invoice_date").isin(days) & F.col("customer_id").isNull(), 1).otherwise(0)).alias("anonymous_on_days"),
        F.sum(F.when(F.col("invoice_date").isin(days) & F.col("is_return"), 1).otherwise(0)).alias("returns_on_days"),
    ).collect()[0].asDict()
    return {k: int(v or 0) for k, v in row.items()}


def gate(spark: SparkSession, run_id: str, days: list[str], stats: dict) -> list[dict]:
    results = evaluate(stats, silver_stats(spark, stats["months"] or ["-"], days))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    rows = [{**r, "run_id": run_id, "first_day": min(days), "last_day": max(days), "checked_at": now} for r in results]
    spark.createDataFrame(rows).write.format("delta").mode("append").save(str(TABLES["dq"]))
    failed = [r["check"] for r in results if r["severity"] == "critical" and not r["passed"]]
    if failed:
        raise DataQualityError(f"critical data-quality checks failed for {min(days)}..{max(days)}: {failed}")
    return results
