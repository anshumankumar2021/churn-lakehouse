"""Bronze: land raw CSV drops in Delta exactly as received, plus lineage columns.

Every column stays a string (schema-on-read), so a malformed upstream value is kept and can be inspected rather
than silently dropped. Ingestion is idempotent: files already in bronze (by `_source_file`) are skipped, so a
re-run or a retried job never duplicates a day.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from lakehouse.config import LANDING, TABLES
from lakehouse.util import table_exists

RAW_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
RAW_SCHEMA = StructType([StructField(c, StringType()) for c in RAW_COLUMNS])


def landing_file(day: str):
    return LANDING / f"date={day}" / "transactions.csv"


def ingested_files(spark: SparkSession) -> set[str]:
    if not table_exists(spark, TABLES["bronze"]):
        return set()
    return {r[0] for r in spark.read.format("delta").load(str(TABLES["bronze"])).select("_source_file").distinct().collect()}


def read_landing(spark: SparkSession, days: list[str]) -> DataFrame:
    paths = [str(landing_file(d)) for d in days]
    return (spark.read.csv(paths, header=True, schema=RAW_SCHEMA, mode="PERMISSIVE", multiLine=False, escape='"')
            .withColumn("_source_file", F.regexp_extract(F.input_file_name(), r"(date=[0-9-]+/[^/]+)$", 1))
            .withColumn("_landing_date", F.to_date(F.regexp_extract("_source_file", r"date=([0-9-]+)", 1)))
            .withColumn("_ingested_at", F.current_timestamp()))


def ingest(spark: SparkSession, days: list[str]) -> dict:
    done = ingested_files(spark)
    todo = [d for d in days if f"date={d}/transactions.csv" not in done and landing_file(d).exists()]
    if not todo:
        return {"days": 0, "rows": 0, "skipped_days": len(days)}
    df = read_landing(spark, todo)
    n = df.count()
    (df.write.format("delta").mode("append").partitionBy("_landing_date").save(str(TABLES["bronze"])))
    return {"days": len(todo), "rows": n, "skipped_days": len(days) - len(todo)}
