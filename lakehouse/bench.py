"""Benchmarks, measured on this machine (see README for the hardware caveat).

1. Incremental vs full: processing one day of new data with MERGE, versus rebuilding silver from all of bronze.
2. Layout: customer lookups on a scaled copy of silver (the real rows repeated with offset customer ids, written
   as many small files, the way frequent incremental writes leave a table) before and after
   OPTIMIZE ... ZORDER BY (customer_id). Reports files actually read (data skipping) and query time.
"""
from __future__ import annotations

import random
import statistics
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lakehouse.config import LAKE, TABLES
from lakehouse.silver import SILVER_COLUMNS, typed
from lakehouse.util import scan_metrics

TARGET_FILE_BYTES = 32 * 1024 * 1024


def full_rebuild_seconds(spark: SparkSession) -> dict:
    out = LAKE / "bench" / "silver_full_rebuild"
    t = time.perf_counter()
    b = spark.read.format("delta").load(str(TABLES["bronze"]))
    df = typed(b).where(F.col("quarantine_reason").isNull()).dropDuplicates(["line_id"]).select(*SILVER_COLUMNS)
    df.write.format("delta").mode("overwrite").partitionBy("invoice_month").save(str(out))
    secs = time.perf_counter() - t
    rows = spark.read.format("delta").load(str(out)).count()
    return {"seconds": round(secs, 2), "rows": rows}


def _lookups(spark, path, ids, repeats=1):
    times, files, bytes_ = [], [], []
    for cid in ids:
        q = spark.read.format("delta").load(str(path)).where(F.col("customer_id") == cid).agg(F.sum("line_amount"), F.count("*"))
        t = time.perf_counter()
        m = scan_metrics(q)
        times.append(time.perf_counter() - t)
        files.append(m["files"])
        bytes_.append(m["bytes"])
    return {"median_ms": round(statistics.median(times) * 1000, 1), "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))] * 1000, 1),
            "median_files_read": statistics.median(files), "median_mb_read": round(statistics.median(bytes_) / 1e6, 2)}


def zorder(spark: SparkSession, factor: int = 20, n_files: int = 400, n_queries: int = 25) -> dict:
    from delta.tables import DeltaTable

    path = LAKE / "bench" / "transactions_scaled"
    silver = spark.read.format("delta").load(str(TABLES["silver"])).where(F.col("customer_id").isNotNull())
    scaled = (silver.crossJoin(spark.range(factor).withColumnRenamed("id", "copy"))
              .withColumn("customer_id", F.col("customer_id") + F.col("copy") * 100000)
              .drop("copy").repartition(n_files))
    scaled.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(str(path))
    detail = lambda: DeltaTable.forPath(spark, str(path)).detail().select("numFiles", "sizeInBytes").collect()[0]
    rows = spark.read.format("delta").load(str(path)).count()
    ids = [r[0] for r in spark.read.format("delta").load(str(path)).select("customer_id").distinct().collect()]
    rng = random.Random(7)
    sample = rng.sample(ids, n_queries)
    _lookups(spark, path, sample[:3])                       # warm-up
    before_files, before_size = detail()
    before = _lookups(spark, path, sample)
    # Delta's default target is 1 GB per file, which would put this whole table in one file and leave nothing to
    # skip; 32 MB is a typical target for tables that are read by key
    prev = spark.conf.get("spark.databricks.delta.optimize.maxFileSize", None)
    spark.conf.set("spark.databricks.delta.optimize.maxFileSize", str(TARGET_FILE_BYTES))
    t = time.perf_counter()
    DeltaTable.forPath(spark, str(path)).optimize().executeZOrderBy("customer_id")
    optimize_s = time.perf_counter() - t
    if prev is None:
        spark.conf.unset("spark.databricks.delta.optimize.maxFileSize")
    else:
        spark.conf.set("spark.databricks.delta.optimize.maxFileSize", prev)
    after_files, after_size = detail()
    _lookups(spark, path, sample[:3])
    after = _lookups(spark, path, sample)
    return {"rows": rows, "copies": factor, "queries": n_queries, "optimize_seconds": round(optimize_s, 1),
            "target_file_mb": TARGET_FILE_BYTES // (1024 * 1024),
            "before": {"files": before_files, "mb": round(before_size / 1e6, 1), **before},
            "after": {"files": after_files, "mb": round(after_size / 1e6, 1), **after}}
