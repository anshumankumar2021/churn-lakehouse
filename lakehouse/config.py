"""Paths and the Spark session (local mode, Delta Lake enabled)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LAKEHOUSE_DATA", ROOT / "data"))
LANDING = DATA / "landing"          # daily CSV drops, as an upstream system would deliver them
LAKE = DATA / "lake"                # Delta tables
TABLES = {
    "bronze": LAKE / "bronze" / "transactions",
    "silver": LAKE / "silver" / "transactions",
    "quarantine": LAKE / "silver" / "quarantine",
    "dq": LAKE / "ops" / "data_quality",
    "runs": LAKE / "ops" / "pipeline_runs",
    "features": LAKE / "gold" / "customer_features",
    "scores": LAKE / "gold" / "churn_scores",
}
DELTA_VERSION = "4.0.1"


def spark_session(app: str = "churn-lakehouse", shuffle_partitions: int = 8):
    """Local Spark with Delta. Jars come from LAKEHOUSE_JARS (a directory, for machines without Maven access)
    or from Maven Central via spark.jars.packages."""
    from pyspark.sql import SparkSession

    b = (SparkSession.builder.appName(app).master(os.environ.get("SPARK_MASTER", "local[*]"))
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.sql.shuffle.partitions", shuffle_partitions)
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "3g"))
         .config("spark.ui.enabled", "false")
         .config("spark.databricks.delta.snapshotPartitions", 2)
         .config("spark.sql.adaptive.enabled", "true"))
    jars = os.environ.get("LAKEHOUSE_JARS")
    if jars:
        b = b.config("spark.jars", ",".join(str(p) for p in sorted(Path(jars).glob("*.jar"))))
    else:
        b = b.config("spark.jars.packages", f"io.delta:delta-spark_2.13:{DELTA_VERSION}")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
