"""Silver: typed, validated, de-duplicated transactions, maintained incrementally with Delta MERGE.

- Types are enforced; rows that can't be trusted (unparseable timestamp, missing stock code, zero quantity,
  negative price) go to a quarantine table with the reason, instead of being dropped or loaded.
- Each line gets a deterministic `line_id` (hash of its content), and new batches are MERGEd on it, so re-running
  a day, or an upstream re-send of the same lines, never double-counts revenue.
- Flags rather than filters: returns (invoice numbers starting with "C"), non-product lines (postage, fees,
  manual adjustments) and anonymous sales (no customer id) stay in silver and are flagged; gold decides what to use.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from lakehouse.config import TABLES
from lakehouse.util import table_exists

NON_PRODUCT_CODES = ["POST", "D", "M", "BANK CHARGES", "AMAZONFEE", "DOT", "CRUK", "S", "PADS", "B", "C2"]


def typed(bronze: DataFrame) -> DataFrame:
    """Parse and flag raw bronze rows. Pure function, unit-tested on small frames."""
    # try_* variants: under Spark 4's ANSI mode a plain cast of a bad value fails the whole batch;
    # here it becomes NULL and the row is quarantined with a reason instead
    df = bronze.select(
        F.trim("InvoiceNo").alias("invoice_no"),
        F.upper(F.trim("StockCode")).alias("stock_code"),
        F.trim("Description").alias("description"),
        F.col("Quantity").try_cast("int").alias("quantity"),
        F.try_to_timestamp("InvoiceDate", F.lit("yyyy-MM-dd HH:mm:ss")).alias("invoice_ts"),
        F.col("UnitPrice").try_cast("decimal(12,2)").alias("unit_price"),
        F.col("CustomerID").try_cast("bigint").alias("customer_id"),
        F.trim("Country").alias("country"),
        "_source_file", "_landing_date",
    )
    df = (df.withColumn("invoice_date", F.to_date("invoice_ts"))
            .withColumn("invoice_month", F.date_format("invoice_ts", "yyyy-MM"))
            .withColumn("is_return", F.col("invoice_no").startswith("C"))
            .withColumn("is_product", ~F.col("stock_code").isin(NON_PRODUCT_CODES))
            .withColumn("line_amount", (F.col("quantity") * F.col("unit_price")).cast("decimal(14,2)")))
    reason = (F.when(F.col("invoice_ts").isNull(), "unparseable timestamp")
               .when(F.col("stock_code").isNull() | (F.col("stock_code") == ""), "missing stock code")
               .when(F.col("quantity").isNull() | (F.col("quantity") == 0), "zero or missing quantity")
               .when(F.col("unit_price").isNull() | (F.col("unit_price") < 0), "negative or missing price")
               .when(F.to_date("invoice_ts") != F.col("_landing_date"), "timestamp outside landing day"))
    df = df.withColumn("quarantine_reason", reason)
    key = F.concat_ws("|", "invoice_no", "stock_code", "description", "quantity", F.col("invoice_ts").cast("string"),
                      F.col("unit_price").cast("string"), F.coalesce(F.col("customer_id").cast("string"), F.lit("")))
    return df.withColumn("line_id", F.sha2(key, 256))


SILVER_COLUMNS = ["line_id", "invoice_no", "stock_code", "description", "quantity", "invoice_ts", "invoice_date",
                  "invoice_month", "unit_price", "line_amount", "customer_id", "country", "is_return", "is_product",
                  "_source_file"]


def upsert(spark: SparkSession, days: list[str]) -> dict:
    from delta.tables import DeltaTable

    bronze = spark.read.format("delta").load(str(TABLES["bronze"])).where(F.col("_landing_date").isin(days))
    t = typed(bronze).cache()
    rows_in = t.count()
    bad = t.where(F.col("quarantine_reason").isNotNull())
    good = t.where(F.col("quarantine_reason").isNull())
    batch = good.dropDuplicates(["line_id"]).select(*SILVER_COLUMNS)
    n_bad = bad.count()
    n_good = good.count()
    n_batch = batch.count()
    if n_bad:
        (bad.select("line_id", "quarantine_reason", "invoice_no", "stock_code", "quantity", "unit_price",
                    "invoice_ts", "customer_id", "_source_file", F.current_timestamp().alias("quarantined_at"))
            .write.format("delta").mode("append").save(str(TABLES["quarantine"])))
    months = [r[0] for r in batch.select("invoice_month").distinct().collect()]
    if not table_exists(spark, TABLES["silver"]):
        batch.write.format("delta").partitionBy("invoice_month").save(str(TABLES["silver"]))
        inserted = n_batch
    else:
        target = DeltaTable.forPath(spark, str(TABLES["silver"]))
        before = target.history(1).select("version").collect()[0][0]
        month_list = ",".join(f"'{m}'" for m in months) or "''"
        (target.alias("t")
            .merge(batch.alias("s"), f"t.invoice_month IN ({month_list}) AND t.invoice_month = s.invoice_month "
                                     "AND t.line_id = s.line_id")
            .whenNotMatchedInsertAll()
            .execute())
        last = target.history(1).select("version", "operationMetrics").collect()[0]
        # a MERGE that changes nothing doesn't commit a new version, so only read metrics from a new commit
        inserted = int(last[1].get("numTargetRowsInserted", 0)) if last[0] > before else 0
    t.unpersist()
    return {"rows_in": rows_in, "quarantined": n_bad, "duplicates_in_batch": n_good - n_batch,
            "already_in_silver": n_batch - inserted, "inserted": inserted, "months": sorted(months)}
