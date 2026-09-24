"""Gold: one row per customer per monthly snapshot, with behavioural features and a churn label.

For snapshot date S, a customer is included if they bought (a product, not a return) before S. Features only use
transactions before S; the label is whether they bought again in the 90 days after S:

    churned = no purchase in (S, S + 90 days]

Only snapshots whose 90-day label window ends before the data does are labelled. The gold build records the
silver Delta version it read, so any snapshot can be rebuilt exactly with time travel.
"""
from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from lakehouse.config import TABLES

HORIZON_DAYS = 90
FEATURES = ["recency_days", "frequency", "monetary", "tenure_days", "avg_order_value", "distinct_products",
            "orders_90d", "spend_90d", "spend_prev_90d", "spend_trend", "avg_days_between_orders",
            "return_rate", "is_uk", "orders_per_month", "avg_items_per_order", "overdue_ratio", "active_months",
            "orders_30d", "active_month_share"]


def snapshot(silver: DataFrame, s: dt.date, label_until: dt.date | None) -> DataFrame:
    """Features as of snapshot date s (exclusive), and the churn label if its window is fully observed."""
    S = F.lit(s.isoformat()).cast("date")
    known = silver.where(F.col("customer_id").isNotNull() & F.col("is_product") & (F.col("invoice_date") < S))
    buys = known.where(~F.col("is_return") & (F.col("quantity") > 0))
    rets = known.where(F.col("is_return"))
    days_ago = F.datediff(S, "invoice_date")
    per = buys.groupBy("customer_id").agg(
        F.datediff(S, F.max("invoice_date")).alias("recency_days"),
        F.countDistinct("invoice_no").alias("frequency"),
        F.sum("line_amount").cast("double").alias("monetary"),
        F.datediff(S, F.min("invoice_date")).alias("tenure_days"),
        F.countDistinct("stock_code").alias("distinct_products"),
        F.countDistinct(F.when(days_ago <= 90, F.col("invoice_no"))).alias("orders_90d"),
        F.countDistinct(F.when(days_ago <= 30, F.col("invoice_no"))).alias("orders_30d"),
        F.countDistinct(F.date_format("invoice_date", "yyyy-MM")).alias("active_months"),
        F.sum(F.when(days_ago <= 90, F.col("line_amount")).otherwise(0)).cast("double").alias("spend_90d"),
        F.sum(F.when((days_ago > 90) & (days_ago <= 180), F.col("line_amount")).otherwise(0)).cast("double").alias("spend_prev_90d"),
        F.sum("quantity").alias("items"),
        F.max(F.when(F.col("country") == "United Kingdom", 1).otherwise(0)).alias("is_uk"),
    )
    returned = rets.groupBy("customer_id").agg(F.sum(F.abs("quantity")).alias("returned_items"))
    f = (per.join(returned, "customer_id", "left").fillna({"returned_items": 0})
            .withColumn("avg_order_value", F.col("monetary") / F.col("frequency"))
            .withColumn("spend_trend", (F.col("spend_90d") - F.col("spend_prev_90d")) / (F.col("spend_prev_90d") + 1))
            .withColumn("avg_days_between_orders",
                        F.when(F.col("frequency") > 1, F.col("tenure_days") / (F.col("frequency") - 1)).otherwise(F.col("tenure_days")))
            .withColumn("return_rate", F.col("returned_items") / (F.col("items") + F.col("returned_items")))
            .withColumn("orders_per_month", F.col("frequency") / F.greatest(F.col("tenure_days") / 30.0, F.lit(1.0)))
            .withColumn("avg_items_per_order", F.col("items") / F.col("frequency"))
            # how late the next order is compared with this customer's own rhythm (1.0 = right on schedule)
            .withColumn("overdue_ratio", F.least(F.col("recency_days") / F.greatest(F.col("avg_days_between_orders"), F.lit(7.0)), F.lit(10.0)))
            .withColumn("active_month_share", F.col("active_months") / F.greatest(F.ceil(F.col("tenure_days") / 30.4), F.lit(1)))
            .withColumn("snapshot_date", S))
    if label_until is None:
        return f.withColumn("churned", F.lit(None).cast("int"))
    end = F.date_add(S, HORIZON_DAYS)
    future = (silver.where(F.col("customer_id").isNotNull() & F.col("is_product") & ~F.col("is_return")
                           & (F.col("quantity") > 0) & (F.col("invoice_date") >= S) & (F.col("invoice_date") < end))
              .select("customer_id").distinct().withColumn("bought_again", F.lit(1)))
    return (f.join(future, "customer_id", "left")
             .withColumn("churned", F.when(F.col("bought_again").isNull(), 1).otherwise(0)).drop("bought_again"))


def build(spark: SparkSession, snapshots: list[dt.date], score_date: dt.date, silver_version: int | None = None) -> dict:
    reader = spark.read.format("delta")
    if silver_version is not None:
        reader = reader.option("versionAsOf", silver_version)
    silver = reader.load(str(TABLES["silver"])).cache()
    last = silver.agg(F.max("invoice_date")).collect()[0][0]
    frames = []
    for s in snapshots:
        labelled = s + dt.timedelta(days=HORIZON_DAYS) <= last
        frames.append(snapshot(silver, s, last if labelled else None))
    frames.append(snapshot(silver, score_date, None))
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    cols = ["customer_id", "snapshot_date", *FEATURES, "churned"]
    out.select(*cols).write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .partitionBy("snapshot_date").save(str(TABLES["features"]))
    silver.unpersist()
    g = spark.read.format("delta").load(str(TABLES["features"]))
    summary = {str(r["snapshot_date"]): {"customers": r["n"], "churn_rate": None if r["rate"] is None else round(r["rate"], 4)}
               for r in g.groupBy("snapshot_date").agg(F.count("*").alias("n"), F.avg("churned").alias("rate")).collect()}
    return {"last_transaction_date": str(last), "snapshots": dict(sorted(summary.items()))}
