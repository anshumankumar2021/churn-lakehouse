"""Run the lakehouse end to end and write results/results.json.

    python -m scripts.run_pipeline                 # backfill + daily increments + gold + model + benchmarks
    python -m scripts.run_pipeline --skip-bench    # faster

Simulates production: history up to BACKFILL_UNTIL arrives as one backfill batch, then each later day arrives
as its own daily drop and is processed incrementally (bronze append -> silver MERGE -> quality gate). A second pass
over already-loaded days checks that re-running is a no-op.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
import uuid

from pyspark.sql import functions as F

from lakehouse import bench, bronze, gold, model, quality, silver
from lakehouse.config import LANDING, ROOT, TABLES, spark_session
from lakehouse.util import version

BACKFILL_UNTIL = "2011-10-31"
SNAPSHOTS = [dt.date(2011, m, 1) for m in range(3, 10)]    # Mar..Sep 2011, each with a full 90-day label window
SCORE_DATE = dt.date(2011, 12, 10)                         # "today": the day after the last transaction


def days_available() -> list[str]:
    return sorted(p.name.split("=", 1)[1] for p in LANDING.glob("date=*"))


def process(spark, days, run_id, kind, log):
    t = time.perf_counter()
    b = bronze.ingest(spark, days)
    tb = time.perf_counter() - t
    s = silver.upsert(spark, days)
    ts = time.perf_counter() - t - tb
    checks = quality.gate(spark, run_id, days, s)
    total = time.perf_counter() - t
    entry = {"kind": kind, "first_day": days[0], "last_day": days[-1], "days": len(days), "bronze_rows": b["rows"],
             "silver_inserted": s["inserted"], "quarantined": s["quarantined"], "duplicates": s["duplicates_in_batch"],
             "already_loaded": s["already_in_silver"], "bronze_s": round(tb, 2), "silver_s": round(ts, 2),
             "total_s": round(total, 2), "warnings": [c["check"] for c in checks if not c["passed"]],
             "silver_version": version(spark, TABLES["silver"])}
    log.append(entry)
    print(json.dumps(entry), flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    spark = spark_session()
    run_id = uuid.uuid4().hex[:8]
    all_days = days_available()
    backfill = [d for d in all_days if d <= BACKFILL_UNTIL]
    daily = [d for d in all_days if d > BACKFILL_UNTIL]
    log: list[dict] = []
    process(spark, backfill, run_id, "backfill", log)
    for d in daily:
        process(spark, [d], run_id, "daily", log)
    # idempotency: replaying the last five days must change nothing
    replay = process(spark, daily[-5:], run_id, "replay", log)
    assert replay["silver_inserted"] == 0, "re-running loaded days inserted rows"
    # a re-sent file under a new name would be caught by MERGE on line_id rather than by the file log
    counts = {name: spark.read.format("delta").load(str(TABLES[name])).count() for name in ("bronze", "silver", "quarantine")}
    silver_df = spark.read.format("delta").load(str(TABLES["silver"]))
    profile = silver_df.agg(
        F.countDistinct("invoice_no").alias("invoices"), F.countDistinct("customer_id").alias("customers"),
        F.countDistinct("stock_code").alias("products"), F.countDistinct("country").alias("countries"),
        F.sum(F.col("is_return").cast("int")).alias("return_lines"),
        F.sum(F.col("customer_id").isNull().cast("int")).alias("anonymous_lines"),
        F.min("invoice_date").alias("first_day"), F.max("invoice_date").alias("last_day")).collect()[0].asDict()
    quarantine = {r[0]: r[1] for r in spark.read.format("delta").load(str(TABLES["quarantine"]))
                  .groupBy("quarantine_reason").count().collect()}
    dq = spark.read.format("delta").load(str(TABLES["dq"]))
    dq_summary = [r.asDict() for r in dq.groupBy("check", "severity").agg(
        F.count("*").alias("batches"), F.sum(F.col("passed").cast("int")).alias("passed"),
        F.round(F.avg("value"), 4).alias("avg_value"), F.max("threshold").alias("threshold")).orderBy("severity", "check").collect()]

    silver_v = version(spark, TABLES["silver"])
    t = time.perf_counter()
    g = gold.build(spark, SNAPSHOTS, SCORE_DATE, silver_version=silver_v)
    gold_s = time.perf_counter() - t
    t = time.perf_counter()
    m = model.train_and_evaluate(spark, SNAPSHOTS, SCORE_DATE)
    model_s = time.perf_counter() - t

    daily_runs = [e for e in log if e["kind"] == "daily"]
    results = {
        "dataset": {"name": "Online Retail (UCI, Chen et al. 2012)", "rows": counts["bronze"], "daily_files": len(all_days),
                    **{k: (str(v) if isinstance(v, dt.date) else v) for k, v in profile.items()}},
        "tables": counts, "quarantine_reasons": quarantine,
        "runs": {"backfill": log[0], "daily": daily_runs, "replay": replay,
                 "daily_median_s": round(statistics.median(e["total_s"] for e in daily_runs), 2),
                 "daily_p95_s": sorted(e["total_s"] for e in daily_runs)[int(0.95 * (len(daily_runs) - 1))]},
        "data_quality": dq_summary,
        "gold": {**g, "silver_version": silver_v, "seconds": round(gold_s, 1)},
        "model": {**m, "seconds": round(model_s, 1)},
        "table_versions": {k: version(spark, TABLES[k]) for k in ("bronze", "silver", "quarantine", "dq", "features", "scores")},
    }
    if not args.skip_bench:
        full = bench.full_rebuild_seconds(spark)
        results["bench"] = {"full_rebuild": full, "incremental_day_median_s": results["runs"]["daily_median_s"],
                            "incremental_saving": round(1 - results["runs"]["daily_median_s"] / full["seconds"], 3),
                            "zorder": bench.zorder(spark)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (ROOT / "public" / "results.json").write_text(json.dumps(results, separators=(",", ":"), default=str))   # dashboard
    print(json.dumps({k: results[k] for k in ("tables", "gold")}, indent=1, default=str))
    print(json.dumps(m["results"], indent=1))
    if "bench" in results:
        print(json.dumps(results["bench"], indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
