# Customer Churn Lakehouse

[![ci](https://github.com/anshumankumar2021/churn-lakehouse/actions/workflows/ci.yml/badge.svg)](https://github.com/anshumankumar2021/churn-lakehouse/actions/workflows/ci.yml)

**Dashboard: [churn-lakehouse.vercel.app](https://churn-lakehouse.vercel.app)**

A medallion lakehouse built with **PySpark and Delta Lake** over a year of real transactions from a UK online
retailer. Daily files land in bronze. They are typed, de-duplicated and MERGEd into silver behind data-quality
gates, and turned into point-in-time customer features in gold. Those features feed a churn model built with
**Spark ML**, tracked in **MLflow**, and evaluated on a month it never saw.

```
 landing/ (305 daily CSVs)
    │  append, skip files already loaded
    ▼
 bronze.transactions      raw strings + lineage (_source_file, _ingested_at)            Delta, partitioned by day
    │  try_cast types, line_id hash, quarantine bad rows, MERGE on line_id (month-pruned)
    ▼
 silver.transactions ──► ops.data_quality (gate after every batch)   silver.quarantine
    │  monthly snapshots, features from data before the snapshot, 90-day churn label, pinned silver version
    ▼
 gold.customer_features ──► Spark ML (recency rule | logistic regression | GBT) ──► MLflow
    ▼
 gold.churn_scores
```

## Data

The [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) dataset (Chen, Sain & Guo, 2012),
redistributed under CC0 in the CRAN package `onlineretail`:

- **541,909** invoice lines from 1 Dec 2010 to 9 Dec 2011
- 25,898 invoices, 4,372 customers, 3,958 products, 38 countries

`scripts/prepare_data.py` splits it into 305 daily CSV files, one per trading day, to simulate an upstream
system's daily deliveries.

## Results

All numbers are from one run on a 2-vCPU machine in Spark local mode, and are in `results/results.json`.

**Pipeline**

| | |
|---|---|
| Backfill (up to 31 Oct 2011) | 431,673 rows in 48 s |
| Daily incremental loads (34 days) | median **5.7 s**, p95 8.1 s |
| Full rebuild of silver from bronze | 10.6 s → incremental load is **46% less work** per day, and the gap grows with history |
| Replaying the last 5 days | **0** rows inserted (idempotent) |
| Exact duplicate lines removed | 5,268 |
| Rows quarantined | 2 (negative prices: bad-debt adjustments) |
| Critical data-quality checks | passed in 36/36 batches |
| Warnings | the share of lines with no customer ID exceeded 40% in 5 of 36 batches |

**Table layout.** Silver was copied 20× with offset customer IDs: 8.0M rows in 400 small files, the way frequent
small writes leave a table. Then 25 single-customer lookups were run before and after
`OPTIMIZE … ZORDER BY (customer_id)`, with a 32 MB target file size.

| | Files | Files read per lookup | MB read per lookup | Median lookup |
|---|---|---|---|---|
| Before | 400 | 400 | 747 | 1,154 ms |
| After Z-ordering | 22 | **1** | 30 | **212 ms** (5.4× faster) |

**Churn model.** A customer has churned at a snapshot if they buy nothing in the next 90 days.

- **Training:** March–July 2011 snapshots (11,918 customer-months).
- **Model selection:** August.
- **Test:** September, reported once: 3,314 customers, 43.9% of whom churned.

| Model | ROC AUC | PR AUC | Precision in riskiest 10% | Lift, top 10% |
|---|---|---|---|---|
| Recency rule (days since last order) | 0.684 | 0.606 | 67.2% | 1.53× |
| Logistic regression | 0.748 | 0.648 | 69.3% | 1.58× |
| **Gradient-boosted trees** (chosen on validation) | **0.749** | **0.652** | **69.0%** | **1.57×** |

The learned models beat the usual recency heuristic by about 6.5 AUC points. The GBT and logistic regression
are within noise of each other, so the tree model's extra complexity buys little here. The strongest signals are
product breadth, basket size, order value and the number of months a customer has been active. All 4,335 customers are scored as of
10 Dec 2011 into `gold.churn_scores`.

### Limits

- The data is one retailer over one year, so seasonality is confounded with the snapshots. Churn is lowest for the
  September snapshot, whose 90-day window covers the Christmas season.
- Timings come from a small machine in local mode. On a cluster, absolute numbers change; the relative
  comparisons (incremental vs rebuild, before vs after Z-order) are the point.
- The Z-order benchmark uses a scaled copy of the real data, because 537K rows is too small for file layout to
  matter.

## Design choices

- **Bronze keeps everything as strings.** A malformed upstream value is stored and inspectable, never lost at
  ingestion.
- **`try_cast` in silver.** Spark 4 runs in ANSI mode, where one bad value in a plain cast fails the whole batch.
  With `try_cast`, it becomes NULL and that one row is quarantined with a reason (covered by a test).
- **MERGE on a content hash.** `line_id` hashes the whole line, and the MERGE condition is pruned to the batch's
  month partitions. Re-sent or re-run data never double-counts revenue.
- **Flags, not filters.** Returns, non-product lines (postage, fees) and anonymous sales stay in silver, flagged.
  Gold decides what to use.
- **Quality gate after every batch** (`ops.data_quality`, a Delta table):
  - row reconciliation: in = inserted + duplicates + already loaded + quarantined
  - unique `line_id` and non-null required columns
  - warnings on quarantine, anonymous and return shares
- **Point-in-time gold.** Features only use data before each snapshot date. The build pins the silver Delta
  version it read, so any snapshot can be rebuilt exactly with time travel.
- **Out-of-time split** (train / validate / test by month, never random), with deterministic training so CI can
  check the metrics.

## Run it

```bash
pip install -r requirements-dev.txt          # Java 17+ required for Spark
python -m scripts.prepare_data               # download + split into daily files
python -m scripts.run_pipeline               # backfill, daily MERGEs, replay, gold, model, benchmarks (~15 min)
pytest -q                                    # unit tests for each layer
```

Spark downloads the Delta jars from Maven Central. On a machine without Maven access, set `LAKEHOUSE_JARS` to a
folder containing the jars; the `vendor-jars` workflow publishes them as a release asset. MLflow runs are logged
to `mlruns/` (SQLite backend).

CI runs two jobs:
- **Unit tests**, on a fresh Spark + Delta.
- **The whole pipeline from the raw download**, which fails if table row counts, quarantine results, gold
  snapshots or model AUCs differ from the published results. Timings are not compared.

## Layout

```
lakehouse/config.py    paths, Spark session with Delta
lakehouse/bronze.py    idempotent raw ingestion
lakehouse/silver.py    typing, quarantine, line_id, MERGE
lakehouse/quality.py   data-quality gate
lakehouse/gold.py      point-in-time customer features and churn labels
lakehouse/model.py     Spark ML models, out-of-time evaluation, MLflow, scoring
lakehouse/bench.py     incremental vs full rebuild, Z-order benchmark
scripts/               prepare_data, run_pipeline
public/                dashboard (reads results.json)
```
