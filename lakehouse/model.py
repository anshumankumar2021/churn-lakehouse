"""Churn model: Spark ML, out-of-time evaluation, MLflow tracking.

Split by snapshot date, never by customer at random, so the test set is genuinely "the future":
    train       snapshots Mar-Jul 2011
    validation  Aug 2011   (model selection)
    test        Sep 2011   (reported once)

Candidates: a recency rule (days since last order, the usual heuristic), logistic regression, and gradient-boosted
trees. Reported: ROC AUC, PR AUC, and what a retention team would use: lift and precision in the top 10% / 20%
highest-risk customers.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from lakehouse.config import ROOT, TABLES
from lakehouse.gold import FEATURES

SEED = 42


def ranking_metrics(y: list[int], score: list[float]) -> dict:
    """ROC AUC, PR AUC (average precision), lift and precision at the top 10% / 20%. Pure Python, unit-tested."""
    pairs = sorted(zip(score, y), key=lambda p: -p[0])
    n, pos = len(pairs), sum(y)
    base = pos / n if n else 0.0
    # ROC AUC via rank sum (ties get average rank)
    order = sorted(range(n), key=lambda i: score[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    neg = n - pos
    auc = (sum(r for r, t in zip(ranks, y) if t) - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float("nan")
    hits, ap = 0, 0.0
    for k, (_, t) in enumerate(pairs, 1):
        if t:
            hits += 1
            ap += hits / k
    ap = ap / pos if pos else float("nan")
    out = {"roc_auc": round(auc, 4), "pr_auc": round(ap, 4), "base_rate": round(base, 4), "n": n}
    for frac in (0.1, 0.2):
        k = max(1, int(math.ceil(n * frac)))
        prec = sum(t for _, t in pairs[:k]) / k
        out[f"precision_top{int(frac * 100)}"] = round(prec, 4)
        out[f"lift_top{int(frac * 100)}"] = round(prec / base, 3) if base else None
        out[f"recall_top{int(frac * 100)}"] = round(sum(t for _, t in pairs[:k]) / pos, 4) if pos else None
    return out


def lift_curve(y, score, points=10):
    pairs = sorted(zip(score, y), key=lambda p: -p[0])
    n, pos = len(pairs), sum(y)
    out = []
    for q in range(1, points + 1):
        k = int(round(n * q / points))
        out.append({"top_share": q / points, "captured_churners": round(sum(t for _, t in pairs[:k]) / pos, 4) if pos else None})
    return out


def split(features: DataFrame, snapshots: list[dt.date]):
    lab = features.where(F.col("churned").isNotNull())
    test_d, val_d = snapshots[-1], snapshots[-2]
    train = lab.where(F.col("snapshot_date") < F.lit(val_d.isoformat()).cast("date"))
    val = lab.where(F.col("snapshot_date") == F.lit(val_d.isoformat()).cast("date"))
    test = lab.where(F.col("snapshot_date") == F.lit(test_d.isoformat()).cast("date"))
    return train, val, test


def pipelines() -> dict:
    asm = VectorAssembler(inputCols=FEATURES, outputCol="raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    lr = LogisticRegression(featuresCol="features", labelCol="churned", maxIter=200, regParam=0.01, elasticNetParam=0.0)
    gbt = GBTClassifier(featuresCol="raw", labelCol="churned", maxDepth=4, maxIter=80, stepSize=0.08,
                        subsamplingRate=0.8, seed=SEED)
    return {"logistic_regression": Pipeline(stages=[asm, scaler, lr]), "gradient_boosted_trees": Pipeline(stages=[asm, gbt])}


def scored(model, df: DataFrame) -> tuple[list[int], list[float]]:
    rows = (model.transform(df).select("churned", vector_to_array("probability")[1].alias("p"))
            .orderBy("customer_id").collect())
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]


def train_and_evaluate(spark: SparkSession, snapshots: list[dt.date], score_date: dt.date) -> dict:
    import mlflow

    feats = spark.read.format("delta").load(str(TABLES["features"])).fillna(0, subset=FEATURES).cache()
    train, val, test = split(feats, snapshots)
    # one partition in a fixed order: tree subsampling depends on partitioning, so this makes training repeatable
    train = train.orderBy("snapshot_date", "customer_id").coalesce(1).cache()
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("customer-churn") is None:
        mlflow.create_experiment("customer-churn", artifact_location=(ROOT / "mlruns" / "artifacts").as_uri())
    mlflow.set_experiment("customer-churn")
    results, models = {}, {}

    # baseline: rank by recency (longer since last order = more likely to churn)
    for name, df in (("validation", val), ("test", test)):
        rows = df.select("churned", "recency_days").orderBy("customer_id").collect()
        results.setdefault("recency_rule", {})[name] = ranking_metrics([int(r[0]) for r in rows], [float(r[1]) for r in rows])

    for name, pipe in pipelines().items():
        with mlflow.start_run(run_name=name):
            model = pipe.fit(train)
            models[name] = model
            mlflow.log_params({"model": name, "features": len(FEATURES), "train_rows": train.count(),
                               "train_snapshots": ",".join(str(s) for s in snapshots[:-2])})
            for split_name, df in (("validation", val), ("test", test)):
                y, p = scored(model, df)
                m = ranking_metrics(y, p)
                results.setdefault(name, {})[split_name] = m
                mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items() if isinstance(v, (int, float)) and v == v})
            out_dir = ROOT / "models" / name
            model.write().overwrite().save(str(out_dir))
            mlflow.log_artifacts(str(out_dir), artifact_path="spark-model")

    best = max(models, key=lambda k: results[k]["validation"]["roc_auc"])
    model = models[best]
    y, p = scored(model, test)
    importances = None
    last = model.stages[-1]
    if hasattr(last, "featureImportances"):
        importances = dict(sorted(((f, round(float(v), 4)) for f, v in zip(FEATURES, last.featureImportances.toArray())),
                                  key=lambda kv: -kv[1]))
    elif hasattr(last, "coefficients"):
        importances = dict(sorted(((f, round(float(v), 4)) for f, v in zip(FEATURES, last.coefficients.toArray())),
                                  key=lambda kv: -abs(kv[1])))

    # score today's customers with the selected model and publish to gold
    current = feats.where(F.col("snapshot_date") == F.lit(score_date.isoformat()).cast("date"))
    risk = (model.transform(current)
            .select("customer_id", "snapshot_date", vector_to_array("probability")[1].alias("churn_probability"),
                    "recency_days", "frequency", "monetary", "orders_90d", "spend_trend"))
    risk.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(str(TABLES["scores"]))
    top = [r.asDict() for r in spark.read.format("delta").load(str(TABLES["scores"]))
           .orderBy(F.desc("churn_probability"), "customer_id").limit(25).collect()]
    for r in top:
        r["snapshot_date"] = str(r["snapshot_date"])
        r["churn_probability"] = round(r["churn_probability"], 4)
        r["monetary"] = round(r["monetary"], 2)
        r["spend_trend"] = round(r["spend_trend"], 3)
    feats.unpersist()
    train.unpersist()
    return {"selected_model": best, "results": results, "feature_importance": importances,
            "lift_curve": {"model": lift_curve(y, p),
                           "recency_rule": lift_curve(*_recency(test))},
            "at_risk_customers": top,
            "scored_customers": spark.read.format("delta").load(str(TABLES["scores"])).count()}


def _recency(df):
    rows = df.select("churned", "recency_days").orderBy("customer_id").collect()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]
