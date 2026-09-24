"""Small helpers: Delta table checks, timing, and scan metrics from Spark's physical plan."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path


def table_exists(spark, path) -> bool:
    from delta.tables import DeltaTable
    return Path(path).exists() and DeltaTable.isDeltaTable(spark, str(path))


def version(spark, path) -> int | None:
    from delta.tables import DeltaTable
    if not table_exists(spark, path):
        return None
    return DeltaTable.forPath(spark, str(path)).history(1).select("version").collect()[0][0]


@contextmanager
def timer(out: dict, key: str):
    t = time.perf_counter()
    yield
    out[key] = round(time.perf_counter() - t, 3)


def scan_metrics(df) -> dict:
    """Run the query and return how many files and bytes the file scans actually read (after data skipping)."""
    df.collect()
    plan = df._jdf.queryExecution().executedPlan()
    totals = {"files": 0, "bytes": 0}

    def walk(node):
        cls = node.getClass().getSimpleName()
        if cls == "AdaptiveSparkPlanExec":          # adaptive execution wraps the final plan
            return walk(node.executedPlan())
        if cls.endswith("QueryStageExec"):
            return walk(node.plan())
        if "Scan" in node.nodeName():
            m = node.metrics()
            for key, out in (("numFiles", "files"), ("filesSize", "bytes")):
                if m.contains(key):
                    totals[out] += int(m.apply(key).value())
        it = node.children().iterator()
        while it.hasNext():
            walk(it.next())

    walk(plan)
    return totals
