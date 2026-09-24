import math

from lakehouse.model import ranking_metrics
from lakehouse.quality import evaluate


def test_ranking_metrics():
    y = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    perfect = ranking_metrics(y, [0.9, 0.8, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert perfect["precision_top10"] == 1.0 and perfect["lift_top10"] == round(1 / 0.3, 3)
    inverse = ranking_metrics(y, [0.0, 0.0, 1, 1, 0.0, 1, 1, 1, 1, 1])
    assert inverse["roc_auc"] == 0.0
    ties = ranking_metrics(y, [0.5] * 10)
    assert ties["roc_auc"] == 0.5


def base_stats(**kw):
    s = {"rows_in": 100, "quarantined": 1, "duplicates_in_batch": 2, "already_in_silver": 0, "inserted": 97}
    s.update(kw)
    return s


def silver(**kw):
    s = {"duplicate_line_ids": 0, "null_required": 0, "rows_on_days": 97, "anonymous_on_days": 20, "returns_on_days": 2}
    s.update(kw)
    return s


def test_quality_checks_pass_and_fail():
    ok = {r["check"]: r for r in evaluate(base_stats(), silver())}
    assert all(r["passed"] for r in ok.values())
    lost = {r["check"]: r for r in evaluate(base_stats(inserted=90), silver())}
    assert not lost["every input row is accounted for"]["passed"]
    dup = {r["check"]: r for r in evaluate(base_stats(), silver(duplicate_line_ids=3))}
    assert not dup["line_id unique in silver"]["passed"] and dup["line_id unique in silver"]["severity"] == "critical"
    noisy = {r["check"]: r for r in evaluate(base_stats(quarantined=5, inserted=93), silver(anonymous_on_days=60))}
    assert not noisy["quarantined share of batch"]["passed"] and not noisy["anonymous (no customer) share"]["passed"]
