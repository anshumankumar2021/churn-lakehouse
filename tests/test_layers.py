import datetime as dt

import pytest
from pyspark.sql import Row

HEADER = "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"


def write_day(root, day, lines):
    p = root / "landing" / f"date={day}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "transactions.csv").write_text(HEADER + "".join(l + "\n" for l in lines))


def test_typed_parses_flags_and_quarantines(spark):
    from lakehouse.silver import typed
    rows = [Row(InvoiceNo="536365", StockCode="85123a ", Description="HEART", Quantity="6", InvoiceDate="2010-12-01 08:26:00",
                UnitPrice="2.55", CustomerID="17850", Country="United Kingdom", _source_file="x", _landing_date=dt.date(2010, 12, 1)),
            Row(InvoiceNo="C536379", StockCode="D", Description="Discount", Quantity="-1", InvoiceDate="2010-12-01 09:41:00",
                UnitPrice="27.50", CustomerID="14527", Country="United Kingdom", _source_file="x", _landing_date=dt.date(2010, 12, 1)),
            Row(InvoiceNo="A563185", StockCode="B", Description="Adjust bad debt", Quantity="1", InvoiceDate="2010-12-01 14:50:00",
                UnitPrice="-11062.06", CustomerID=None, Country="United Kingdom", _source_file="x", _landing_date=dt.date(2010, 12, 1)),
            Row(InvoiceNo="536999", StockCode="22", Description="X", Quantity="1", InvoiceDate="not a date",
                UnitPrice="1", CustomerID=None, Country="UK", _source_file="x", _landing_date=dt.date(2010, 12, 1))]
    out = {r["invoice_no"]: r for r in typed(spark.createDataFrame(rows)).collect()}
    a = out["536365"]
    assert a["stock_code"] == "85123A" and a["quantity"] == 6 and float(a["line_amount"]) == 15.30
    assert a["customer_id"] == 17850 and not a["is_return"] and a["is_product"] and a["quarantine_reason"] is None
    assert out["C536379"]["is_return"] and not out["C536379"]["is_product"]
    assert out["A563185"]["quarantine_reason"] == "negative or missing price"
    assert out["536999"]["quarantine_reason"] == "unparseable timestamp"


def test_incremental_load_is_idempotent_and_deduplicates(spark, lake):
    from lakehouse import bronze, quality, silver
    line = "536365,85123A,HEART,6,2010-12-01 08:26:00,2.55,17850,United Kingdom"
    write_day(lake, "2010-12-01", [line, line, "536366,22633,HAND WARMER,6,2010-12-01 08:28:00,1.85,17850,United Kingdom"])
    write_day(lake, "2010-12-02", ["536367,84879,BIRD ORNAMENT,32,2010-12-02 08:34:00,1.69,13047,United Kingdom"])
    bronze.ingest(spark, ["2010-12-01"])
    s1 = silver.upsert(spark, ["2010-12-01"])
    assert (s1["rows_in"], s1["duplicates_in_batch"], s1["inserted"]) == (3, 1, 2)
    quality.gate(spark, "t1", ["2010-12-01"], s1)
    bronze.ingest(spark, ["2010-12-01", "2010-12-02"])          # day 1 is skipped: already ingested
    s2 = silver.upsert(spark, ["2010-12-02"])
    assert s2["inserted"] == 1
    again = silver.upsert(spark, ["2010-12-01", "2010-12-02"])   # replay: nothing new
    assert again["inserted"] == 0 and again["already_in_silver"] == 3
    quality.gate(spark, "t2", ["2010-12-01", "2010-12-02"], again)
    assert spark.read.format("delta").load(str(lake / "lake" / "silver")).count() == 3
    assert spark.read.format("delta").load(str(lake / "lake" / "bronze")).count() == 4


def test_gold_features_and_labels(spark):
    from lakehouse.gold import snapshot
    d = dt.date

    def r(inv, cust, day, qty=1, price=10.0, ret=False, prod=True, country="United Kingdom"):
        return Row(invoice_no=inv, stock_code=f"S{inv}", customer_id=cust, invoice_date=day, quantity=qty,
                   line_amount=qty * price, is_return=ret, is_product=prod, country=country)
    silver = spark.createDataFrame([
        r("1", 1, d(2011, 1, 5), qty=2), r("2", 1, d(2011, 2, 10)), r("3", 1, d(2011, 4, 1)),   # A: comes back in April
        r("4", 2, d(2011, 1, 20), country="France"),                                             # B: never returns
        r("C5", 2, d(2011, 1, 25), qty=-1, ret=True),                                            # B's return
        r("6", None, d(2011, 2, 1)),                                                             # anonymous: ignored
        r("7", 3, d(2011, 3, 15)),                                                               # C: first order after snapshot
    ])
    out = {x["customer_id"]: x for x in snapshot(silver, d(2011, 3, 1), label_until=d(2011, 12, 9)).collect()}
    assert set(out) == {1, 2}                               # C isn't a customer yet on 1 March
    a, b = out[1], out[2]
    assert a["recency_days"] == 19 and a["frequency"] == 2 and a["tenure_days"] == 55 and a["monetary"] == 30.0
    assert a["churned"] == 0 and b["churned"] == 1
    assert b["is_uk"] == 0 and b["return_rate"] == pytest.approx(0.5)
    unlabelled = snapshot(silver, d(2011, 3, 1), label_until=None).collect()
    assert all(x["churned"] is None for x in unlabelled)


def test_bad_values_are_quarantined_not_fatal(spark):
    from lakehouse.silver import typed
    rows = [Row(InvoiceNo="1", StockCode="A", Description="x", Quantity="six", InvoiceDate="2010-12-01 08:26:00",
                UnitPrice="abc", CustomerID="n/a", Country="UK", _source_file="x", _landing_date=dt.date(2010, 12, 1))]
    r = typed(spark.createDataFrame(rows)).collect()[0]
    assert r["quantity"] is None and r["unit_price"] is None and r["customer_id"] is None
    assert r["quarantine_reason"] == "zero or missing quantity"
