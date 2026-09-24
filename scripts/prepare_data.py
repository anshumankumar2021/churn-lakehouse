"""Download the Online Retail dataset and write it as daily CSV drops into data/landing/.

Source: UCI "Online Retail" (Chen, Sain & Guo, 2012): 541,909 invoice lines from a UK online retailer,
1 Dec 2010 - 9 Dec 2011, redistributed under CC0 in the CRAN package `onlineretail` (GitHub mirror cran/onlineretail).

    python -m scripts.prepare_data
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from lakehouse.config import DATA, LANDING

REPO = "https://github.com/cran/onlineretail"


def main():
    import pandas as pd
    import pyreadr

    src = DATA / "source" / "onlineretail"
    if not (src / "data" / "onlineretail.rda").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", REPO, str(src)], check=True)
    df = next(iter(pyreadr.read_r(str(src / "data" / "onlineretail.rda")).values()))
    df.columns = [c.strip() for c in df.columns]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = df["Quantity"].astype("int64")
    df["CustomerID"] = df["CustomerID"].astype("Int64")      # nullable: ~25% of lines have no customer
    day = df["InvoiceDate"].dt.strftime("%Y-%m-%d")
    LANDING.mkdir(parents=True, exist_ok=True)
    n = 0
    for d, part in df.groupby(day, sort=True):
        out = LANDING / f"date={d}" / "transactions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.assign(InvoiceDate=part["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(out, index=False)
        n += 1
    print(f"{len(df):,} rows -> {n} daily files in {LANDING}")


if __name__ == "__main__":
    main()
