#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_tiny_db.py — generate a tiny offline DuckDB OHLCV table for smoke tests.

Usage:
  python scripts/make_tiny_db.py --out data/tiny.duckdb --table ohlcv_hk \      --tickers 0700.HK,3690.HK --rows 60 --start 2020-01-01 --seed 42

Default schema (per-row):
  ts_code, trade_date (DATE), open, high, low, close, vol, amount, stock_name_cn, stock_symbol
This matches typical loaders expecting HK-style OHLCV with a per-ticker date index.
"""

import argparse
import pathlib
from typing import List

import duckdb
import numpy as np
import pandas as pd


def _gen_one_ticker(ticker: str, dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_price = float(rng.uniform(20, 80))
    # small positive drift for readability
    rets = rng.normal(loc=0.0007, scale=0.01, size=len(dates))
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.0005, 0.01, size=len(dates))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    vol = rng.integers(2_000, 50_000, size=len(dates)).astype("int64")
    amount = (vol * close).astype("float64")

    df = pd.DataFrame(
        {
            "ts_code": ticker,
            "trade_date": dates.date,  # DATE in DuckDB
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "vol": vol,
            "amount": np.round(amount, 2),
            "stock_name_cn": [None] * len(dates),
            "stock_symbol": [ticker.split(".")[0]] * len(dates),
        }
    )
    return df


def make_tiny_db(
    out_path: pathlib.Path,
    table: str,
    tickers: List[str],
    start: str,
    rows: int,
    seed: int,
    freq: str = "B",
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=rows, freq=freq)
    frames = []
    for i, t in enumerate(tickers):
        frames.append(_gen_one_ticker(t, dates, seed + i))
    df = pd.concat(frames, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out_path))
    try:
        # Use a registered relation for reliable CREATE ... AS SELECT
        con.register("df", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
        summary = con.execute(
            f"""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT ts_code) AS tickers,
                   MIN(trade_date) AS start,
                   MAX(trade_date) AS end
            FROM {table}
            """
        ).fetchdf()
    finally:
        con.close()

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a tiny DuckDB OHLCV table for smoke tests (offline)")
    p.add_argument("--out", default="data/tiny.duckdb", help="Output DuckDB file path (will be created if missing)")
    p.add_argument("--table", default="ohlcv_hk", help="Table name to create/overwrite")
    p.add_argument(
        "--tickers",
        default="0700.HK,3690.HK",
        help="Comma-separated tickers, e.g. 0700.HK,3690.HK",
    )
    p.add_argument("--rows", type=int, default=60, help="Number of business days to generate per ticker")
    p.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD) for the generated series")
    p.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    p.add_argument("--freq", default="B", help="Pandas date_range frequency, default B (business days)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [s.strip() for s in args.tickers.split(",") if s.strip()]
    out = pathlib.Path(args.out)
    summary = make_tiny_db(out, args.table, tickers, args.start, args.rows, args.seed, args.freq)
    print(f"Wrote DuckDB: {out}  table: {args.table}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
