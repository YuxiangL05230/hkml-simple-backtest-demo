# -*- coding: utf-8 -*-
from __future__ import annotations
"""
my_simple_strategy.py — Parameterizable Top-K momentum weights hook.

This module is designed to be used with your `--weights-mod` hook in simple_cli.py.
It computes equal-weight allocations among the top-K tickers by total return over
a lookback window.

Parameters are controlled via environment variables (set them BEFORE running Python):

- HKML_LOOKBACK: int, number of trading days for lookback window. Default "60".
- HKML_TOPK:     int, number of winners to hold each rebalance. Default "2".
- HKML_DEBUG:    "1" to print debug info during weight generation. Default off.

Expected input schema by the engine:
- `prices_upto`: wide price matrix with index as DatetimeIndex of trading days,
  columns as `ts_code` tickers, and values as `close` price.
- `asof_date`: a timestamp (or date-like) indicating the rebalance date (inclusive).
- `universe`: an iterable of allowed tickers for the current date; can be None.

Return value:
- `pandas.Series` with index = selected tickers, values = weights (sum to 1).
- Return an empty Series to signal "fallback to equal-weight" to the engine.
"""
import os
import pandas as pd

def _env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        val = int(raw)
    except ValueError:
        val = default
    if min_value is not None:
        val = max(min_value, val)
    if max_value is not None:
        val = min(max_value, val)
    return val

LOOKBACK: int = _env_int("HKML_LOOKBACK", 60, min_value=2)
TOP_K: int = _env_int("HKML_TOPK", 2, min_value=1)
DEBUG: bool = os.getenv("HKML_DEBUG", "") not in ("", "0", "false", "False", "no", "No")

def generate_weights(asof_date, prices_upto: pd.DataFrame, universe):
    """Compute Top-K momentum weights for a given as-of date."""
    # Restrict to data up to asof_date (inclusive)
    px = prices_upto.loc[:asof_date]

    # Require at least (LOOKBACK+1) rows to form total return
    if len(px) < LOOKBACK + 1:
        if DEBUG:
            print(f"[weights] {asof_date}: not enough history ({len(px)}<{LOOKBACK+1}), fallback to equal-weight")
        return pd.Series(dtype=float)

    start = px.index[-(LOOKBACK + 1)]
    end = px.index[-1]

    # Total return over lookback window per ticker
    ret = px.loc[end] / px.loc[start] - 1.0
    ret = ret.dropna()

    # Respect universe if provided
    if universe is not None:
        # universe may be a list/Index/Series; cast to set for speed
        u = set(universe)
        ret = ret[ret.index.isin(u)]

    if ret.empty:
        if DEBUG:
            print(f"[weights] {asof_date}: empty return vector after filtering, fallback to equal-weight")
        return pd.Series(dtype=float)

    k = max(1, min(int(TOP_K), len(ret)))
    winners = ret.sort_values(ascending=False).head(k).index

    w = pd.Series(1.0 / len(winners), index=winners)

    if DEBUG:
        print(f"[weights] {asof_date}: lookback={LOOKBACK}, top_k={k}, winners={list(winners)}")

    return w

if __name__ == "__main__":
    # Lightweight self-check when run standalone (optional)
    import numpy as np
    import pandas as pd
    dates = pd.bdate_range("2020-01-01", periods=65)  # just enough for default LOOKBACK=60
    tickers = ["0700.HK", "3690.HK", "9988.HK"]
    rng = np.random.default_rng(42)
    mat = np.cumprod(1 + rng.normal(0.0005, 0.01, size=(len(dates), len(tickers))), axis=0)
    prices = pd.DataFrame(mat * 50, index=dates, columns=tickers)
    print(generate_weights(dates[-1], prices, tickers))
