# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd

# ---------- Data Loading ----------

def load_prices_from_duckdb(
    db_path: str,
    table: str = "ohlcv_hk",
    universe: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    price_col: str = "close",
    date_col: str = "trade_date",
    code_col: str = "ts_code",
) -> pd.DataFrame:
    """
    Load OHLCV daily prices from a DuckDB table and pivot to a wide price matrix.
    Returns a DataFrame with index=DatetimeIndex (trade_date), columns=ts_code, values=close.
    Missing values are forward-filled by column.
    """
    con = duckdb.connect(database=db_path, read_only=True)
    # Build SQL
    where = []
    if start_date:
        where.append(f"{date_col} >= DATE '{start_date}'")
    if end_date:
        where.append(f"{date_col} <= DATE '{end_date}'")
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT {date_col}::DATE AS trade_date, {code_col}::VARCHAR AS ts_code, {price_col}::DOUBLE AS price
        FROM {table}
        {where_clause}
    """
    df = con.execute(sql).df()
    con.close()

    if df.empty:
        raise ValueError("No data loaded from DuckDB. Check db path, table name, and date range.")

    # Optional universe filter
    if universe is not None:
        uni = set(u.strip() for u in universe if u and str(u).strip())
        df = df[df["ts_code"].isin(uni)]
        if df.empty:
            raise ValueError("After universe filtering, no rows remain. Check universe tickers.")

    # Pivot to wide matrix and forward-fill
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    prices = df.pivot(index="trade_date", columns="ts_code", values="price").sort_index()
    prices = prices.ffill().dropna(how="all")

    # Drop columns that are entirely NaN (should not happen after ffill, but just in case)
    prices = prices.dropna(axis=1, how="all")
    return prices


# ---------- Rebalance Schedule ----------

def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False

def get_rebalance_dates(dates: pd.DatetimeIndex, rebal: str = "M") -> pd.DatetimeIndex:
    """
    Compute rebalance dates from a list of trading dates.
    - 'D'  : every trading day
    - 'nD' : every n-th trading day, e.g. '5D'
    - 'W'  : last trading day of each week
    - 'M'  : last trading day of each month (default)
    - 'Q'  : last trading day of each quarter
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    r = rebal.upper()
    if r == "D":
        return dates
    if r.endswith("D") and _is_int(r[:-1] or "0") and r != "D":
        step = int(r[:-1])
        if step <= 0:
            raise ValueError("Invalid nD step; must be positive.")
        return dates[::step]
    if r == "W":
        # last trading day of each calendar week
        df = pd.DataFrame(index=dates)
        return df.groupby(df.index.to_period("W")).apply(lambda x: x.index.max()).to_numpy()
    if r == "M":
        df = pd.DataFrame(index=dates)
        return df.groupby(df.index.to_period("M")).apply(lambda x: x.index.max()).to_numpy()
    if r == "Q":
        df = pd.DataFrame(index=dates)
        return df.groupby(df.index.to_period("Q")).apply(lambda x: x.index.max()).to_numpy()
    raise ValueError(f"Unsupported rebal spec: {rebal}")


# ---------- Strategy Hook ----------

WeightsFn = Callable[[pd.Timestamp, pd.DataFrame, List[str]], pd.Series]
"""
Signature:
    generate_weights(asof_date, prices_upto, universe) -> pd.Series
Return:
    pd.Series indexed by ts_code with non-negative weights summing to 1 (will be re-normalized).
"""


# ---------- Backtest Engine ----------

@dataclass
class BacktestResult:
    returns: pd.Series  # daily portfolio returns for strategy
    returns_eq: pd.Series  # daily portfolio returns for equal-weight baseline
    weights: pd.DataFrame  # target weights on each rebalance date (strategy)
    weights_eq: pd.DataFrame  # target weights on each rebalance date (equal-weight)
    rebal_dates: pd.DatetimeIndex


def _normalize_weights(w: pd.Series) -> pd.Series:
    w = w.fillna(0.0).clip(lower=0.0)
    s = float(w.sum())
    if s <= 0.0:
        # No positions -> all cash (return 0). But we enforce non-empty by equal-weight fallback.
        return w * 0.0
    return w / s


def _gen_equal_weight(asof_date: pd.Timestamp, prices_upto: pd.DataFrame, universe: List[str]) -> pd.Series:
    last_row = prices_upto.loc[:asof_date].iloc[-1]
    tradable = last_row.dropna().index.intersection(universe)
    if len(tradable) == 0:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / len(tradable), index=list(tradable))
    return _normalize_weights(w)


def run_backtest(
    prices: pd.DataFrame,
    rebal: str = "M",
    strategy_fn: Optional[WeightsFn] = None,
    universe: Optional[Iterable[str]] = None,
    tc_bps: float = 0.0,
) -> BacktestResult:
    """
    Wide price matrix backtest with no-lookahead; long-only weights; bps*turnover cost.
    """
    # Universe
    if universe is None:
        universe = list(prices.columns)
    else:
        universe = [u for u in universe if u in prices.columns]
        if not universe:
            raise ValueError("Universe after intersecting with columns is empty.")

    dates = prices.index
    rebal_dates = pd.DatetimeIndex(get_rebalance_dates(dates, rebal))

    # Strategy function fallback
    strat = strategy_fn if strategy_fn is not None else _gen_equal_weight

    # Returns
    rets = prices.pct_change().fillna(0.0)
    one_plus_rets = 1.0 + rets

    # Storage
    weights_records = []
    weights_eq_records = []
    port_ret = pd.Series(0.0, index=dates, name="strategy")
    port_ret_eq = pd.Series(0.0, index=dates, name="equal_weight")

    # Initial weights (align to full universe)
    first_date = dates[0]
    prev_target_w = _gen_equal_weight(first_date, prices.loc[:first_date], list(universe)) \
        .reindex(universe).fillna(0.0)
    prev_target_w_eq = prev_target_w.copy()

    w_curr = prev_target_w.copy()
    w_curr_eq = prev_target_w_eq.copy()

    tc = float(tc_bps) / 10000.0
    rebal_set = set(pd.Timestamp(d) for d in rebal_dates)
    pending_cost = 0.0
    pending_cost_eq = 0.0

    for dt in dates:
        # Daily return (pandas alignment)
        asset_rets_t = rets.loc[dt].reindex(universe).fillna(0.0)
        w_curr    = w_curr.reindex(universe).fillna(0.0)
        w_curr_eq = w_curr_eq.reindex(universe).fillna(0.0)
        port_ret[dt]    = float((w_curr    * asset_rets_t).sum()) - pending_cost
        port_ret_eq[dt] = float((w_curr_eq * asset_rets_t).sum()) - pending_cost_eq

        # reset pending costs
        pending_cost = 0.0
        pending_cost_eq = 0.0

        # Drift & renormalize
        drift = one_plus_rets.loc[dt].reindex(universe).fillna(1.0).values
        def _drift_norm(w, drift):
            v = w.reindex(universe).fillna(0.0).values * drift
            s = v.sum()
            if s <= 0:
                return pd.Series(0.0, index=universe)
            return pd.Series(v / s, index=universe)
        w_after = _drift_norm(w_curr, drift)
        w_after_eq = _drift_norm(w_curr_eq, drift)

        # Rebalance at end-of-day (targets effective next day)
        if dt in rebal_set:
            asof = pd.Timestamp(dt)
            prices_upto = prices.loc[:asof]

            w_target = strat(asof, prices_upto, list(universe))
            w_target = _normalize_weights(w_target.reindex(universe).fillna(0.0))
            if w_target.sum() == 0.0:
                w_target = _gen_equal_weight(asof, prices_upto, list(universe))
            weights_records.append((asof, w_target.copy()))

            w_target_eq = _gen_equal_weight(asof, prices_upto, list(universe))
            weights_eq_records.append((asof, w_target_eq.copy()))

            # Turnover cost (index-aligned)
            turnover    = float((w_target    - w_after).abs().reindex(universe).fillna(0.0).sum())
            turnover_eq = float((w_target_eq - w_after_eq).abs().reindex(universe).fillna(0.0).sum())
            pending_cost    = tc * turnover
            pending_cost_eq = tc * turnover_eq

            # Apply new targets (effective next day)
            w_curr = w_target.copy()
            w_curr_eq = w_target_eq.copy()
        else:
            w_curr = w_after
            w_curr_eq = w_after_eq

    # Build weights frames
    if weights_records:
        idx, rows = zip(*weights_records)
        weights_df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).reindex(columns=universe).fillna(0.0)
    else:
        weights_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=universe)

    if weights_eq_records:
        idx, rows = zip(*weights_eq_records)
        weights_eq_df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).reindex(columns=universe).fillna(0.0)
    else:
        weights_eq_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=universe)

    return BacktestResult(
        returns=port_ret.astype(float),
        returns_eq=port_ret_eq.astype(float),
        weights=weights_df,
        weights_eq=weights_eq_df,
        rebal_dates=rebal_dates,
    )



# ---------- I/O helpers ----------

def save_outputs(
    outdir: str,
    result: BacktestResult,
    metrics: Dict[str, Dict[str, float]],
    prices: Optional[pd.DataFrame] = None,
):
    os.makedirs(outdir, exist_ok=True)
    # Weights
    result.weights.to_csv(os.path.join(outdir, "weights_strategy.csv"), float_format="%.8f")
    result.weights_eq.to_csv(os.path.join(outdir, "weights_equal_weight.csv"), float_format="%.8f")
    # Returns
    pd.DataFrame({"strategy": result.returns, "equal_weight": result.returns_eq}).to_csv(
        os.path.join(outdir, "portfolio_returns.csv"), float_format="%.8f"
    )
    # Metrics
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # Optional: save the trading calendar
    if prices is not None:
        pd.Series(prices.index, name="trade_date").to_frame().to_csv(os.path.join(outdir, "calendar.csv"), index=False)
