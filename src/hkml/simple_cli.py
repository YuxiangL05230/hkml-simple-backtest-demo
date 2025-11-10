# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import os
from typing import Optional
import pandas as pd

from hkml.simple_backtest import (
    load_prices_from_duckdb,
    run_backtest,
    save_outputs,
)
from hkml.metrics import compute_metrics, plot_equity_curves, plot_metrics_bars
from hkml.abtest import compare_strategy_vs_eq, plot_excess_equity, plot_bootstrap_hist

def _load_weights_module(mod_path_or_name: Optional[str]):
    if mod_path_or_name is None:
        return None
    # If file exists, load by path
    if os.path.isfile(mod_path_or_name):
        spec = importlib.util.spec_from_file_location("user_weights_mod", mod_path_or_name)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import strategy module from file: {mod_path_or_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    # Otherwise treat as module name
    return importlib.import_module(mod_path_or_name)

def parse_args():
    p = argparse.ArgumentParser(description="HK single-table backtest (MVP)")
    p.add_argument("--db", required=True, help="Path to DuckDB database, e.g., data/hk_ohlcv.duckdb")
    p.add_argument("--table", default="ohlcv_hk", help="DuckDB table name with OHLCV data")
    p.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--rebal", default="M", help="Rebalance schedule: D / nD / W / M / Q")
    p.add_argument("--universe", default=None, help="Comma-separated ts_code list, e.g. '00700.HK,09988.HK'")
    p.add_argument("--weights-mod", default=None, help="Python module name or file path that defines generate_weights(...)")
    p.add_argument("--tc-bps", type=float, default=10.0, help="Transaction cost in basis points per turnover")
    p.add_argument("--outdir", default="outputs_simple", help="Output directory")
    p.add_argument("--abtest", action="store_true", help="Run A/B test vs equal-weight and save abtest.json & plots")
    p.add_argument("--ab-block", type=int, default=5, help="Block size (trading days) for block bootstrap")
    p.add_argument("--ab-B", type=int, default=2000, help="Bootstrap replicates")
    p.add_argument("--ab-seed", type=int, default=42, help="Random seed for bootstrap")
    return p.parse_args()

def main():
    args = parse_args()
    universe = None
    if args.universe:
        universe = [s.strip() for s in args.universe.split(",") if s.strip()]

    prices = load_prices_from_duckdb(
        db_path=args.db,
        table=args.table,
        universe=universe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Load user strategy module if provided
    strategy_fn = None
    if args.weights_mod:
        mod = _load_weights_module(args.weights_mod)
        if not hasattr(mod, "generate_weights"):
            raise AttributeError("weights-mod module must define a function: generate_weights(asof_date, prices_upto, universe) -> pd.Series")
        strategy_fn = getattr(mod, "generate_weights")

    # Run backtest
    result = run_backtest(
        prices=prices,
        rebal=args.rebal,
        strategy_fn=strategy_fn,
        universe=universe,
        tc_bps=args.tc_bps,
    )

    # Compute and save metrics and plots
    metrics = {
        "strategy": compute_metrics(result.returns),
        "equal_weight": compute_metrics(result.returns_eq),
    }

    os.makedirs(args.outdir, exist_ok=True)
    save_outputs(args.outdir, result, metrics, prices=prices)

    # Equity curve
    plot_equity_curves(
        curves={"strategy": result.returns, "equal_weight": result.returns_eq},
        outpath=os.path.join(args.outdir, "equity_curve.png"),
        title="Strategy vs. Equal-Weight",
    )

    # Metrics comparison bars
    plot_metrics_bars(metrics, os.path.join(args.outdir, "metrics_bars.png"))

    # Optional A/B test
    if args.abtest:
        ab = compare_strategy_vs_eq(
            result.returns, result.returns_eq,
            block=args.ab_block, B=args.ab_B, seed=args.ab_seed, return_samples=True
        )
        with open(os.path.join(args.outdir, "abtest.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in ab.items() if k != "boot_means"}, f, ensure_ascii=False, indent=2)
        # Plots
        plot_excess_equity(result.returns, result.returns_eq, os.path.join(args.outdir, "excess_equity.png"))
        if "boot_means" in ab and len(ab["boot_means"]) > 0:
            import numpy as np
            plot_bootstrap_hist(np.array(ab["boot_means"]), ab["delta_mean"], os.path.join(args.outdir, "bootstrap_hist.png"))

        print(f"[ABTest] n={ab['n']}  mean_excess={ab['delta_mean']:.6g}  "
              f"ΔSharpe={ab['delta_sharpe']:.3f}  p(one-sided, >0)={ab['pvalue_greater']:.4f}")

    print(f"Done. Outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()
