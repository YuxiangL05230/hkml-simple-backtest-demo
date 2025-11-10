# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Dict, Mapping, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    r = returns.fillna(0.0)
    if len(r) == 0:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_drawdown=0.0, cum_return=0.0)
    equity = (1.0 + r).cumprod()
    n = len(r)
    ann_return = equity.iloc[-1] ** (periods_per_year / max(n, 1)) - 1.0
    ann_vol = float(r.std(ddof=0)) * math.sqrt(periods_per_year)
    sharpe = 0.0 if ann_vol == 0 else float(r.mean()) / float(r.std(ddof=0)) * math.sqrt(periods_per_year)
    peak = equity.cummax()
    mdd = float(((equity / peak) - 1.0).min())
    return dict(
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=float(mdd),
        cum_return=float(equity.iloc[-1] - 1.0),
    )

def plot_equity_curves(curves: Mapping[str, pd.Series], outpath: str, title: Optional[str] = None) -> None:
    plt.figure()
    for name, r in curves.items():
        eq = (1.0 + r.fillna(0.0)).cumprod()
        plt.plot(eq.index, eq.values, label=name)
    plt.legend()
    if title:
        plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_metrics_bars(metrics_map: Mapping[str, Dict[str, float]], outpath: str) -> None:
    """画 ann_return/ann_vol/sharpe/max_drawdown 的并列柱状图"""
    keys = ["ann_return", "ann_vol", "sharpe", "max_drawdown"]
    names = list(metrics_map.keys())
    X = range(len(keys))
    plt.figure()
    width = 0.38
    for i, name in enumerate(names):
        vals = [metrics_map[name].get(k, 0.0) for k in keys]
        plt.bar([x + i*width for x in X], vals, width=width, label=name)
    plt.xticks([x + width*(len(names)-1)/2 for x in X], keys, rotation=15)
    plt.title("Metrics Comparison")
    plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
