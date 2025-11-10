# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _sharpe(r: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(r).dropna()
    s = r.std(ddof=0)
    return 0.0 if s == 0 else float(r.mean() / s * np.sqrt(periods_per_year))

def compare_strategy_vs_eq(
    r_strategy: pd.Series,
    r_equal: pd.Series,
    block: int = 5,
    B: int = 2000,
    seed: int = 42,
    return_samples: bool = False,
) -> dict:
    """
    成对区块自助法（circular block bootstrap）：
    H0: mean(r_strategy - r_equal) = 0
    H1: mean(r_strategy - r_equal) > 0  （单侧检验）
    返回 p 值、均值差、Sharpe 差；可选返回自助样本均值数组用于画图。
    """
    rs = pd.Series(r_strategy).astype(float)
    re = pd.Series(r_equal).reindex_like(rs).fillna(0.0).astype(float)
    d = (rs - re).to_numpy()
    n = d.shape[0]
    if n == 0:
        out = {"n": 0, "delta_mean": 0.0, "delta_sharpe": 0.0, "pvalue_greater": 1.0}
        if return_samples:
            out["boot_means"] = []
        return out

    rng = np.random.default_rng(seed)
    L = max(1, int(np.ceil(n / block)))

    # 观测统计量
    obs_mean = float(d.mean())
    delta_sharpe = _sharpe(rs) - _sharpe(re)

    # 在 H0 下去中心化
    d0 = d - obs_mean
    starts = rng.integers(0, n, size=(B, L))
    idx = (starts[:, :, None] + np.arange(block)) % n
    boot = d0[idx.reshape(B, -1)[:, :n]]  # (B, n)
    boot_means = boot.mean(axis=1)

    # 单侧 p 值（策略>等权）
    p_greater = float((np.sum(boot_means >= obs_mean) + 1) / (B + 1))

    out = {
        "n": int(n),
        "delta_mean": obs_mean,
        "delta_sharpe": float(delta_sharpe),
        "pvalue_greater": p_greater,
        "block": int(block),
        "B": int(B),
    }
    if return_samples:
        out["boot_means"] = boot_means.tolist()
    return out

def plot_excess_equity(r_strategy: pd.Series, r_equal: pd.Series, outpath: str) -> None:
    """画相对净值：cumprod(1+rs) / cumprod(1+re)"""
    rs = pd.Series(r_strategy).fillna(0.0)
    re = pd.Series(r_equal).reindex_like(rs).fillna(0.0)
    rel = (1.0 + rs).cumprod() / (1.0 + re).cumprod()
    plt.figure()
    plt.plot(rel.index, rel.values)
    plt.axhline(1.0, linestyle="--")
    plt.title("Excess Equity (Strategy / Equal-Weight)")
    plt.xlabel("Date"); plt.ylabel("Relative Equity")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def plot_bootstrap_hist(boot_means: np.ndarray, obs_mean: float, outpath: str) -> None:
    """画自助均值分布 + 观测均值竖线"""
    plt.figure()
    plt.hist(boot_means, bins=40)
    plt.axvline(obs_mean, linestyle="--", label=f"observed mean = {obs_mean:.6f}")
    plt.legend(); plt.title("Block Bootstrap of Mean Excess Return")
    plt.xlabel("bootstrapped mean(d)"); plt.ylabel("frequency")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()
