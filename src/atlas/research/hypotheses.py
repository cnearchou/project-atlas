"""Pre-registered hypotheses H1–H3 (Section A.2; Ref 3.12).

H1 (combination and conditioning): ΔIC_t = IC_full,t - IC_base,t, mean > 0 with Newey–West t > 3.
H2 (valuation–expectations gap), Fama–MacBeth each month:
    R_{i,t->t+12} = a + b1 (g_implied - g_base)_i + b2 bear_gap_i + b3 (interaction) + c^T controls + e
    pass if mean b1 < 0 with t < -3 (controls: value, momentum, quality).
    The 12-month returns overlap, so the Newey–West lag is set to at least 12 months.
H3 (slow diffusion of linked information), Fama–MacBeth on 1-month residual returns:
    u_{i,t+1} = a + b1 CustMom_i + b2 (CustMom_i x LowCoverage_i) + c^T controls + e
    pass if b1 > 0 (t > 3) and b2 > 0 (larger effect for low-coverage stocks).

Each hypothesis is written to the registry with its test and threshold *before* the test is run.
"""

from __future__ import annotations

import pandas as pd

from atlas.research.baseline import compare_ic
from atlas.research.inference import fama_macbeth
from atlas.research.registry import Registry

H_TEXT = {
    "H1": "Conditioning known signals on regime, sentiment and crowding raises out-of-sample IC over the baseline composite.",
    "H2": "Stocks whose reverse-DCF implied growth is far above the base rate underperform over 12 months.",
    "H3": "Customer momentum predicts 1-month residual returns, more strongly for low-coverage stocks.",
}
H_TEST = {
    "H1": ("NW t-stat of mean(IC_full - IC_base)", "t > 3"),
    "H2": ("Fama-MacBeth b1 on implied-minus-base growth", "t < -3"),
    "H3": ("Fama-MacBeth b1 on customer momentum and b2 on low-coverage interaction", "t(b1) > 3 and b2 > 0"),
}


def register_all(reg: Registry) -> dict[str, str]:
    """Register H1–H3 once (skips hypotheses already in the registry)."""
    existing = reg.load()
    ids = {}
    for h, text in H_TEXT.items():
        if not existing.empty and (existing["hypothesis_text"] == text).any():
            continue
        test, thr = H_TEST[h]
        ids[h] = reg.register(text, signal_definition=h, test=test, threshold=thr, universe="US common stocks")
    return ids


def test_h1(ic_full: pd.Series, ic_base: pd.Series) -> dict:
    return {"hypothesis": "H1", **compare_ic(ic_full, ic_base)}


def test_h2(panel: pd.DataFrame, y: str = "ret_12m", lags: int = 12) -> dict:
    """panel: date, growth_gap, bear_gap, value, momentum, quality, ret_12m (one row per stock-month)."""
    p = panel.copy()
    p["gap_x_bear"] = p["growth_gap"] * (p["bear_gap"] > 0)
    res = fama_macbeth(p, y, ["growth_gap", "bear_gap", "gap_x_bear", "value", "momentum", "quality"], lags=lags)
    t = res.loc["growth_gap", "t"]
    return {"hypothesis": "H2", "b1": float(res.loc["growth_gap", "mean"]), "t_b1": float(t), "pass": bool(t < -3), "table": res}


def test_h3(panel: pd.DataFrame, y: str = "resid_ret_1m") -> dict:
    """panel: date, cust_mom, low_coverage (0/1), momentum, size and the 1-month residual return."""
    p = panel.copy()
    p["cust_x_low"] = p["cust_mom"] * p["low_coverage"]
    res = fama_macbeth(p, y, ["cust_mom", "cust_x_low", "momentum", "size"])
    t1 = res.loc["cust_mom", "t"]
    b2 = res.loc["cust_x_low", "mean"]
    return {"hypothesis": "H3", "b1": float(res.loc["cust_mom", "mean"]), "t_b1": float(t1), "b2": float(b2),
            "pass": bool(t1 > 3 and b2 > 0), "table": res}
