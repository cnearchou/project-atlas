"""Fundamentals output table (Spec §1.7).

fundamentals_features(id, date, quality_z, value_z, growth_z, f_score, altman_z, m_score,
    dcf_mean, dcf_sd, p_bull_dcf, p_base_dcf, p_bear_dcf, implied_g, implied_phi,
    sue, eps_surprise_score, opt_p_bull, opt_p_bear, bear_gap)

The bear gap compares the DCF bear probability with the options-implied one. Two adjustments are made
before it is used: it is measured against its own history for each stock (risk-neutral
probabilities overstate the downside), and option probabilities are taken from the expiry closest to
12 months:

    bear_gap_adj = (p_bear_DCF - p_bear_opt) - mean(past bear_gap of the stock)
"""

from __future__ import annotations

import pandas as pd

from atlas.fundamentals.expectations import reverse_dcf
from atlas.fundamentals.monte_carlo import (DCFInputs, convert_to_12m_returns, run_dcf_monte_carlo,
                                            scenario_probabilities)

COLUMNS = ["entity_id", "date", "quality_z", "value_z", "growth_z", "f_score", "altman_z", "m_score",
           "dcf_mean", "dcf_sd", "p_bull_dcf", "p_base_dcf", "p_bear_dcf", "implied_g", "implied_phi",
           "sue", "eps_surprise_score", "opt_p_bull", "opt_p_bear", "bear_gap", "bear_gap_adj"]


def dcf_row(x: DCFInputs, price: float, kappa: float, factor_return_12m: float, resid_vol_12m: float,
            n: int = 10_000, threshold: float = 0.15, seed: int = 0) -> dict:
    """Run the full Monte Carlo for one company and return the DCF columns of the output table."""
    mc = run_dcf_monte_carlo(x, n=n, seed=seed)
    r12 = convert_to_12m_returns(mc.values, price, kappa, factor_return_12m, resid_vol_12m, seed=seed + 1)
    probs = scenario_probabilities(r12, threshold)
    return {"dcf_mean": mc.mean, "dcf_sd": mc.sd, "p_bull_dcf": probs["p_bull"], "p_base_dcf": probs["p_base"],
            "p_bear_dcf": probs["p_bear"], "implied_g": reverse_dcf(x, price, "g0"),
            "implied_phi": reverse_dcf(x, price, "phi", 0.0, 0.99)}


def adjusted_bear_gap(df: pd.DataFrame) -> pd.Series:
    """bear_gap minus the stock's own expanding historical mean (excluding the current date)."""
    gap = df["p_bear_dcf"] - df["opt_p_bear"]
    hist = gap.groupby(df["entity_id"]).transform(lambda s: s.shift(1).expanding().mean())
    return gap - hist.fillna(0.0)
