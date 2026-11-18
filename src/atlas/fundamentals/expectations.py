"""Market-expectation comparisons (Spec §1.6; Ref 3.3).

Reverse DCF: Brent's root-finding on f(g) = V(g) - Price gives g*, the growth rate implied by the
current share price.

Standardised unexpected earnings (SUE):
    Analyst-based      SUE = (Actual EPS - Consensus EPS) / sigma(analyst estimates)
    Time-series based  SUE = (EPS_q - EPS_{q-4} - drift) / sigma(past 8 quarterly changes)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from atlas.fundamentals.monte_carlo import DCFInputs, run_dcf_monte_carlo


def deterministic_value(x: DCFInputs) -> float:
    """The DCF value with all shocks switched off (a single central path)."""
    zero = replace(x, shock_cov=np.eye(3) * 1e-16, se_phi=0.0, se_psi=0.0, se_g_bar=0.0,
                   g_inf_sd=0.0, roic_inf_spread_sd=0.0, roic_new_sd=0.0)
    return float(run_dcf_monte_carlo(zero, n=2, seed=0, parameter_uncertainty=False).mean)


def reverse_dcf(x: DCFInputs, price: float, param: str = "g0", lo: float = -0.5, hi: float = 1.5) -> float:
    """Solve V(param) = price for the chosen input (default: starting growth g0).

    Brent's method narrows a bracketing interval until the value equals the price. If the price lies
    outside the values reachable in [lo, hi], NaN is returned rather than an extrapolated answer.
    """
    f = lambda v: deterministic_value(replace(x, **{param: v})) - price  # noqa: E731
    try:
        return float(brentq(f, lo, hi, xtol=1e-6))
    except ValueError:
        return float("nan")


def sue_analyst(actual_eps: pd.Series, consensus: pd.Series, dispersion: pd.Series) -> pd.Series:
    """Analyst-based SUE. Zero dispersion (one analyst) gives a missing value."""
    return (actual_eps - consensus) / dispersion.where(dispersion > 0)


def sue_time_series(eps: pd.DataFrame, id_col: str = "entity_id", eps_col: str = "eps", window: int = 8) -> pd.Series:
    """Time-series SUE from a quarterly panel sorted by entity and date.

    The year-on-year change EPS_q - EPS_{q-4} is compared with its own drift (the mean of the last
    eight changes) and scaled by their standard deviation, both measured before quarter q.
    """
    d = eps.groupby(id_col)[eps_col].diff(4)
    drift = d.groupby(eps[id_col]).transform(lambda s: s.shift(1).rolling(window, min_periods=4).mean())
    sd = d.groupby(eps[id_col]).transform(lambda s: s.shift(1).rolling(window, min_periods=4).std())
    return (d - drift) / sd.where(sd > 0)
