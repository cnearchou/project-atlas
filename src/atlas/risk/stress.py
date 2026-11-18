"""Stress tests (Spec §8.5; Ref 3.10).

Historical: past crisis factor returns are replayed through today's exposures,
    Loss_scenario = sum_k (B^T w)_k Δf_k^scenario   (per stock: sum_k B_{i,k} Δf_k).
Hypothetical: shocks to macro variables or news indexes are converted to stock returns through macro
betas (Spec §2.6) and propagated through the supply-chain network (Spec §6.6).
Reverse stress test: the smallest factor shock (in Mahalanobis distance under F) giving a portfolio
loss L* solves  min Δf^T F^{-1} Δf  s.t.  x^T Δf = -L*,  with closed form
    Δf* = -L* F x / (x^T F x),   x = B^T w.
It is the most plausible scenario that produces the loss.

Worked example (Spec §8.5 note): market beta 1.15, growth exposure 0.8, 2022 shock (market -20%,
growth -15%) -> 1.15(-20%) + 0.8(-15%) = -35%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate peak-to-trough factor moves used as default historical scenarios. They are illustrative
# starting values; in production they are replaced by the factor returns estimated over each window.
HISTORICAL_SCENARIOS = {
    "stress_2008": {"market": -0.45, "size": -0.05, "value": -0.10, "momentum": 0.05, "volatility": -0.20},
    "stress_2020": {"market": -0.34, "size": -0.08, "value": -0.12, "momentum": 0.03, "volatility": -0.15},
    "stress_2022": {"market": -0.20, "growth": -0.15, "value": 0.10, "momentum": -0.05},
}


def replay(B: pd.DataFrame, shocks: dict[str, float]) -> pd.Series:
    """Per-stock loss under a factor-shock scenario."""
    s = pd.Series(shocks).reindex(B.columns).fillna(0.0)
    return B @ s


def historical_stress(B: pd.DataFrame, w: pd.Series | None = None, scenarios=HISTORICAL_SCENARIOS) -> pd.DataFrame:
    out = pd.DataFrame({name: replay(B, sh) for name, sh in scenarios.items()})
    if w is not None:
        out.loc["PORTFOLIO"] = (out.mul(w.reindex(B.index).fillna(0.0), axis=0)).sum()
    return out


def hypothetical_stress(macro_betas: pd.DataFrame, macro_shocks: dict[str, float],
                        network_inverse: np.ndarray | None = None) -> pd.Series:
    """Stock returns from macro shocks (betas: rows stocks, columns macro variables)."""
    r = macro_betas @ pd.Series(macro_shocks).reindex(macro_betas.columns).fillna(0.0)
    if network_inverse is not None:
        r = pd.Series(network_inverse @ r.values, index=r.index)
    return r


def reverse_stress(B: pd.DataFrame, w: pd.Series, F: np.ndarray, loss: float = 0.20) -> pd.Series:
    x = B.T.values @ w.reindex(B.index).fillna(0.0).values
    denom = float(x @ F @ x)
    df = -loss * (F @ x) / denom
    return pd.Series(df, index=B.columns, name="reverse_stress_shock")
