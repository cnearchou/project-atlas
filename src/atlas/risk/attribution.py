"""Risk and performance attribution (Spec §8.3 and §8.9; Ref 3.10).

Euler risk decomposition:
    sigma_p = sqrt(w^T Sigma w),  MCR_i = (Sigma w)_i / sigma_p,  CR_i = w_i MCR_i,  sum_i CR_i = sigma_p
Factor versus specific variance with Sigma = B F B^T + D:
    sigma_p^2 = (B^T w)^T F (B^T w) + w^T D w

Ex-post P&L attribution (Spec §8.9):
    PnL_t = sum_k (B^T w)_k f_{k,t}  +  sum_i w_i u_{i,t}  -  costs_t
Signal attribution of the specific part to feature k in proportion to SHAP contributions:
    PnL_k = sum_i w_i u_i phi_{i,k} / sum_j phi_{i,j}
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def euler_decomposition(w: pd.Series, Sigma: pd.DataFrame) -> pd.DataFrame:
    w = w.reindex(Sigma.index).fillna(0.0)
    Sw = Sigma.values @ w.values
    vol = float(np.sqrt(w.values @ Sw))
    mcr = Sw / vol
    cr = w.values * mcr
    return pd.DataFrame({"weight": w.values, "mcr": mcr, "cr": cr, "cr_pct": cr / vol}, index=Sigma.index)


def factor_specific_split(w: pd.Series, B: pd.DataFrame, F: np.ndarray, d2: pd.Series) -> dict:
    w = w.reindex(B.index).fillna(0.0)
    x = B.values.T @ w.values
    fac = float(x @ F @ x)
    spec = float((w.values ** 2 * d2.reindex(B.index).fillna(d2.mean()).values).sum())
    by_factor = pd.Series(x * (F @ x), index=B.columns)   # Euler split of factor variance by factor
    tot = fac + spec
    return {"factor_var": fac, "specific_var": spec, "factor_risk_pct": fac / tot, "idio_risk_pct": spec / tot,
            "factor_exposures": pd.Series(x, index=B.columns), "factor_var_by_factor": by_factor}


def pnl_attribution(w: pd.Series, B: pd.DataFrame, f: pd.Series, u: pd.Series, costs: float = 0.0) -> dict:
    w = w.reindex(B.index).fillna(0.0)
    x = B.T @ w
    factor_pnl = x * f.reindex(B.columns).fillna(0.0)
    specific = float((w * u.reindex(B.index).fillna(0.0)).sum())
    return {"factor": factor_pnl, "factor_total": float(factor_pnl.sum()), "specific": specific, "costs": -costs,
            "total": float(factor_pnl.sum()) + specific - costs}


def signal_attribution(w: pd.Series, u: pd.Series, shap: pd.DataFrame) -> pd.Series:
    """Split the specific P&L by feature using each stock's share of SHAP contributions."""
    s = shap.reindex(w.index).fillna(0.0)
    denom = s.sum(axis=1).replace(0, np.nan)
    share = s.div(denom, axis=0).fillna(0.0)
    return share.mul(w * u.reindex(w.index).fillna(0.0), axis=0).sum()
