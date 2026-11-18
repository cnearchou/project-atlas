"""Macro betas (Spec §2.6; Ref 3.4).

* Market beta: beta = Cov(R_e, R_m) / Var(R_m).
* Release-day regression of asset returns on standardised surprises:
      r_{i,t} = a + beta_i^T S_t + eps_t
  with Newey–West standard errors (heteroskedasticity and autocorrelation).
* Time-varying version: beta follows a random walk, beta_t = beta_{t-1} + eta_t, updated by a
  Kalman filter.
* Betas are estimated at sector level and company estimates are shrunk toward the sector (Spec §6.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.research.inference import ols
from atlas.sectors.betas import vasicek_shrink


def market_beta(r_stock: pd.Series, r_market: pd.Series) -> float:
    d = pd.concat([r_stock, r_market], axis=1).dropna()
    return float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())


def release_day_betas(returns: pd.Series, surprises: pd.DataFrame) -> pd.DataFrame:
    """Regress returns on surprises using release days only. Returns beta and Newey–West se per surprise."""
    S = surprises.dropna(how="all").fillna(0.0)
    d = pd.concat([returns.rename("r"), S], axis=1, join="inner").dropna(subset=["r"])
    res = ols(d["r"].values, d[S.columns].values)
    return pd.DataFrame({"beta": res.params[1:], "se": res.se[1:]}, index=S.columns)


def kalman_time_varying_beta(r: np.ndarray, S: np.ndarray, q: float = 1e-5, r_var: float | None = None) -> np.ndarray:
    """Random-walk betas: beta_t = beta_{t-1} + eta_t (var q), r_t = beta_t^T S_t + eps_t.

    Larger q lets the betas move faster. The observation variance defaults to the variance of r.
    Returns the filtered beta path (T x k).
    """
    T, k = S.shape
    R = np.var(r) if r_var is None else r_var
    b = np.zeros(k)
    P = np.eye(k)
    out = np.zeros((T, k))
    for t in range(T):
        P = P + q * np.eye(k)                       # predict (random walk)
        x = S[t]
        if np.isfinite(r[t]) and np.isfinite(x).all():
            s = x @ P @ x + R
            K = P @ x / s
            b = b + K * (r[t] - x @ b)
            P = P - np.outer(K, x) @ P
        out[t] = b
    return out


def shrink_company_betas(company: pd.DataFrame, sector: pd.DataFrame, prior_var: float) -> pd.Series:
    """Shrink company macro betas toward their sector's (columns beta, se; sector has beta)."""
    return vasicek_shrink(company["beta"], company["se"], sector["beta"].reindex(company.index), prior_var)
