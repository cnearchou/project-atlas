"""Projection parameters (Spec §1.4; Ref 3.3).

Panel regression for growth persistence across all companies and years in a sector:

    (g_{i,t+1} - g_bar) = phi (g_{i,t} - g_bar) + eps_{i,t}

* phi is estimated separately by growth bucket, because very high growth fades faster.
* sigma_g is the standard deviation of the residuals eps.
* The same procedure gives psi and sigma_m for margins.
* Joint shock covariance: the residual pairs (eps, eta) are stacked and their covariance gives the
  growth–margin correlation used in the Monte Carlo (Spec §1.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ARFit:
    persistence: float      # phi or psi
    mean: float             # g_bar or m_sector
    sigma: float            # residual standard deviation
    se: float               # standard error of the persistence estimate (used for parameter uncertainty)
    n: int
    residuals: pd.Series


def fit_panel_ar1(panel: pd.DataFrame, col: str, mean: float | None = None, id_col: str = "entity_id") -> ARFit:
    """Pooled AR(1) of deviations from a long-run mean, estimated by OLS through the origin.

    With x_t = value_t - mean, phi = sum(x_t x_{t+1}) / sum(x_t^2). Its standard error is
    sigma / sqrt(sum x_t^2), which feeds the parameter-uncertainty step (Spec §1.5, step 6).
    """
    df = panel[[id_col, col]].dropna()
    mu = df[col].mean() if mean is None else mean
    x = df[col] - mu
    x_next = x.groupby(df[id_col]).shift(-1)
    ok = x_next.notna()
    xt, xn = x[ok].values, x_next[ok].values
    denom = np.sum(xt ** 2)
    phi = float(np.sum(xt * xn) / denom) if denom > 0 else 0.0
    resid = pd.Series(xn - phi * xt, index=x[ok].index)
    sigma = float(resid.std(ddof=1))
    se = sigma / np.sqrt(denom) if denom > 0 else np.nan
    return ARFit(phi, float(mu), sigma, float(se), int(ok.sum()), resid)


def fit_by_bucket(panel: pd.DataFrame, col: str, n_buckets: int = 3, id_col: str = "entity_id") -> dict[int, ARFit]:
    """Persistence estimated separately for each growth bucket (low, middle, high)."""
    buckets = pd.qcut(panel[col], n_buckets, labels=False, duplicates="drop")
    return {int(b): fit_panel_ar1(panel[buckets == b], col, mean=panel[col].mean(), id_col=id_col)
            for b in sorted(buckets.dropna().unique())}


def shock_covariance(growth_fit: ARFit, margin_fit: ARFit, wacc_sigma: float = 0.01,
                     wacc_growth_corr: float = -0.3) -> np.ndarray:
    """Covariance of the (growth, margin, WACC) shocks used by the Monte Carlo.

    Growth and margin residuals are aligned on their common rows to estimate their correlation. The
    WACC shock is not observed in accounting data, so its volatility and correlation with growth are
    assumptions (defaults follow the example in Spec §1.5: sigma = 1 point, rho = -0.3).
    """
    joint = pd.concat([growth_fit.residuals.rename("g"), margin_fit.residuals.rename("m")], axis=1).dropna()
    rho_gm = float(joint.corr().iloc[0, 1]) if len(joint) > 5 else 0.0
    sg, sm, sw = growth_fit.sigma, margin_fit.sigma, wacc_sigma
    corr = np.array([[1.0, rho_gm, wacc_growth_corr],
                     [rho_gm, 1.0, wacc_growth_corr * 0.5],
                     [wacc_growth_corr, wacc_growth_corr * 0.5, 1.0]])
    sd = np.array([sg, sm, sw])
    cov = corr * np.outer(sd, sd)
    # Guard against a non-positive-definite matrix (possible with assumed correlations).
    w, v = np.linalg.eigh(cov)
    return v @ np.diag(np.clip(w, 1e-12, None)) @ v.T
