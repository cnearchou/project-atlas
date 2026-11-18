"""Company betas and shrinkage (Spec §6.4; Ref 3.8).

Rolling 252-day regression gives beta_hat_i and its standard error se_i. The Vasicek formula shrinks
the company estimate toward the sector beta:

    beta_adj = sigma_prior^2 / (sigma_prior^2 + se_i^2) beta_hat_i + se_i^2 / (sigma_prior^2 + se_i^2) beta_sector

Precise estimates keep most of their own weight; noisy ones are pulled toward the sector.

Hierarchical model for other parameters (growth persistence, margins, macro betas):
    theta_i ~ N(theta_sector, tau^2),   theta_sector ~ N(theta_market, tau_sector^2)
This module estimates it by empirical Bayes (method of moments), which gives the same shrinkage
structure in closed form. A full PyMC version is provided for when PyMC is installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas._optional import require


def rolling_beta(r_stock: pd.Series, r_market: pd.Series, window: int = 252) -> pd.DataFrame:
    """Rolling OLS beta and its standard error (classical errors) over `window` days."""
    d = pd.concat([r_stock.rename("y"), r_market.rename("x")], axis=1).dropna()
    cov = d["y"].rolling(window).cov(d["x"])
    var = d["x"].rolling(window).var()
    beta = cov / var
    alpha = d["y"].rolling(window).mean() - beta * d["x"].rolling(window).mean()
    resid_var = (d["y"].rolling(window).var() - beta ** 2 * var) * (window - 1) / (window - 2)
    se = np.sqrt(resid_var.clip(lower=0) / (var * (window - 1)))
    return pd.DataFrame({"beta": beta, "se": se, "alpha": alpha})


def vasicek_shrink(beta_hat, se, beta_sector, prior_var: float):
    """Vasicek shrinkage. Works element-wise on floats, arrays or aligned Series.

    Worked example (Spec §6.4 note): beta 1.6, se 0.4, sector 1.1, prior variance 0.04 -> weight 0.2,
    adjusted beta 1.2.
    """
    w = prior_var / (prior_var + np.asarray(se) ** 2)
    out = w * np.asarray(beta_hat) + (1 - w) * np.asarray(beta_sector)
    return pd.Series(out, index=beta_hat.index) if isinstance(beta_hat, pd.Series) else out


def empirical_bayes_shrink(estimates: pd.Series, ses: pd.Series, groups: pd.Series) -> pd.Series:
    """Two-level shrinkage of company estimates toward their group (sector) means.

    For each group, tau^2 (true between-company variance) is estimated by method of moments as the
    variance of the estimates minus the average sampling variance (floored at a small positive value).
    Each estimate then receives weight tau^2 / (tau^2 + se_i^2). Group means are themselves shrunk
    toward the overall mean in the same way, giving the theta_sector ~ N(theta_market, ...) level.
    """
    df = pd.DataFrame({"x": estimates, "se2": ses ** 2, "g": groups}).dropna()
    grand = np.average(df["x"], weights=1 / df["se2"])
    out = pd.Series(np.nan, index=estimates.index)
    gstats = df.groupby("g").agg(m=("x", "mean"), v=("x", "var"), s2=("se2", "mean"), n=("x", "size"))
    tau2_sector = max(gstats["m"].var() - (gstats["v"] / gstats["n"]).mean(), 1e-8) if len(gstats) > 1 else 1e-8
    for g, sub in df.groupby("g"):
        st = gstats.loc[g]
        # shrink the sector mean toward the market (grand) mean
        w_sec = tau2_sector / (tau2_sector + st["s2"] / st["n"])
        sector_mean = w_sec * st["m"] + (1 - w_sec) * grand
        tau2 = max((st["v"] if np.isfinite(st["v"]) else 0.0) - st["s2"], 1e-8)
        w = tau2 / (tau2 + sub["se2"])
        out.loc[sub.index] = w * sub["x"] + (1 - w) * sector_mean
    return out


def pymc_hierarchical(estimates: pd.Series, ses: pd.Series, groups: pd.Series, draws: int = 1000):
    """Full Bayesian version of the hierarchy with PyMC (pip install 'project-atlas[bayes]').

    Returns posterior means of theta_i. Measurement model: estimate_i ~ N(theta_i, se_i^2).
    """
    pm = require("pymc", "bayes")
    codes, uniques = pd.factorize(groups)
    with pm.Model():
        mu_market = pm.Normal("mu_market", 0, 10)
        tau_sector = pm.HalfNormal("tau_sector", 1)
        theta_sector = pm.Normal("theta_sector", mu_market, tau_sector, shape=len(uniques))
        tau = pm.HalfNormal("tau", 1)
        theta = pm.Normal("theta", theta_sector[codes], tau, shape=len(estimates))
        pm.Normal("obs", theta, ses.values, observed=estimates.values)
        trace = pm.sample(draws, tune=draws, progressbar=False)
    return pd.Series(trace.posterior["theta"].mean(("chain", "draw")).values, index=estimates.index)
