"""Distribution and scenario probabilities (Spec §7.6 and Section A.3; Ref 3.9 and 3.12).

1. Historical standardised residuals  e_t = r_t / sigma_t  are calculated.
2. The degrees of freedom nu of a Student-t are fitted to e by maximum likelihood.
3. The t is scaled to the forecast volatility: R ~ mu + sigma sqrt((nu - 2)/nu) T_nu.
4. With threshold x and horizon 12 months (Section A.3):
       P(bull) = 1 - F_nu((x - mu) / s),   P(bear) = F_nu((-x - mu) / s),   P(base) = 1 - P(bull) - P(bear),
   where s = sigma sqrt((nu - 2)/nu).
Absolute scenarios use mu = beta^T E[f] + alpha and x = 15%. Relative scenarios (return in excess of
the sector) use mu = alpha, the residual volatility and x = 10%; they are the primary output because
they do not depend on the weak factor-timing estimate (Technical Reference, Part 1.6).
5. Cross-check against the quantile models: stocks are flagged when q10/q90 from the trees fall far
   outside the t-implied 10th/90th percentiles.

Worked example (Spec §7.6 note): mu = 2%, sigma = 32%, nu = 5 gives P(bull) = 31.1%, P(bear) = 26.2%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


def fit_nu(std_resid, bounds=(2.1, 50.0)) -> float:
    """MLE of nu for a unit-variance Student-t."""
    e = np.asarray(std_resid, float)
    e = e[np.isfinite(e)]

    def nll(nu):
        s = np.sqrt((nu - 2) / nu)
        return -np.sum(stats.t.logpdf(e / s, nu) - np.log(s))
    return float(minimize_scalar(nll, bounds=bounds, method="bounded").x)


def t_probabilities(mu, sigma, nu: float, threshold: float) -> pd.DataFrame:
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    s = sigma * np.sqrt((nu - 2) / nu)
    bull = 1 - stats.t.cdf((threshold - mu) / s, nu)
    bear = stats.t.cdf((-threshold - mu) / s, nu)
    return pd.DataFrame({"p_bull": bull, "p_base": 1 - bull - bear, "p_bear": bear})


def scenario_table(mu_abs: pd.Series, sigma_total: pd.Series, alpha: pd.Series, sigma_resid: pd.Series,
                   nu: float, abs_threshold: float = 0.15, rel_threshold: float = 0.10) -> pd.DataFrame:
    """Absolute and relative scenario probabilities for every stock (12-month horizon)."""
    a = t_probabilities(mu_abs.values, sigma_total.reindex(mu_abs.index).values, nu, abs_threshold)
    r = t_probabilities(alpha.reindex(mu_abs.index).values, sigma_resid.reindex(mu_abs.index).values, nu, rel_threshold)
    r.columns = [f"{c}_rel" for c in r.columns]
    return pd.concat([a, r], axis=1).set_index(mu_abs.index)


def quantile_disagreement(mu, sigma, nu: float, q10, q90, tol: float = 0.5) -> np.ndarray:
    """Flag when the tree quantiles differ from the t-implied 10/90 percentiles by more than tol x sigma."""
    s = np.asarray(sigma) * np.sqrt((nu - 2) / nu)
    t10, t90 = np.asarray(mu) + s * stats.t.ppf(0.1, nu), np.asarray(mu) + s * stats.t.ppf(0.9, nu)
    return (np.abs(np.asarray(q10) - t10) > tol * np.asarray(sigma)) | (np.abs(np.asarray(q90) - t90) > tol * np.asarray(sigma))
