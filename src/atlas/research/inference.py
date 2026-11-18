"""Statistical inference used across the system (Spec §9.3; Ref 3.11 and 3.4).

* OLS with Newey–West (HAC) standard errors, used for macro betas (§2.6), event studies (§3.5) and
  hypothesis tests (§9.3):
      V_NW = Gamma_0 + sum_{l=1..L} (1 - l/(L+1)) (Gamma_l + Gamma_l^T)
* Newey–West t-statistic of a mean (the IC test), with the default lag L = 4 (T/100)^{2/9}.
* Fama–MacBeth cross-sectional regressions.
* Benjamini–Hochberg false-discovery-rate control.
* Deflated Sharpe ratio (Bailey and López de Prado).
* Stationary block bootstrap (Politis and Romano).
* Effective sample size for overlapping forecasts (Technical Reference, Part 1.6).

Everything is implemented with NumPy/SciPy so results do not depend on optional packages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649


@dataclass
class OLSResult:
    params: np.ndarray
    se: np.ndarray
    tstats: np.ndarray
    resid: np.ndarray
    r2: float
    nobs: int


def default_lags(T: int) -> int:
    """Newey–West rule of thumb L = floor(4 (T/100)^{2/9})."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def ols(y, X, hac_lags: int | None = None, add_const: bool = True) -> OLSResult:
    """OLS with classical or Newey–West standard errors.

    hac_lags=None gives heteroskedasticity-and-autocorrelation-consistent errors with the default
    lag; hac_lags=0 gives White (heteroskedasticity-only) errors.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    L = default_lags(n) if hac_lags is None else hac_lags
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e @ e) / tss if tss > 0 else np.nan
    return OLSResult(beta, se, beta / np.where(se > 0, se, np.nan), e, float(r2), n)


def nw_tstat_mean(x, lags: int | None = None) -> tuple[float, float, float]:
    """Mean, Newey–West standard error and t-statistic of a time series (e.g. monthly IC).

        Var_NW(mean) = (1/T) [gamma_0 + 2 sum_l (1 - l/(L+1)) gamma_l]
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3:
        return float("nan"), float("nan"), float("nan")
    L = default_lags(T) if lags is None else lags
    d = x - x.mean()
    var = d @ d / T
    for l in range(1, L + 1):
        var += 2 * (1 - l / (L + 1)) * (d[l:] @ d[:-l]) / T
    se = np.sqrt(max(var, 0) / T)
    return float(x.mean()), float(se), float(x.mean() / se) if se > 0 else float("nan")


def fama_macbeth(df: pd.DataFrame, y: str, xs: list[str], date_col: str = "date",
                 lags: int | None = None) -> pd.DataFrame:
    """Fama–MacBeth (1973): one cross-sectional regression per date, then a Newey–West t-test on the
    time series of each coefficient. Returns mean coefficient, standard error and t-statistic."""
    coefs = []
    for d, g in df.groupby(date_col):
        g = g[[y, *xs]].dropna()
        if len(g) < len(xs) + 5:
            continue
        X = np.column_stack([np.ones(len(g)), g[xs].values])
        b, *_ = np.linalg.lstsq(X, g[y].values, rcond=None)
        coefs.append(pd.Series(b, index=["const", *xs], name=d))
    C = pd.DataFrame(coefs)
    rows = {c: nw_tstat_mean(C[c].values, lags) for c in C}
    return pd.DataFrame(rows, index=["mean", "se", "t"]).T.assign(n_periods=len(C))


def benjamini_hochberg(pvalues, q: float = 0.05) -> np.ndarray:
    """Boolean array: True where the hypothesis is rejected at false-discovery rate q.

    Sorted p-values p_(k) are compared with (k/m) q; all hypotheses up to the largest passing k are
    rejected. Example (Spec §9 note): 0.001, 0.008, 0.020, 0.040, 0.300 -> first four rejected.
    """
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= (np.arange(1, m + 1) / m) * q
    reject = np.zeros(m, bool)
    if passed.any():
        kmax = np.max(np.nonzero(passed)[0])
        reject[order[: kmax + 1]] = True
    return reject


def deflated_sharpe_ratio(sr_hat: float, sr_trials_var: float, n_trials: int, T: int,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability that the true Sharpe ratio exceeds what N random trials would produce.

        SR_0 = sqrt(V[SR_trials]) [(1 - gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e))]
        DSR  = Phi( (SR_hat - SR_0) sqrt(T - 1) / sqrt(1 - skew SR_hat + (kurt - 1)/4 SR_hat^2) )

    Sharpe ratios are per period (not annualised), consistent with T periods of returns.
    """
    N = max(int(n_trials), 2)
    sr0 = np.sqrt(sr_trials_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / N) + EULER_GAMMA * norm.ppf(1 - 1 / (N * np.e)))
    denom = np.sqrt(max(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2, 1e-12))
    return float(norm.cdf((sr_hat - sr0) * np.sqrt(T - 1) / denom))


def stationary_bootstrap(x, stat, n_boot: int = 10_000, mean_block: float = 6.0, seed: int | None = 0) -> np.ndarray:
    """Politis–Romano stationary bootstrap: blocks of geometric random length (mean `mean_block`).

    Returns the bootstrap distribution of `stat(sample)`; its percentiles give confidence intervals
    that respect autocorrelation.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    T = len(x)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        idx[0] = rng.integers(T)
        new_block = rng.random(T) < p
        starts = rng.integers(T, size=T)
        for t in range(1, T):
            idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % T
        out[b] = stat(x[idx])
    return out


def effective_sample_size(errors, max_lag: int) -> float:
    """n_eff = n / (1 + 2 sum_{l=1}^{h-1} rho_l) for overlapping forecasts (Technical Reference §1.6).

    With monthly 12-month forecasts, consecutive errors share 11 months of returns, so n_eff is
    roughly n/12 or smaller.
    """
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)] - np.nanmean(e)
    n = len(e)
    if n < 3:
        return float(n)
    var = e @ e / n
    rho = [(e[l:] @ e[:-l]) / n / var for l in range(1, min(max_lag, n - 1) + 1)]
    return float(n / max(1 + 2 * np.sum(np.clip(rho, 0, None)), 1.0))
