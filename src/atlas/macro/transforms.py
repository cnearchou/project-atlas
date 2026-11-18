"""Macro processing steps (Spec §2.2; Ref 3.4).

Step 1  Point-in-time vintages: a backtest dated d uses the value known on d (see
        atlas.ingest.fred.FredClient.vintages and atlas.data.point_in_time).
Step 2  Stationarity transforms:
            pi_t     = 100 ln(CPI_t / CPI_{t-12})     annual inflation
            Δy_t     = 100 ln(GDP_t / GDP_{t-1})      growth
            rates: level and first difference; unemployment: first difference
        and the Augmented Dickey–Fuller test for a unit root.
Step 3  Standardised surprises and the economic surprise index:
            S_k = (Actual_k - Consensus_k) / sigma(historical surprises of k)
            ESI_t = sum_k w_k S_{k,t}, with exponential decay (half-life about one month)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas._optional import optional_import

# MacKinnon (2010) asymptotic critical values for the ADF test with a constant.
ADF_CRITICAL = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


def log_change(s: pd.Series, periods: int) -> pd.Series:
    """100 ln(X_t / X_{t-periods}); periods=12 for annual inflation from monthly CPI."""
    return 100 * np.log(s / s.shift(periods))


def adf_test(x: pd.Series, max_lag: int | None = None) -> dict:
    """Augmented Dickey–Fuller test: Δy_t = a + gamma y_{t-1} + sum delta_j Δy_{t-j} + e_t.

    H0: gamma = 0 (unit root). statsmodels' adfuller is used when installed (it gives exact
    MacKinnon p-values); otherwise the t-statistic is computed here and compared with the 5% critical
    value. A rejection is evidence against a unit root, not proof of stationarity (Spec §2.2).
    """
    x = pd.Series(x).dropna()
    sm = optional_import("statsmodels.tsa.stattools")
    if sm is not None:
        stat, p, lags, *_ = sm.adfuller(x.values, maxlag=max_lag, autolag="AIC")
        return {"stat": float(stat), "pvalue": float(p), "lags": int(lags), "stationary_5pct": p < 0.05}
    p = max_lag if max_lag is not None else int(np.floor(12 * (len(x) / 100) ** 0.25))
    dy = x.diff()
    cols = {"y_lag": x.shift(1)}
    for j in range(1, p + 1):
        cols[f"dy_lag{j}"] = dy.shift(j)
    df = pd.DataFrame(cols).assign(dy=dy).dropna()
    X = np.column_stack([np.ones(len(df)), df.drop(columns="dy").values])
    b, *_ = np.linalg.lstsq(X, df["dy"].values, rcond=None)
    e = df["dy"].values - X @ b
    s2 = e @ e / (len(df) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    stat = b[1] / se
    return {"stat": float(stat), "pvalue": float("nan"), "lags": p, "stationary_5pct": stat < ADF_CRITICAL[0.05]}


def standardised_surprise(actual: pd.Series, consensus: pd.Series, min_history: int = 12) -> pd.Series:
    """S_k = (actual - consensus) / std of past surprises, using only surprises before each release."""
    raw = actual - consensus
    sd = raw.shift(1).expanding(min_periods=min_history).std()
    return raw / sd


def economic_surprise_index(surprises: pd.DataFrame, half_life_days: float = 21.0,
                            weights: dict[str, float] | None = None) -> pd.Series:
    """Decayed, weighted sum of standardised surprises.

    `surprises` has a DatetimeIndex of release dates and one column per indicator (NaN when there
    was no release). Each release's influence halves every `half_life_days` days.
    """
    s = surprises.sort_index()
    w = pd.Series(weights or {c: 1.0 for c in s.columns})
    daily = s.mul(w, axis=1).sum(axis=1, min_count=1).resample("D").sum(min_count=1).fillna(0.0)
    decay = 0.5 ** (1 / half_life_days)
    out = np.zeros(len(daily))
    acc = 0.0
    for i, v in enumerate(daily.values):
        acc = acc * decay + v
        out[i] = acc
    return pd.Series(out, index=daily.index, name="esi")
