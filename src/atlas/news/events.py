"""Event detection and event studies (Spec §3.4–§3.5; Ref 3.5).

Event detection: an event is declared when |Z_t| > 2, or when the article count n is unusually high
under a Poisson model with the rolling 60-day mean as baseline:  P(N >= n | Poisson(lambda)) < 0.01.

Event study:
1. Market model fitted on days -250 to -30:  E[R_i] = alpha_i + beta_i R_m
2. Abnormal return  AR_{i,t} = R_{i,t} - (alpha_i + beta_i R_{m,t})
3. CAR_i = sum_{t=-1..+5} AR_{i,t}
4. SCAR_i = CAR_i / (sigma_i sqrt(L))
5. Boehmer–Musumeci–Poulsen test: t_BMP = mean(SCAR) sqrt(N) / sd(SCAR)
6. When many firms share an event date, a portfolio approach avoids overstating significance.

Shrinkage of event-type x sector impacts toward the event-type mean:
    Impact_cell = w mean_cell + (1 - w) mean_type,   w = tau^2 / (tau^2 + sigma_cell^2 / n_cell)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def detect_events(z: pd.Series, counts: pd.Series, z_threshold: float = 2.0, p_threshold: float = 0.01,
                  baseline_window: int = 60) -> pd.DataFrame:
    """Flag event days from the standardised index and from article-volume spikes."""
    lam = counts.shift(1).rolling(baseline_window, min_periods=20).mean()
    p_spike = pd.Series(poisson.sf(counts - 1, lam), index=counts.index)  # P(N >= n)
    out = pd.DataFrame({"z": z, "count": counts, "p_volume": p_spike})
    out["event_z"] = out["z"].abs() > z_threshold
    out["event_volume"] = out["p_volume"] < p_threshold
    out["event"] = out["event_z"] | out["event_volume"]
    return out


def event_study(r_stock: pd.Series, r_market: pd.Series, event_date, est_window=(-250, -30),
                event_window=(-1, 5)) -> dict:
    """CAR and SCAR for one company and one event (returns indexed by trading date)."""
    d = pd.concat([r_stock.rename("r"), r_market.rename("m")], axis=1).dropna()
    pos = d.index.searchsorted(pd.Timestamp(event_date))
    est = d.iloc[max(pos + est_window[0], 0):pos + est_window[1]]
    if len(est) < 60:
        return {"car": np.nan, "scar": np.nan}
    beta = np.cov(est["r"], est["m"])[0, 1] / est["m"].var()
    alpha = est["r"].mean() - beta * est["m"].mean()
    sigma = (est["r"] - alpha - beta * est["m"]).std()
    win = d.iloc[pos + event_window[0]:pos + event_window[1] + 1]
    ar = win["r"] - (alpha + beta * win["m"])
    L = len(win)
    car = ar.sum()
    return {"car": float(car), "scar": float(car / (sigma * np.sqrt(L))), "alpha": alpha, "beta": beta, "L": L}


def bmp_test(scars: np.ndarray) -> float:
    """Boehmer–Musumeci–Poulsen t-statistic, robust to event-induced volatility."""
    s = np.asarray(scars, float)
    s = s[np.isfinite(s)]
    return float(s.mean() * np.sqrt(len(s)) / s.std(ddof=1)) if len(s) > 2 else float("nan")


def portfolio_car(cars_same_day: pd.DataFrame) -> pd.Series:
    """Average CAR across firms sharing an event date, so each date counts once (step 6)."""
    return cars_same_day.groupby("event_date")["car"].mean()


def shrink_impacts(cells: pd.DataFrame) -> pd.DataFrame:
    """cells: event_type, sector, mean_car, sd_car, n. Adds shrunk_impact.

    tau^2 is the variance across sector means within each event type (method of moments), so cells
    with few or noisy observations are pulled toward their event-type average.
    """
    out = cells.copy()
    type_mean = out.groupby("event_type")["mean_car"].transform("mean")
    tau2 = out.groupby("event_type")["mean_car"].transform("var").fillna(0.0).clip(lower=1e-8)
    w = tau2 / (tau2 + out["sd_car"] ** 2 / out["n"].clip(lower=1))
    out["shrunk_impact"] = w * out["mean_car"] + (1 - w) * type_mean
    out["t_stat"] = out["mean_car"] / (out["sd_car"] / np.sqrt(out["n"].clip(lower=1)))
    return out
