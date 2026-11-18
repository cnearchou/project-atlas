"""News indexes (Spec §3.3; Ref 3.5).

Sentiment-intensity index for each topic, sector and day:
    x_t = sum_j w_source,j p_topic,j s_j r_j n_j
    I_t = lambda I_{t-1} + (1 - lambda) x_t
    Z_t = (I_t - mu_I,past) / sigma_I,past       (expanding-window statistics, no look-ahead)
The smoothing lambda is chosen by grid search over 0.5–0.95, maximising the correlation between Z_t
and next-day sector returns in the training period only. Half-life in days = ln(0.5) / ln(lambda).

Uncertainty index (EPU style): share of each source's articles matching the uncertainty term sets,
standardised by source, averaged across sources and rescaled to mean 100.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ECONOMY = r"\b(econom\w*)\b"
UNCERTAIN = r"\b(uncertain\w*)\b"
POLICY = r"\b(congress|regulat\w*|central bank|federal reserve|deficit|legislation|white house|tariff\w*)\b"


def daily_signal(articles: pd.DataFrame, topic: str) -> pd.Series:
    """x_t from processed articles with columns session_date, w_source, p_<topic>, sentiment, relevance, novelty."""
    contrib = (articles["w_source"] * articles[f"p_{topic}"] * articles["sentiment"]
               * articles["relevance"] * articles["novelty"])
    return contrib.groupby(articles["session_date"]).sum().sort_index()


def smooth(x: pd.Series, lam: float) -> pd.Series:
    """I_t = lambda I_{t-1} + (1 - lambda) x_t on a business-day calendar (days without news get x = 0)."""
    x = x.asfreq("B", fill_value=0.0) if isinstance(x.index, pd.DatetimeIndex) else x
    return x.ewm(alpha=1 - lam, adjust=False).mean()


def standardise_expanding(I: pd.Series, min_periods: int = 20) -> pd.Series:
    """Z_t using the mean and standard deviation of past values only (shifted by one day)."""
    mu = I.expanding(min_periods).mean().shift(1)
    sd = I.expanding(min_periods).std().shift(1)
    return (I - mu) / sd


def fit_lambda(x: pd.Series, next_returns: pd.Series, grid=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
               train_end=None) -> tuple[float, pd.DataFrame]:
    """Grid-search lambda on the training period; the test period never influences the choice."""
    res = []
    for lam in grid:
        z = standardise_expanding(smooth(x, lam))
        d = pd.concat([z.rename("z"), next_returns.rename("r")], axis=1).dropna()
        if train_end is not None:
            d = d[d.index <= pd.Timestamp(train_end)]
        res.append({"lambda": lam, "corr": d["z"].corr(d["r"]) if len(d) > 10 else np.nan,
                    "half_life_days": np.log(0.5) / np.log(lam)})
    table = pd.DataFrame(res)
    best = float(table.loc[table["corr"].idxmax(), "lambda"]) if table["corr"].notna().any() else 0.8
    return best, table


def uncertainty_index(articles: pd.DataFrame) -> pd.Series:
    """EPU-style count index. articles: source, session_date, text."""
    t = articles["text"].str.lower()
    match = t.str.contains(ECONOMY, flags=re.I) & t.str.contains(UNCERTAIN, flags=re.I) & t.str.contains(POLICY, flags=re.I)
    share = match.groupby([articles["source"], articles["session_date"]]).mean().unstack(0)
    z = share / share.std()          # each source standardised by its own standard deviation
    avg = z.mean(axis=1)
    return (avg / avg.mean() * 100).rename("uncertainty_index")
