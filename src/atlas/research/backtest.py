"""Quantile-portfolio backtester (Spec §9.2 and Build Phase 2).

At each rebalance date t the stocks are sorted into quantiles by the signal known at t; the
equal-weighted (or cap-weighted) return of each quantile over the next period is recorded, and the
long–short return is top minus bottom. Costs are charged on turnover:
    r_net,t = r_LS,t - c x turnover_t
The signal must already be lagged correctly by the caller (known at t, returns from t to t+1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_portfolios(signal: pd.DataFrame, fwd_returns: pd.DataFrame, n_q: int = 5,
                        cost_per_turnover: float = 0.0, min_names: int = 25) -> dict:
    s, r = signal.align(fwd_returns, join="inner")
    q_ret, weights = {}, {}
    for d in s.index:
        a, b = s.loc[d], r.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() < min_names:
            continue
        q = pd.qcut(a[ok].rank(method="first"), n_q, labels=False) + 1
        q_ret[d] = b[ok].groupby(q).mean()
        w = pd.Series(0.0, index=a.index)
        top, bot = q[q == n_q].index, q[q == 1].index
        w[top] = 1 / len(top)
        w[bot] = -1 / len(bot)
        weights[d] = w
    Q = pd.DataFrame(q_ret).T.sort_index()
    W = pd.DataFrame(weights).T.sort_index().fillna(0.0)
    ls = Q[n_q] - Q[1]
    to = 0.5 * W.diff().abs().sum(axis=1)
    to.iloc[0] = 0.5 * W.iloc[0].abs().sum()
    net = ls - cost_per_turnover * to.reindex(ls.index)
    return {"quantile_returns": Q, "long_short": ls.rename("ls"), "long_short_net": net.rename("ls_net"),
            "turnover": to, "weights": W, "monotonic": bool(np.all(np.diff(Q.mean().values) > 0))}
