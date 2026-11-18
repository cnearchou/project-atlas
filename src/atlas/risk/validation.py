"""Risk-model validation (Spec §8.7; Ref 3.10).

Kupiec proportion-of-failures test, with x breaches in N days and breach probability p:
    LR_uc = -2 ln[ (1-p)^{N-x} p^x ] + 2 ln[ (1 - x/N)^{N-x} (x/N)^x ]  ~ chi^2(1);  reject if > 3.84
Christoffersen independence test: with n_ij the number of days in state j after state i
(0 = no breach, 1 = breach) and pi_ij the transition frequencies,
    LR_ind = -2 ln[ (1-pi)^{n00+n10} pi^{n01+n11} ] + 2 ln[ (1-pi_01)^{n00} pi_01^{n01} (1-pi_11)^{n10} pi_11^{n11} ]
Conditional coverage LR_cc = LR_uc + LR_ind ~ chi^2(2).
Bias statistic: b = sd(r_t / sigma_hat_t), which should be close to 1; the 95% band for a correct
model is roughly 1 +/- sqrt(2/T).

Worked example (Spec §8.7 note): 45 breaches in 500 days at p = 5% (25 expected) -> rejected.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(breaches, p: float = 0.05) -> dict:
    b = np.asarray(breaches, bool)
    N, x = len(b), int(b.sum())
    pi = x / N
    ll0 = _xlogy(N - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(N - x, 1 - pi) + _xlogy(x, pi)
    lr = -2 * (ll0 - ll1)
    return {"breaches": x, "expected": N * p, "lr": float(lr), "p_value": float(chi2.sf(lr, 1)), "reject": lr > 3.84}


def christoffersen(breaches) -> dict:
    b = np.asarray(breaches, int)
    prev, cur = b[:-1], b[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = -2 * (ll0 - ll1)
    return {"lr_ind": float(lr), "p_value": float(chi2.sf(lr, 1)), "reject": lr > 3.84, "pi01": pi01, "pi11": pi11}


def conditional_coverage(breaches, p: float = 0.05) -> dict:
    k, c = kupiec(breaches, p), christoffersen(breaches)
    lr = k["lr"] + c["lr_ind"]
    return {"lr_cc": lr, "p_value": float(chi2.sf(lr, 2)), "reject": lr > 5.99}


def bias_statistic(returns, sigma_hat) -> dict:
    z = np.asarray(returns, float) / np.asarray(sigma_hat, float)
    z = z[np.isfinite(z)]
    b = float(z.std(ddof=1))
    band = np.sqrt(2 / len(z))
    return {"bias": b, "lower": 1 - 1.96 * band, "upper": 1 + 1.96 * band, "ok": abs(b - 1) <= 1.96 * band}
