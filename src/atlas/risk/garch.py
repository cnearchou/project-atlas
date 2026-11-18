"""GJR-GARCH(1,1) estimation (Spec §8.1; Ref 3.10).

    sigma^2_t = omega + (alpha + gamma I_{t-1}) eps^2_{t-1} + beta sigma^2_{t-1},
    I_{t-1} = 1 if eps_{t-1} < 0
Parameters maximise the Student-t log-likelihood subject to omega > 0, alpha, gamma, beta >= 0 and
alpha + gamma/2 + beta < 1 (covariance stationarity). The `arch` package is used when installed;
otherwise an equivalent maximum-likelihood fit with scipy is run.

Worked example (Spec §8.1 note, gamma = 0): omega = 0.000002, alpha = 0.08, beta = 0.90,
eps_{t-1} = -4%, sigma_{t-1} = 2%  ->  sigma^2_t = 0.00049, sigma_t = 2.21%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

from atlas._optional import optional_import


@dataclass
class GARCHResult:
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float
    mu: float
    sigma: np.ndarray        # in-sample conditional volatility sigma_t
    next_sigma: float        # one-step-ahead forecast sigma_{T+1}
    backend: str

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.omega / max(1 - self.persistence, 1e-8)))

    def forecast_variance(self, h: int) -> np.ndarray:
        """Multi-step variance path E[sigma^2_{T+k}], mean-reverting to the long-run variance."""
        lr = self.omega / max(1 - self.persistence, 1e-8)
        k = np.arange(h)
        return lr + (self.next_sigma ** 2 - lr) * self.persistence ** k

    def horizon_vol(self, h: int) -> float:
        return float(np.sqrt(self.forecast_variance(h).sum()))


def gjr_recursion(eps: np.ndarray, omega, alpha, gamma, beta, s2_0: float | None = None) -> np.ndarray:
    """Conditional variance path; returns T+1 values (the last is the one-step forecast).

    The recursion s2_{t+1} = beta s2_t + x_t with x_t = omega + (alpha + gamma I_t) eps_t^2 is a
    first-order linear filter in s2, so it is evaluated with scipy.signal.lfilter instead of a Python
    loop (about 100 times faster, which matters when every stock is refitted weekly).
    """
    s0 = np.var(eps) if s2_0 is None else s2_0
    x = omega + (alpha + gamma * (eps < 0)) * eps ** 2
    y, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])
    return np.concatenate([[s0], y])


def _t_loglik(eps, s2, nu):
    """Log-likelihood of a unit-variance Student-t scaled by sigma_t."""
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return np.sum(c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(eps ** 2 / (s2 * (nu - 2))))


def fit_gjr_garch(returns, use_arch: bool = True) -> GARCHResult:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    arch = optional_import("arch") if use_arch else None
    if arch is not None:
        am = arch.arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        res = am.fit(disp="off")
        p = res.params
        f = res.forecast(horizon=1).variance.values[-1, 0]
        return GARCHResult(p["omega"] / 1e4, p["alpha[1]"], p["gamma[1]"], p["beta[1]"], p["nu"], p["mu"] / 100,
                           np.asarray(res.conditional_volatility) / 100, float(np.sqrt(f)) / 100, "arch")

    mu0 = r.mean()
    var0 = r.var()

    def nll(th):
        mu, om, a, g, b, nu = th
        if om <= 0 or a < 0 or g < 0 or b < 0 or a + g / 2 + b >= 0.999 or nu <= 2.05:
            return 1e10
        eps = r - mu
        s2 = gjr_recursion(eps, om, a, g, b, var0)[:-1]
        if np.any(s2 <= 0):
            return 1e10
        return -_t_loglik(eps, s2, nu)

    x0 = [mu0, var0 * 0.05, 0.05, 0.05, 0.88, 8.0]
    best = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-6})
    mu, om, a, g, b, nu = best.x
    s2 = gjr_recursion(r - mu, om, a, g, b, var0)
    return GARCHResult(float(om), float(a), float(g), float(b), float(nu), float(mu),
                       np.sqrt(s2[:-1]), float(np.sqrt(s2[-1])), "scipy")


def standardised_residuals(returns, res: GARCHResult) -> np.ndarray:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    return (r - res.mu) / res.sigma
