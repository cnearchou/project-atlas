"""Vector autoregression (Spec §2.4; Ref 3.4).

    x_t = c + sum_{j=1..p} A_j x_{t-j} + eps_t

Processing steps:
1. Variables: inflation factor, growth factor, change in unemployment, policy rate, credit spread.
2. Lag order p chosen by BIC = k ln(n) - 2 ln(L).
3. Each equation estimated by OLS.
4. Stability: every eigenvalue of the companion matrix has |lambda| < 1.
5. Impulse responses with a Cholesky ordering (slow-moving variables first, policy rate last).
6. Forecast distribution: residual bootstrap of 10,000 future paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VARResult:
    p: int
    c: np.ndarray            # intercepts (k,)
    A: list[np.ndarray]      # lag matrices A_1..A_p (k x k)
    resid: np.ndarray        # (T-p) x k
    sigma: np.ndarray        # residual covariance
    names: list[str]

    def companion(self) -> np.ndarray:
        k = len(self.c)
        top = np.hstack(self.A)
        if self.p == 1:
            return top
        bottom = np.hstack([np.eye(k * (self.p - 1)), np.zeros((k * (self.p - 1), k))])
        return np.vstack([top, bottom])

    def is_stable(self) -> bool:
        """Step 4: shocks die out only if all companion eigenvalues lie inside the unit circle."""
        return bool(np.all(np.abs(np.linalg.eigvals(self.companion())) < 1))

    def irf(self, horizon: int = 8) -> np.ndarray:
        """Step 5: orthogonalised impulse responses, shape (horizon+1, k responses, k shocks).

        The Cholesky factor of the residual covariance identifies the shocks; the result depends on
        the column order of the data, which must therefore be stated (slow variables first).
        """
        k = len(self.c)
        P = np.linalg.cholesky(self.sigma)
        C = self.companion()
        out = np.zeros((horizon + 1, k, k))
        M = np.eye(C.shape[0])
        for h in range(horizon + 1):
            out[h] = M[:k, :k] @ P
            M = C @ M
        return out

    def simulate(self, history: np.ndarray, steps: int, n_paths: int = 10_000, seed: int | None = 0) -> np.ndarray:
        """Step 6: bootstrap future paths by resampling past residuals. Returns (n_paths, steps, k)."""
        rng = np.random.default_rng(seed)
        k = len(self.c)
        paths = np.zeros((n_paths, steps, k))
        lags = np.repeat(history[-self.p:][None, ::-1, :], n_paths, axis=0)  # most recent first
        for s in range(steps):
            shock = self.resid[rng.integers(len(self.resid), size=n_paths)]
            x = self.c + sum(lags[:, j, :] @ self.A[j].T for j in range(self.p)) + shock
            paths[:, s, :] = x
            lags = np.concatenate([x[:, None, :], lags[:, :-1, :]], axis=1)
        return paths


def _design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    T = len(Y)
    X = np.column_stack([np.ones(T - p)] + [Y[p - j - 1:T - j - 1] for j in range(p)])
    return X, Y[p:]


def fit_var(df: pd.DataFrame, p: int) -> VARResult:
    """Step 3: equation-by-equation OLS (identical to joint OLS when all equations share regressors)."""
    Y = df.dropna().values
    k = Y.shape[1]
    X, Yt = _design(Y, p)
    B, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    resid = Yt - X @ B
    sigma = resid.T @ resid / (len(Yt) - X.shape[1])
    A = [B[1 + j * k:1 + (j + 1) * k].T for j in range(p)]
    return VARResult(p, B[0], A, resid, sigma, list(df.columns))


def select_lag(df: pd.DataFrame, max_lag: int = 6) -> int:
    """Step 2: choose p by BIC, comparing all lags on the same estimation sample."""
    Y = df.dropna().values
    T, k = Y.shape
    best, best_p = np.inf, 1
    for p in range(1, max_lag + 1):
        X, Yt = _design(Y[max_lag - p:], p)
        B, *_ = np.linalg.lstsq(X, Yt, rcond=None)
        e = Yt - X @ B
        sigma = e.T @ e / len(Yt)
        n_params = k * (k * p + 1)
        bic = len(Yt) * np.log(np.linalg.det(sigma)) + n_params * np.log(len(Yt))
        if bic < best:
            best, best_p = bic, p
    return best_p
