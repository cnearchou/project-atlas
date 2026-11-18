"""Dynamic factor model and Kalman filter (Spec §2.3; Ref 3.4).

Model:
    Observation  y_t = Lambda f_t + e_t,   e_t ~ N(0, R)
    State        f_t = A f_{t-1} + u_t,    u_t ~ N(0, Q)
with f_t = (growth factor, inflation factor).

Kalman filter, run every day:
    Predict  f^-_t = A f_{t-1};   P^-_t = A P_{t-1} A^T + Q
    Update   K_t = P^-_t H^T (H P^-_t H^T + R)^{-1}
             f_t = f^-_t + K_t (y_t - H f^-_t);   P_t = (I - K_t H) P^-_t

Mixed frequencies: on days with no release, the rows of y and H for missing indicators are dropped;
if nothing is released, only the prediction step runs.

Parameters are estimated with the two-step method of Doz, Giannone and Reichlin (principal components
then regressions), a standard approximation to EM. When statsmodels is installed,
`statsmodels.tsa.statespace.DynamicFactorMQ` can be used instead for full EM estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanResult:
    factors: pd.DataFrame        # filtered factor estimates
    variances: np.ndarray        # P_t for each date (T x k x k)


class KalmanFilter:
    """Linear Gaussian Kalman filter that skips missing observations."""

    def __init__(self, A: np.ndarray, Lambda: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A, self.Lambda, self.Q, self.R = A, Lambda, Q, R

    def filter(self, Y: pd.DataFrame, f0: np.ndarray | None = None, P0: np.ndarray | None = None) -> KalmanResult:
        k = self.A.shape[0]
        f = np.zeros(k) if f0 is None else f0
        P = np.eye(k) * 10.0 if P0 is None else P0          # diffuse start: large initial uncertainty
        out_f, out_P = [], []
        for _, row in Y.iterrows():
            # Predict
            f = self.A @ f
            P = self.A @ P @ self.A.T + self.Q
            # Update only with indicators released today
            obs = row.notna().values
            if obs.any():
                H = self.Lambda[obs]
                Rk = self.R[np.ix_(obs, obs)]
                S = H @ P @ H.T + Rk
                K = P @ H.T @ np.linalg.inv(S)
                f = f + K @ (row.values[obs] - H @ f)
                P = (np.eye(k) - K @ H) @ P
            out_f.append(f.copy())
            out_P.append(P.copy())
        return KalmanResult(pd.DataFrame(out_f, index=Y.index, columns=[f"factor_{i}" for i in range(k)]),
                            np.array(out_P))


def estimate_dfm(Y: pd.DataFrame, n_factors: int = 2) -> tuple[KalmanFilter, pd.DataFrame]:
    """Two-step estimation of the dynamic factor model.

    1. Standardise each indicator and extract principal components from the balanced part of the panel.
    2. Lambda: regress each indicator on the factors; R: residual variances (diagonal).
    3. A and Q: a VAR(1) on the factors and its residual covariance.
    Returns the fitted filter and the standardisation (mean, std) needed to apply it to new data.
    """
    scale = pd.DataFrame({"mean": Y.mean(), "std": Y.std()})
    Z = (Y - scale["mean"]) / scale["std"]
    balanced = Z.dropna()
    u, s, vt = np.linalg.svd(balanced.values - balanced.values.mean(0), full_matrices=False)
    F = u[:, :n_factors] * s[:n_factors] / np.sqrt(len(balanced))
    Lam, *_ = np.linalg.lstsq(F, balanced.values, rcond=None)
    Lambda = Lam.T                                            # (n_indicators x n_factors)
    resid = balanced.values - F @ Lam
    R = np.diag(np.maximum(resid.var(axis=0), 1e-4))
    A, *_ = np.linalg.lstsq(F[:-1], F[1:], rcond=None)
    A = A.T
    u_res = F[1:] - F[:-1] @ A.T
    Q = np.cov(u_res.T) + 1e-6 * np.eye(n_factors)
    return KalmanFilter(A, Lambda, Q, R), scale


def nowcast(Y: pd.DataFrame, n_factors: int = 2) -> KalmanResult:
    """Fit the model and return filtered factors for every date in Y (missing values allowed).

    Factor signs are arbitrary in a factor model, so factor 0 is oriented to correlate positively with
    the first column of Y (put the growth indicator first) and factor 1 with the second column.
    """
    kf, scale = estimate_dfm(Y, n_factors)
    Z = (Y - scale["mean"]) / scale["std"]
    res = kf.filter(Z)
    for i in range(min(n_factors, Y.shape[1])):
        c = np.corrcoef(res.factors.iloc[:, i], Z.iloc[:, i].fillna(Z.iloc[:, i].mean()))[0, 1]
        if c < 0:
            res.factors.iloc[:, i] *= -1
    res.factors.columns = ["growth_factor", "inflation_factor"][:n_factors] + [
        f"factor_{i}" for i in range(2, n_factors)]
    return res
