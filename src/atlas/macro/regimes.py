"""Economic regimes with a Gaussian hidden Markov model (Spec §2.5; Ref 3.4).

Processing:
* Monthly observations y_t = [growth factor, inflation factor, Δ credit spread, curve slope].
* Emissions: y_t | s ~ N(mu_s, Sigma_s).
* Number of states: models with 2 to 6 states are fitted and BIC picks one.
* Parameters (T, mu_s, Sigma_s) are estimated with the Baum–Welch EM algorithm.
* States are labelled from their means (high growth and low inflation -> "Expansion", etc.).
* Probabilities are FILTERED with the forward algorithm, never smoothed:
      alpha_t(j) ∝ p(y_t | j) sum_i alpha_{t-1}(i) T_ij,   sum_j alpha_t(j) = 1
  Smoothed probabilities use later data and would introduce look-ahead bias.

The implementation is self-contained (NumPy/SciPy); hmmlearn is not required. Note that hmmlearn's
`predict_proba` returns smoothed probabilities, so the filter here is needed even when it is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


@dataclass
class GaussianHMM:
    n_states: int
    pi0: np.ndarray = None         # initial probabilities
    T: np.ndarray = None           # transition matrix T_ij = P(s_{t+1}=j | s_t=i)
    means: np.ndarray = None
    covs: np.ndarray = None
    loglik: float = -np.inf

    # ---- emission densities ---------------------------------------------------------------------
    def _emission(self, Y: np.ndarray) -> np.ndarray:
        B = np.column_stack([multivariate_normal(self.means[j], self.covs[j], allow_singular=True).pdf(Y)
                             for j in range(self.n_states)])
        return np.maximum(B, 1e-300)

    # ---- forward filter (Spec §2.5) ---------------------------------------------------------
    def filter(self, Y: np.ndarray) -> np.ndarray:
        """Filtered probabilities P(s_t = j | y_1..y_t): only data up to t is used."""
        B = self._emission(Y)
        alpha = np.zeros_like(B)
        a = self.pi0 * B[0]
        alpha[0] = a / a.sum()
        for t in range(1, len(Y)):
            a = (alpha[t - 1] @ self.T) * B[t]
            alpha[t] = a / a.sum()
        return alpha

    # ---- Baum–Welch EM -------------------------------------------------------------------------
    def fit(self, Y: np.ndarray, n_iter: int = 200, tol: float = 1e-6, seed: int = 0) -> "GaussianHMM":
        rng = np.random.default_rng(seed)
        n, d = Y.shape
        K = self.n_states
        # Initialisation: k-means-like split on the first variable, sticky transitions.
        order = np.argsort(Y[:, 0])
        groups = np.array_split(order, K)
        self.means = np.array([Y[g].mean(0) for g in groups]) + 1e-3 * rng.standard_normal((K, d))
        self.covs = np.array([np.cov(Y.T) + 1e-3 * np.eye(d) for _ in range(K)])
        self.T = np.full((K, K), 0.1 / max(K - 1, 1)) + np.eye(K) * (0.9 - 0.1 / max(K - 1, 1))
        self.pi0 = np.full(K, 1.0 / K)
        prev = -np.inf
        for _ in range(n_iter):
            B = self._emission(Y)
            # E-step: scaled forward–backward recursions.
            alpha = np.zeros((n, K)); c = np.zeros(n)
            alpha[0] = self.pi0 * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ self.T) * B[t]; c[t] = alpha[t].sum(); alpha[t] /= c[t]
            beta = np.ones((n, K))
            for t in range(n - 2, -1, -1):
                beta[t] = (self.T @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            xi = np.zeros((K, K))
            for t in range(n - 1):
                x = alpha[t][:, None] * self.T * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
                xi += x / x.sum()
            # M-step
            self.pi0 = gamma[0]
            self.T = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            self.means = (gamma.T @ Y) / w[:, None]
            for j in range(K):
                D = Y - self.means[j]
                self.covs[j] = (gamma[:, j, None] * D).T @ D / w[j] + 1e-4 * np.eye(d)
            ll = np.log(c).sum()
            if abs(ll - prev) < tol:
                break
            prev = ll
        self.loglik = float(prev if np.isfinite(prev) else ll)
        return self

    def n_params(self, d: int) -> int:
        K = self.n_states
        return K * (K - 1) + (K - 1) + K * d + K * d * (d + 1) // 2

    def bic(self, n: int, d: int) -> float:
        return self.n_params(d) * np.log(n) - 2 * self.loglik


def select_hmm(Y: np.ndarray, states=range(2, 7), seed: int = 0) -> GaussianHMM:
    """Fit HMMs with 2–6 states and return the one with the lowest BIC."""
    best = None
    for k in states:
        m = GaussianHMM(k).fit(Y, seed=seed)
        if best is None or m.bic(*Y.shape) < best.bic(*Y.shape):
            best = m
    return best


def label_states(model: GaussianHMM, growth_col: int = 0, inflation_col: int = 1) -> list[str]:
    """Descriptive labels from state means. Labels help interpretation; the model never sees them."""
    labels = []
    g_med = np.median(model.means[:, growth_col])
    for mu in model.means:
        hi_g = mu[growth_col] >= g_med
        hi_i = mu[inflation_col] > np.median(model.means[:, inflation_col]) if model.n_states > 2 else False
        if hi_g and not hi_i:
            labels.append("Expansion")
        elif hi_g and hi_i:
            labels.append("Overheating")
        elif not hi_g and hi_i:
            labels.append("Stagflation")
        else:
            labels.append("Recession" if mu[growth_col] == model.means[:, growth_col].min() else "Slowdown")
    return labels


def regime_probabilities(model: GaussianHMM, Y: pd.DataFrame) -> pd.DataFrame:
    """Filtered regime probabilities as a DataFrame with descriptive column names."""
    probs = model.filter(Y.values)
    labels = label_states(model)
    cols, seen = [], {}
    for lab in labels:  # make duplicate labels unique (e.g. two "Slowdown" states)
        seen[lab] = seen.get(lab, 0) + 1
        cols.append(lab if seen[lab] == 1 else f"{lab}_{seen[lab]}")
    return pd.DataFrame(probs, index=Y.index, columns=[f"p_{c.lower()}" for c in cols])


def next_period(probs_now: np.ndarray, T: np.ndarray) -> np.ndarray:
    """One-step regime forecast pi_{t+1} = pi_t T (worked example in Spec §2.5)."""
    return probs_now @ T
