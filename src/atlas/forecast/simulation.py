"""Joint Monte Carlo (Spec §7.8; Ref 3.9).

For each simulated path:
1. The next regime s is drawn from the HMM transition matrix given the current regime probabilities.
2. Factor returns f ~ mu_s + multivariate Student-t with the regime's factor covariance F_s, drawn as
   f = mu_s + L z sqrt((nu - 2)/W),  L = chol(F_s),  z ~ N(0, I),  W ~ chi^2_nu
   (one W per path, so all factors share the same fat-tailed shock, as in a real crisis).
3. Each stock's specific return eps_i ~ Student-t with variance d_i^2 (Spec §6.3).
4. Returns are combined and sector shocks propagated through the supply-chain network (Spec §6.6):
   r = (I - a W)^{-1} (B f + eps)   when a network is supplied, otherwise r = B f + eps.

Output: an (n_paths x n_stocks) matrix of horizon returns, used for portfolio VaR/ES (Spec §8.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_joint(B: pd.DataFrame, regime_probs: np.ndarray, transition: np.ndarray,
                   regime_mu: list[np.ndarray], regime_cov: list[np.ndarray], d2: pd.Series,
                   n_paths: int = 10_000, nu: float = 5.0, alpha_vec: pd.Series | None = None,
                   network_inverse: np.ndarray | None = None, seed: int | None = 0) -> np.ndarray:
    """Simulate one-period joint returns of all stocks in B (rows stocks, columns factors)."""
    rng = np.random.default_rng(seed)
    n, k = B.shape
    p_next = np.asarray(regime_probs) @ np.asarray(transition)
    states = rng.choice(len(p_next), size=n_paths, p=p_next / p_next.sum())
    f = np.empty((n_paths, k))
    W = rng.chisquare(nu, n_paths)
    scale = np.sqrt((nu - 2) / W)
    for s in np.unique(states):
        idx = np.nonzero(states == s)[0]
        L = np.linalg.cholesky(regime_cov[s] + 1e-12 * np.eye(k))
        z = rng.standard_normal((len(idx), k))
        f[idx] = regime_mu[s] + (z @ L.T) * scale[idx, None]
    sd = np.sqrt(d2.reindex(B.index).fillna(d2.mean()).values)
    eps = rng.standard_t(nu, (n_paths, n)) * np.sqrt((nu - 2) / nu) * sd
    R = f @ B.values.T + eps
    if alpha_vec is not None:
        R += alpha_vec.reindex(B.index).fillna(0.0).values
    if network_inverse is not None:
        R = R @ network_inverse.T
    return R


def portfolio_pnl(R: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return R @ np.asarray(weights)
