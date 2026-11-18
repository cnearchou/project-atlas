"""Supply-chain network (Spec §6.6; Ref 3.8).

1. Customer disclosures are extracted from 10-K filings ("Customer A accounted for 18% of revenue").
2. W_ij is the share of company i's revenue attributable to company j; supplier links are added with
   their direction recorded.
3. W is row-normalised where data permits.
4. The propagation parameter alpha is chosen so the spectral radius rho(alpha W) < 1.
5. Propagated shock:  s_total = (I - alpha W)^{-1} s_initial = sum_k (alpha W)^k s_initial
6. Customer momentum: CustMom_i = sum_j W_ij return_{j, last month}
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Matches sentences such as "Customer A accounted for 18% of total revenue".
_CUSTOMER_RE = re.compile(r"(?P<name>[A-Z][\w&.,\- ]{1,60}?)\s+(?:accounted for|represented|comprised)\s+"
                          r"(?:approximately\s+)?(?P<pct>\d{1,2}(?:\.\d+)?)\s?%\s+of\s+(?:our\s+|total\s+|net\s+)*"
                          r"(?:revenue|revenues|net sales|sales)", re.IGNORECASE)


def extract_customer_disclosures(text: str) -> list[tuple[str, float]]:
    """Return (customer name, revenue share) pairs found in filing text."""
    return [(m.group("name").strip(" ,."), float(m.group("pct")) / 100) for m in _CUSTOMER_RE.finditer(text)]


def build_matrix(links: pd.DataFrame, ids: list[str], row_normalise: bool = True) -> np.ndarray:
    """W from a link table with columns id_i, id_j, weight."""
    idx = {e: k for k, e in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)))
    for _, r in links.iterrows():
        if r["id_i"] in idx and r["id_j"] in idx:
            W[idx[r["id_i"]], idx[r["id_j"]]] += r["weight"]
    if row_normalise:
        s = W.sum(axis=1, keepdims=True)
        W = np.where(s > 1, W / np.where(s == 0, 1, s), W)  # only rows summing above 1 are rescaled
    return W


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0


def choose_alpha(W: np.ndarray, target: float = 0.8, margin: float = 0.95) -> float:
    """Largest alpha <= target that keeps rho(alpha W) below `margin` (so the series converges)."""
    rho = spectral_radius(W)
    return target if rho == 0 else min(target, margin / rho)


def propagate(W: np.ndarray, s_initial: np.ndarray, alpha: float) -> np.ndarray:
    """s_total = (I - alpha W)^{-1} s_initial. Solved as a linear system for numerical stability.

    Worked example (Spec §6.6 note): foundry shock -10% with W21 = 0.4, W31 = 0.3, W32 = 0.5,
    alpha = 0.8 gives -3.2% for the device maker and -3.68% for the software company.
    """
    if spectral_radius(alpha * W) >= 1:
        raise ValueError("rho(alpha W) must be below 1 for the propagation series to converge.")
    n = len(W)
    total = np.linalg.solve(np.eye(n) - alpha * W, s_initial)
    return total


def customer_momentum(W: np.ndarray, last_month_returns: np.ndarray) -> np.ndarray:
    """CustMom_i = sum_j W_ij r_j (recent performance of company i's customers)."""
    return W @ np.nan_to_num(last_month_returns)
