"""Elections and laws (Spec §3.6–§3.7; Ref 3.5).

Elections: prediction-market prices are normalised to remove the fee (vig),
    p_i = price_i / sum_j price_j,
and sector sensitivity is estimated from daily probability changes,
    r_sector,t = alpha + delta Δp_t + beta r_m,t + eps_t.
At resolution the remaining move is delta (100 - p_now) if the outcome occurs and delta (0 - p_now)
if it does not. Because responses near resolution can be non-linear, delta is also estimated
separately for low (< 30%), middle (30–70%) and high (> 70%) probability ranges.

Laws: bill passage probability from a logistic regression on historical bills
    P(pass) = 1 / (1 + exp(-(b0 + b1 stage + b2 sponsor_in_majority + b3 ln(cosponsors) + b4 bipartisan)))
Company exposure: cosine similarity between the bill text and the company's 10-K Item 1; the sign of
the effect comes from the impact estimate because cosine similarity is unsigned:
    Expected impact_i = P(pass) x exposure_i x impact if passed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from atlas.research.inference import ols
from atlas.sectors.peers import cosine_matrix, embed_texts


def normalise_prices(prices: pd.Series) -> pd.Series:
    """Remove the market margin so outcome probabilities sum to one."""
    return prices / prices.sum()


def election_sensitivity(r_sector: pd.Series, p_pct: pd.Series, r_market: pd.Series) -> dict:
    """delta (return per 1-point probability change) overall and by probability band."""
    d = pd.concat([r_sector.rename("r"), p_pct.diff().rename("dp"), r_market.rename("m"),
                   p_pct.rename("p")], axis=1).dropna()
    overall = ols(d["r"].values, d[["dp", "m"]].values)
    bands = {}
    for name, lo, hi in (("low", 0, 30), ("middle", 30, 70), ("high", 70, 100)):
        sub = d[(d["p"] >= lo) & (d["p"] < hi)]
        if len(sub) > 20:
            bands[name] = float(ols(sub["r"].values, sub[["dp", "m"]].values).params[1])
    return {"delta": float(overall.params[1]), "delta_se": float(overall.se[1]), "delta_by_band": bands}


def resolution_moves(delta: float, p_now_pct: float) -> dict:
    """Remaining sector move if the outcome happens (win) or not (lose).

    Worked example (Spec §3.6 note): delta = 0.12% per point, p_now = 55 -> +5.4% / -6.6%.
    """
    return {"win": delta * (100 - p_now_pct), "lose": delta * (0 - p_now_pct)}


def fit_bill_passage(bills: pd.DataFrame) -> LogisticRegression:
    """bills columns: stage (ordinal), sponsor_in_majority (0/1), cosponsors, bipartisan (0/1), passed (0/1)."""
    X = np.column_stack([bills["stage"], bills["sponsor_in_majority"], np.log1p(bills["cosponsors"]),
                         bills["bipartisan"]])
    return LogisticRegression(max_iter=1000).fit(X, bills["passed"])


def bill_features(bills: pd.DataFrame) -> np.ndarray:
    return np.column_stack([bills["stage"], bills["sponsor_in_majority"], np.log1p(bills["cosponsors"]),
                            bills["bipartisan"]])


def policy_exposure(bill_text: str, company_texts: pd.Series) -> pd.Series:
    """Cosine similarity between a bill and each company's business description."""
    E = embed_texts([bill_text, *company_texts.tolist()])
    S = cosine_matrix(E)
    return pd.Series(S[0, 1:], index=company_texts.index, name="exposure")


def expected_policy_impact(p_pass: float, exposure: pd.Series, impact_if_passed: float) -> pd.Series:
    return (p_pass * exposure * impact_if_passed).rename("expected_impact")
