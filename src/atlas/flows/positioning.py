"""Flows and positioning calculations (Spec §4.2; Ref 3.6).

ETF flow attribution:
    Flow_E,t = ΔShares outstanding_E,t x NAV_E,t
    Flow_i,t = sum_E Flow_E,t x w_{i,E}
    Pressure_i,t = Flow_i,t / ADV$_i
Short interest:
    Days to cover = SI / ADV_20d;  ΔSI = (SI_t - SI_{t-1}) / float;  Utilisation = on loan / lendable
    Squeeze score = mean of standardised days to cover, utilisation, borrow fee and momentum
13F crowding (holdings lagged 45 days):
    sim(f, g) = h_f . h_g / (|h_f| |h_g|)
    Crowding_i = average sim over pairs of funds holding i + z(hedge-fund ownership % of float)
Dealer gamma exposure (assumes dealers are long calls and short puts; a modelling convention):
    GEX = sum_calls Gamma OI 100 S^2 0.01 - sum_puts Gamma OI 100 S^2 0.01
Index inclusion: logistic model of P(inclusion), demand Q = P x passive share x float, and
square-root impact  ΔP/P ~= Y sigma sqrt(Q / ADV).
Insider score = (purchases - sales, excluding 10b5-1 plan sales, 90 days) / market cap; cluster flag
when at least three distinct insiders bought within 30 days.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from atlas.data.standardise import rank_gauss
from atlas.fundamentals.options import bs_gamma


# ---- ETF flows --------------------------------------------------------------------------------
def etf_flow_pressure(etf: pd.DataFrame, holdings: pd.DataFrame, adv_dollars: pd.Series) -> pd.Series:
    """etf: (etf_id, date, shares_outstanding, nav); holdings: (etf_id, entity_id, weight).

    Returns flow pressure per (date, entity_id). This is an estimate of ETF-attributed demand, not a
    record of actual trades in the stock.
    """
    e = etf.sort_values(["etf_id", "date"]).copy()
    e["flow"] = e.groupby("etf_id")["shares_outstanding"].diff() * e["nav"]
    m = e.merge(holdings, on="etf_id")
    stock_flow = (m["flow"] * m["weight"]).groupby([m["date"], m["entity_id"]]).sum()
    adv = adv_dollars.reindex(stock_flow.index.get_level_values("entity_id")).values
    return (stock_flow / adv).rename("flow_pressure")


# ---- Short interest ----------------------------------------------------------------------------
def short_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """df columns: entity_id, date, short_interest, adv_20d, float_shares, on_loan, lendable,
    borrow_fee, momentum. Missing optional columns are skipped."""
    d = df.sort_values(["entity_id", "date"]).copy()
    d["days_to_cover"] = d["short_interest"] / d["adv_20d"]
    d["d_si"] = d.groupby("entity_id")["short_interest"].diff() / d["float_shares"]
    if {"on_loan", "lendable"} <= set(d):
        d["utilisation"] = d["on_loan"] / d["lendable"]
    parts = [c for c in ("days_to_cover", "utilisation", "borrow_fee", "momentum") if c in d]
    z = pd.concat([d.groupby("date")[c].transform(rank_gauss) for c in parts], axis=1)
    d["squeeze_score"] = z.mean(axis=1)
    return d


# ---- 13F crowding ----------------------------------------------------------------------------
def crowding_scores(holdings: pd.DataFrame, hf_ownership_pct: pd.Series | None = None,
                    max_pairs: int = 5000, seed: int = 0) -> pd.Series:
    """holdings: (fund_id, entity_id, weight) for one 13F date (already lagged 45 days).

    For each stock, the cosine similarity of every pair of funds holding it is averaged; for stocks
    with many holders a random sample of pairs keeps the computation fast.
    """
    H = holdings.pivot_table(index="fund_id", columns="entity_id", values="weight", fill_value=0.0)
    Hn = H.values / np.maximum(np.linalg.norm(H.values, axis=1, keepdims=True), 1e-12)
    sim = Hn @ Hn.T
    rng = np.random.default_rng(seed)
    out = {}
    for j, eid in enumerate(H.columns):
        holders = np.nonzero(H.values[:, j] > 0)[0]
        if len(holders) < 2:
            out[eid] = np.nan
            continue
        pairs = list(itertools.combinations(holders, 2))
        if len(pairs) > max_pairs:
            pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
        out[eid] = float(np.mean([sim[a, b] for a, b in pairs]))
    s = pd.Series(out, name="crowding")
    if hf_ownership_pct is not None:
        s = s + rank_gauss(hf_ownership_pct.reindex(s.index)).fillna(0.0)
    return s


# ---- Dealer gamma ----------------------------------------------------------------------------
def gamma_exposure(chain: pd.DataFrame, S: float, r: float) -> float:
    """chain: strike, expiry_years, iv, open_interest, cp ('C' or 'P'). Dollar gamma per 1% move."""
    g = bs_gamma(S, chain["strike"].values, chain["expiry_years"].values, r, chain["iv"].values)
    dollar = g * chain["open_interest"].values * 100 * S ** 2 * 0.01
    sign = np.where(chain["cp"].str.upper().str.startswith("C"), 1.0, -1.0)
    return float(np.nansum(sign * dollar))


# ---- Index inclusion and impact ------------------------------------------------------------
def fit_inclusion_model(history: pd.DataFrame) -> LogisticRegression:
    """Logistic regression of past additions on (cutoff rank - stock rank) and eligibility flags.

    history columns: rank_gap, eligible (0/1), added (0/1).
    """
    X = history[["rank_gap", "eligible"]].values
    return LogisticRegression().fit(X, history["added"].values)


def expected_index_demand(p_inclusion: float, passive_share: float, float_shares: float) -> float:
    """Q = P(inclusion) x passive ownership share x float shares."""
    return p_inclusion * passive_share * float_shares


def sqrt_impact(Q: float, adv: float, sigma: float, Y: float = 0.8) -> float:
    """Square-root market impact ΔP/P ~= Y sigma sqrt(Q / ADV) (units must be consistent).

    Worked example (Spec §4.2 note): Q = 6M shares, ADV = 2M, sigma = 2.5%, Y = 0.8 -> 3.46%.
    """
    return Y * sigma * np.sqrt(Q / adv)


# ---- Insiders ---------------------------------------------------------------------------------
def insider_scores(trades: pd.DataFrame, market_cap: pd.Series, asof, window_days: int = 90,
                   cluster_days: int = 30) -> pd.DataFrame:
    """trades: entity_id, insider_id, date, value (positive buy, negative sell), plan_10b5_1 (bool)."""
    asof = pd.Timestamp(asof)
    t = trades[(trades["date"] > asof - pd.Timedelta(days=window_days)) & (trades["date"] <= asof)]
    t = t[~((t["value"] < 0) & t.get("plan_10b5_1", False))]  # scheduled plan sales carry no information
    net = t.groupby("entity_id")["value"].sum()
    recent_buys = t[(t["value"] > 0) & (t["date"] > asof - pd.Timedelta(days=cluster_days))]
    cluster = recent_buys.groupby("entity_id")["insider_id"].nunique() >= 3
    out = pd.DataFrame({"insider_score": net / market_cap.reindex(net.index)})
    out["insider_cluster"] = cluster.reindex(out.index).fillna(False).astype(bool)
    return out
