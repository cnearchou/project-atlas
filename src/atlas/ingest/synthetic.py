"""Synthetic data generator.

Licensed data (CRSP, Compustat, I/B/E/S) cannot be published on GitHub (Technical Reference, Part
2.1), so the repository ships with a generator that produces data in the same layout as the real
sources. The generated market has a known structure, which makes it useful for testing:

* returns follow a factor model (market + sector + style + stock-specific noise), as in Spec §6.3;
* a weak momentum effect and a weak quality effect are planted, so the research harness (Spec §9)
  should detect them, and a pure-noise signal is included, which it should reject;
* volatility clusters (GARCH-like), as assumed in Spec §8.1;
* quarterly fundamentals are released with a filing lag, so point-in-time handling (Spec §0.4)
  matters;
* some companies are delisted with a negative delisting return (Spec §0.5);
* occasional splits and dividends test the corporate-action logic (Spec §0.3);
* macro series switch between expansion and recession regimes (Spec §2.5).

All randomness comes from one seed, so every run is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas._compat import QUARTER_END

SECTORS = ["Technology", "Financials", "Health Care", "Energy", "Industrials", "Consumer", "Utilities", "Materials"]


@dataclass
class SyntheticMarket:
    prices: pd.DataFrame          # entity_id, date, price, dividend, split_ratio, shares_outstanding, volume, ret
    companies: pd.DataFrame       # entity_id, ticker, sector, listed_from, delisted_on
    fundamentals: pd.DataFrame    # point-in-time: entity_id, field, event_date, knowledge_date, value
    delistings: pd.DataFrame      # entity_id, date, delisting_return
    macro: pd.DataFrame           # date-indexed monthly macro series and true regime
    factor_returns: pd.DataFrame  # true daily factor returns (for validating Spec §6.3 estimates)


def make_market(n_stocks: int = 300, n_days: int = 1260, start: str = "2018-01-01", seed: int = 7) -> SyntheticMarket:
    """Generate a synthetic equity market.

    n_days=1260 is about five years of trading days. The default size keeps the full demo fast on a
    laptop; larger values are fine for stress-testing performance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    ids = [f"A{i:04d}" for i in range(n_stocks)]
    sectors = rng.choice(SECTORS, size=n_stocks)

    # --- Macro regime path (monthly), used to modulate market drift and volatility ---------------
    months = pd.period_range(dates[0], dates[-1], freq="M")
    regime = np.zeros(len(months), dtype=int)
    T = np.array([[0.95, 0.05], [0.20, 0.80]])  # 0 = expansion, 1 = recession
    for t in range(1, len(months)):
        regime[t] = rng.choice(2, p=T[regime[t - 1]])
    month_of_day = dates.to_period("M")
    regime_daily = pd.Series(regime, index=months).reindex(month_of_day).values

    # --- Factor returns -----------------------------------------------------------------------
    mkt_mu = np.where(regime_daily == 0, 0.0004, -0.0006)
    mkt_sd = np.where(regime_daily == 0, 0.009, 0.018)
    f_mkt = mkt_mu + mkt_sd * rng.standard_normal(n_days)
    f_sector = {s: 0.006 * rng.standard_normal(n_days) for s in SECTORS}
    f_size = 0.003 * rng.standard_normal(n_days)
    f_value = 0.003 * rng.standard_normal(n_days)

    # --- Stock characteristics ----------------------------------------------------------------
    beta = rng.normal(1.0, 0.3, n_stocks).clip(0.2, 2.2)
    size_exp = rng.standard_normal(n_stocks)
    value_exp = rng.standard_normal(n_stocks)
    quality = rng.standard_normal(n_stocks)         # persistent quality, rewarded slightly (planted)
    idio_base = rng.uniform(0.012, 0.025, n_stocks)  # daily specific volatility

    # --- Stock-specific returns with GARCH-like clustering and planted effects ------------------
    eps = np.zeros((n_days, n_stocks))
    h = idio_base ** 2
    for t in range(n_days):
        z = rng.standard_t(6, n_stocks) / np.sqrt(6 / 4)
        eps[t] = np.sqrt(h) * z
        h = 0.05 * idio_base ** 2 + 0.08 * eps[t] ** 2 + 0.87 * h
    alpha_quality = 0.0002 * quality    # about 5% a year per unit of quality (monthly IC of roughly 0.03)
    ret = np.zeros((n_days, n_stocks))
    for t in range(n_days):
        # Planted momentum: residual returns of the past 12-2 months predict next returns weakly.
        mom = 0.0
        if t > 252:
            mom = eps[t - 252:t - 21].sum(axis=0)
            # 0.4% a month per standard deviation of past residual return (monthly IC of roughly 0.04).
            mom = 0.004 * (mom - mom.mean()) / (mom.std() + 1e-12) / 21
        sec = np.array([f_sector[s][t] for s in sectors])
        ret[t] = (beta * f_mkt[t] + sec + size_exp * f_size[t] + value_exp * f_value[t]
                  + alpha_quality + mom + eps[t])

    # --- Delistings: some companies stop trading and suffer a delisting return -------------------
    delist_day = np.full(n_stocks, n_days, dtype=int)
    n_delist = max(1, n_stocks // 20)
    who = rng.choice(n_stocks, n_delist, replace=False)
    delist_day[who] = rng.integers(n_days // 3, n_days - 5, n_delist)

    # --- Prices, volumes, dividends and splits -------------------------------------------------
    rows = []
    shares0 = rng.uniform(5e7, 2e9, n_stocks)
    p0 = rng.uniform(10, 200, n_stocks)
    for j, eid in enumerate(ids):
        T_j = delist_day[j]
        r = ret[:T_j, j]
        # Quarterly dividend yields delta; the observed price falls by the dividend on the ex-date,
        # so P_t = P_{t-1}(1 + r_t)(1 - delta_t) and D_t = P_t * delta_t / (1 - delta_t). This keeps the
        # total-return identity (P_t + D_t) / P_{t-1} - 1 = r_t of Spec §0.3 exact.
        delta = np.zeros(T_j)
        delta[np.arange(60, T_j, 63)] = rng.uniform(0.0, 0.006)
        price = p0[j] * np.cumprod((1 + r) * (1 - delta))
        div = price * delta / (1 - delta)
        split = np.ones(T_j)
        if rng.random() < 0.1 and T_j > 400:
            d = rng.integers(200, T_j - 100)
            split[d] = 2.0
            price[d:] = price[d:] / 2.0
            div[d:] = div[d:] / 2.0
        shares = shares0[j] * np.cumprod(split)
        volume = shares * rng.lognormal(np.log(0.004), 0.4, T_j)
        # Observed price drops by the dividend on the ex-date; the total return adds it back.
        rows.append(pd.DataFrame({
            "entity_id": eid, "date": dates[:T_j], "price": price, "dividend": div,
            "split_ratio": split, "shares_outstanding": shares, "volume": volume, "ret": r}))
    prices = pd.concat(rows, ignore_index=True)

    delistings = pd.DataFrame({
        "entity_id": [ids[j] for j in who],
        "date": [dates[min(delist_day[j], n_days - 1)] for j in who],
        "delisting_return": rng.uniform(-0.9, -0.3, n_delist)})

    companies = pd.DataFrame({
        "entity_id": ids, "ticker": [f"T{i:04d}" for i in range(n_stocks)], "sector": sectors,
        "listed_from": dates[0], "delisted_on": [dates[min(d, n_days - 1)] if d < n_days else pd.NaT for d in delist_day],
        "true_beta": beta, "true_quality": quality})

    mcap = prices.assign(mcap=prices["price"] * prices["shares_outstanding"]).pivot(index="date", columns="entity_id", values="mcap")
    fundamentals = _make_fundamentals(rng, ids, quality, dates, delist_day, mcap)
    macro = _make_macro(rng, months, regime)
    factor_returns = pd.DataFrame({"market": f_mkt, "size": f_size, "value": f_value,
                                   **{f"sector_{s}": f_sector[s] for s in SECTORS}}, index=dates)
    return SyntheticMarket(prices, companies, fundamentals, delistings, macro, factor_returns)


def _make_fundamentals(rng, ids, quality, dates, delist_day, mcap: pd.DataFrame) -> pd.DataFrame:
    """Quarterly fundamentals in point-in-time form, released 30–60 days after each quarter end.

    Annual revenue starts at 0.6–1.4 times market capitalisation (a price-to-sales ratio of about
    0.7–1.7). Each quarter it grows with the AR(1) growth rate and is also pulled 15% of the way toward
    that price-to-sales anchor, so that accounts and prices stay on the same scale (DCF values per
    share are then comparable with prices) while valuations still vary across companies.
    """
    q_ends = pd.date_range(dates[0], dates[-1], freq=QUARTER_END)
    recs = []
    for j, eid in enumerate(ids):
        ps_inv = rng.uniform(0.6, 1.4)                   # sales-to-price ratio of the company
        rev = mcap[eid].iloc[0] * ps_inv
        g = rng.normal(0.08, 0.08)
        margin = np.clip(0.16 + 0.04 * quality[j] + rng.normal(0, 0.03), 0.01, 0.45)
        assets = rev * rng.uniform(0.8, 2.0)
        debt = assets * rng.uniform(0.0, 0.3)
        for q in q_ends:
            if q >= dates[min(delist_day[j], len(dates) - 1)]:
                break
            g = 0.06 + 0.7 * (g - 0.06) + rng.normal(0, 0.03)          # AR(1) growth (Spec §1.4)
            anchor = mcap[eid].asof(q) * ps_inv
            rev = rev * (1 + g / 4) * (anchor / rev) ** 0.15
            rev_q = rev / 4
            margin = np.clip(0.16 + 0.8 * (margin - 0.16) + rng.normal(0, 0.01) + 0.01 * quality[j], 0.0, 0.5)
            ebit = rev_q * margin
            ni = ebit * 0.75
            ocf = ni * rng.uniform(0.8, 1.3)
            assets *= 1 + g / 4
            equity = assets - debt
            filed = q + pd.Timedelta(days=int(rng.integers(30, 60)))
            for field, val in [("revenue", rev_q), ("ebit", ebit), ("net_income", ni), ("operating_cash_flow", ocf),
                               ("total_assets", assets), ("total_debt", debt), ("equity", equity),
                               ("cash", assets * 0.08), ("gross_profit", rev_q * (margin + 0.25))]:
                recs.append((eid, field, q, filed, float(val)))
    return pd.DataFrame(recs, columns=["entity_id", "field", "event_date", "knowledge_date", "value"])


def _make_macro(rng, months, regime) -> pd.DataFrame:
    """Monthly macro indicators whose means depend on the hidden regime (for Spec §2.3–§2.5 tests)."""
    n = len(months)
    growth = np.where(regime == 0, 2.5, -1.0) + rng.normal(0, 0.6, n)
    infl = 2.0 + 0.5 * np.cumsum(rng.normal(0, 0.1, n)) + np.where(regime == 1, -0.4, 0.0)
    unemp = 4.5 + np.cumsum(np.where(regime == 1, 0.15, -0.03) + rng.normal(0, 0.05, n))
    rate = np.clip(2.0 + 0.6 * (infl - 2.0) + rng.normal(0, 0.1, n), 0, None)
    spread = 1.2 + np.where(regime == 1, 1.5, 0.0) + rng.normal(0, 0.2, n)
    consensus_infl = infl + rng.normal(0, 0.15, n)
    return pd.DataFrame({"growth": growth, "inflation": infl, "unemployment": unemp, "policy_rate": rate,
                         "credit_spread": spread, "curve_slope": 1.5 - 0.5 * regime + rng.normal(0, 0.2, n),
                         "inflation_consensus": consensus_infl, "true_regime": regime},
                        index=months.to_timestamp(how="end").normalize())
