"""Accounting scores and composites (Spec §1.3; Ref 3.3).

Piotroski F-score (0–9): one point for each test passed across profitability, leverage/liquidity
and operating efficiency.

Altman Z-score:  Z = 1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 MVE/TL + 1.0 Sales/TA
    Z < 1.8 is the distress zone and Z > 3.0 the safe zone.

Beneish M-score: M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI
                     - 0.172 SGAI + 4.679 TATA - 0.327 LVGI;  M > -1.78 flags likely manipulation.

Composite scores (each component is rank-Gaussianised first, Spec §0.7):
    Quality = mean[z(ROIC), z(GP), -z(Accruals), -z(Leverage), -z(sigma_ROE,5y)]
    Value   = mean[z(E/P), z(B/P), z(FCF/EV), z(EBITDA/EV)]
    Growth  = mean[z(g_revenue), z(g_EPS), z(Δ margin)]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.data.standardise import rank_gauss


def piotroski_f(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    """F-score from this year's and last year's annual values (same index for both frames).

    Required columns: net_income, total_assets, operating_cash_flow, long_term_debt, current_assets,
    current_liabilities, shares, gross_profit, revenue.
    """
    roa = cur["net_income"] / cur["total_assets"]
    roa_prev = prev["net_income"] / prev["total_assets"]
    cr = cur["current_assets"] / cur["current_liabilities"]
    cr_prev = prev["current_assets"] / prev["current_liabilities"]
    tests = [
        roa > 0,                                                        # profitability
        cur["operating_cash_flow"] > 0,
        roa > roa_prev,
        cur["operating_cash_flow"] > cur["net_income"],
        cur["long_term_debt"] / cur["total_assets"] < prev["long_term_debt"] / prev["total_assets"],  # leverage
        cr > cr_prev,
        cur["shares"] <= prev["shares"],                                # no new shares issued
        cur["gross_profit"] / cur["revenue"] > prev["gross_profit"] / prev["revenue"],              # efficiency
        cur["revenue"] / cur["total_assets"] > prev["revenue"] / prev["total_assets"],
    ]
    return sum(t.astype(int) for t in tests)


def altman_z(wc, re, ebit, mve, tl, sales, ta) -> pd.Series:
    """Altman (1968) Z-score. Inputs are aligned Series."""
    return 1.2 * wc / ta + 1.4 * re / ta + 3.3 * ebit / ta + 0.6 * mve / tl + 1.0 * sales / ta


def beneish_m(dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi) -> pd.Series:
    """Beneish (1999) eight-variable M-score. Each index compares this year with last year."""
    return (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi
            - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)


def beneish_indices(cur: pd.DataFrame, prev: pd.DataFrame) -> dict[str, pd.Series]:
    """The eight Beneish indices from two years of annual data."""
    gm = lambda d: (d["revenue"] - d["cogs"]) / d["revenue"]  # noqa: E731
    aq = lambda d: 1 - (d["current_assets"] + d["ppe"]) / d["total_assets"]  # noqa: E731
    return {
        "dsri": (cur["receivables"] / cur["revenue"]) / (prev["receivables"] / prev["revenue"]),
        "gmi": gm(prev) / gm(cur),
        "aqi": aq(cur) / aq(prev),
        "sgi": cur["revenue"] / prev["revenue"],
        "depi": (prev["depreciation"] / (prev["depreciation"] + prev["ppe"]))
                / (cur["depreciation"] / (cur["depreciation"] + cur["ppe"])),
        "sgai": (cur["sga"] / cur["revenue"]) / (prev["sga"] / prev["revenue"]),
        "tata": (cur["net_income"] - cur["operating_cash_flow"]) / cur["total_assets"],
        "lvgi": (cur["total_liabilities"] / cur["total_assets"]) / (prev["total_liabilities"] / prev["total_assets"]),
    }


def composite(df: pd.DataFrame, positive: list[str], negative: list[str] | None = None,
              date_col: str = "date") -> pd.Series:
    """Average of rank-Gaussianised components; `negative` components enter with a minus sign.

    The minus signs flip measures where lower is better, so a higher composite always means more
    attractive. Components that are missing for a company are ignored in its average.
    """
    parts = []
    for c in positive:
        parts.append(df.groupby(date_col)[c].transform(rank_gauss))
    for c in negative or []:
        parts.append(-df.groupby(date_col)[c].transform(rank_gauss))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def quality_value_growth(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """The three composites of Spec §1.3, using whichever component columns are available."""
    have = lambda cols: [c for c in cols if c in df]  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["quality_z"] = composite(df, have(["roic", "gross_profitability"]),
                                 have(["accruals", "leverage", "roe_vol_5y"]), date_col)
    out["value_z"] = composite(df, have(["earnings_yield", "book_to_price", "fcf_yield", "ebitda_ev"]), [], date_col)
    out["growth_z"] = composite(df, have(["g_revenue", "g_eps", "d_margin"]), [], date_col)
    return out.replace([np.inf, -np.inf], np.nan)
