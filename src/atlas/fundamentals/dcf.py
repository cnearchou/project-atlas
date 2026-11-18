"""Valuation formulas (Spec §1.5; Ref 3.3).

DCF value:
    V = sum_{t=1..5} FCF_t / (1 + WACC)^t + TV / (1 + WACC)^5

Growth linked to reinvestment (first consistency rule):
    Reinvestment_t = g_{t+1} / ROIC_new x NOPAT_t,    FCF_t = NOPAT_t - Reinvestment_t

Value-driver terminal value (second consistency rule):
    TV = NOPAT_6 (1 - g_inf / ROIC_inf) / (WACC - g_inf)

Residual income model for banks, insurers and REITs (third consistency rule):
    V_0 = B_0 + sum_{t=1..5} (ROE_t - r_e) B_{t-1} / (1 + r_e)^t + TV_RI / (1 + r_e)^5

Cost of capital:
    r_e = r_f + beta x ERP,   WACC = E/(D+E) r_e + D/(D+E) r_d (1 - tax)

All functions accept NumPy arrays, so the Monte Carlo can value thousands of paths at once.
"""

from __future__ import annotations

import numpy as np

# Sectors valued with the residual income model instead of a DCF (Spec §1.5, third rule).
FINANCIAL_SECTORS = {"Financials", "Banks", "Insurance", "Real Estate", "REITs"}


def cost_of_equity(rf: float, beta: float, erp: float) -> float:
    """CAPM cost of equity r_e = r_f + beta x ERP."""
    return rf + beta * erp


def wacc(rf: float, beta: float, erp: float, rd: float, tax: float, weight_equity: float) -> float:
    """Weighted average cost of capital with target capital-structure weights."""
    re = cost_of_equity(rf, beta, erp)
    return weight_equity * re + (1 - weight_equity) * rd * (1 - tax)


def value_driver_terminal(nopat_next, g_inf, roic_inf, wacc_):
    """TV = NOPAT_{T+1} (1 - g_inf / ROIC_inf) / (WACC - g_inf).

    Growth beyond the forecast period only adds value when ROIC_inf exceeds WACC; when ROIC_inf equals
    WACC, the terminal value equals NOPAT / WACC regardless of growth.
    """
    spread = np.maximum(np.asarray(wacc_) - np.asarray(g_inf), 1e-3)  # keeps the formula finite
    return nopat_next * (1 - np.asarray(g_inf) / np.asarray(roic_inf)) / spread


def discounted_value(fcf, tv, wacc_):
    """Enterprise value from an (n_paths x T) FCF array, terminal values and per-path WACC."""
    fcf = np.atleast_2d(fcf)
    T = fcf.shape[1]
    w = np.asarray(wacc_).reshape(-1, 1)
    disc = (1 + w) ** np.arange(1, T + 1)
    return (fcf / disc).sum(axis=1) + np.asarray(tv) / (1 + w[:, 0]) ** T


def residual_income_value(book0, roe, re, g_inf=0.0):
    """Residual income value per path.

    `roe` is an (n_paths x T) array of return on equity, book value grows with retained earnings
    (a full-retention simplification), and the terminal value capitalises the final residual income
    with growth g_inf.
    """
    roe = np.atleast_2d(roe)
    n, T = roe.shape
    book = np.empty((n, T + 1))
    book[:, 0] = book0
    ri = np.empty((n, T))
    for t in range(T):
        ri[:, t] = (roe[:, t] - re) * book[:, t]
        book[:, t + 1] = book[:, t] * (1 + roe[:, t])
    disc = (1 + re) ** np.arange(1, T + 1)
    tv_ri = ri[:, -1] * (1 + g_inf) / max(re - g_inf, 1e-3)
    return book0 + (ri / disc).sum(axis=1) + tv_ri / (1 + re) ** T
