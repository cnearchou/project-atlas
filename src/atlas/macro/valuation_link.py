"""Linking macro estimates to valuation (Spec §2.7; Ref 3.4).

* The VAR's forecast path for the 10-year yield is used as the risk-free rate in the DCF.
* The equity risk premium is implied from the market: the S&P 500's own DCF is solved for the ERP
  that makes the model value equal the index level:

      Index = sum_{t=1..T} CF_t / (1 + r_f + ERP)^t + TV / (1 + r_f + ERP)^T
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def implied_erp(index_level: float, cash_flows: np.ndarray, rf: float, g_terminal: float) -> float:
    """Solve for the ERP that equates the discounted index cash flows to the index level.

    `cash_flows` are expected index-level cash flows to shareholders (dividends plus buybacks) for
    the next T years; the terminal value grows at g_terminal.
    """
    cf = np.asarray(cash_flows, float)
    T = len(cf)

    def value(erp):
        r = rf + erp
        disc = (1 + r) ** np.arange(1, T + 1)
        tv = cf[-1] * (1 + g_terminal) / (r - g_terminal)
        return (cf / disc).sum() + tv / (1 + r) ** T - index_level

    return float(brentq(value, g_terminal - rf + 1e-4, 0.30))


def expected_market_return(rf: float, erp: float) -> float:
    """E[r_m] = r_f + ERP, used for the market factor's expected return in Spec §7.4."""
    return rf + erp
