"""Liquidity-adjusted risk (Spec §8.4; Ref 3.10).

    Days to liquidate = position value / (participation x ADV$)
    LVaR = VaR + (1/2) spread x position + Y sigma_daily sqrt(Q / ADV) x position
The last term is the square-root impact of Spec §4.2 for selling Q shares.
Worked example (Spec §8.4 note): $20M position, $110M ADV, 20% participation -> about 0.9 days.
"""

from __future__ import annotations

import numpy as np

from atlas.flows.positioning import sqrt_impact


def days_to_liquidate(position_value, adv_dollars, participation: float = 0.20):
    return np.abs(position_value) / (participation * np.asarray(adv_dollars))


def lvar(var_dollars: float, position_value: float, spread: float, adv_dollars: float, sigma_daily: float,
         Y: float = 0.8) -> float:
    pv = abs(position_value)
    impact = sqrt_impact(pv, adv_dollars, sigma_daily, Y)
    return float(var_dollars + 0.5 * spread * pv + impact * pv)
