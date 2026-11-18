"""Section 1: fundamentals."""

import numpy as np
import pytest

from atlas.fundamentals.expectations import deterministic_value, reverse_dcf
from atlas.fundamentals.monte_carlo import DCFInputs, convert_to_12m_returns, run_dcf_monte_carlo, scenario_probabilities
from atlas.fundamentals.options import bs_price, implied_vol, risk_neutral_density


def _inputs():
    return DCFInputs(revenue0=1e9, margin0=0.15, g0=0.08, shares=1e8, net_debt=1e8)


def test_monte_carlo_precision_and_antithetic():
    mc = run_dcf_monte_carlo(_inputs(), n=4000, seed=0)
    assert mc.se == pytest.approx(mc.sd / np.sqrt(4000))
    assert mc.mean > 0 and np.isfinite(mc.values).all()


def test_12m_conversion_noise_widens_distribution():
    mc = run_dcf_monte_carlo(_inputs(), n=4000, seed=0)
    price = mc.mean
    narrow = convert_to_12m_returns(mc.values, price, 0.3, 0.07, 0.0)
    wide = convert_to_12m_returns(mc.values, price, 0.3, 0.07, 0.30)
    assert wide.std() > narrow.std()
    p = scenario_probabilities(wide)
    assert abs(sum(p.values()) - 1) < 1e-12


def test_reverse_dcf_recovers_growth():
    x = _inputs()
    price = deterministic_value(x)
    assert reverse_dcf(x, price, "g0") == pytest.approx(0.08, abs=1e-4)


def test_implied_vol_round_trip_and_density_integrates_to_one():
    S, T, r = 100.0, 0.5, 0.03
    p = bs_price(S, 105.0, T, r, 0.3)
    assert implied_vol(p, S, 105.0, T, r) == pytest.approx(0.3, abs=1e-5)
    strikes = np.linspace(60, 150, 25)
    ivs = 0.25 + 0.1 * (np.log(strikes / S)) ** 2
    rnd = risk_neutral_density(S, T, r, strikes, ivs)
    assert rnd.integral == pytest.approx(1.0, abs=0.02)
