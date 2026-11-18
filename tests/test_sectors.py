"""Section 6: sector structure."""

import numpy as np
import pytest

from atlas.sectors.betas import vasicek_shrink
from atlas.sectors.factor_model import estimate_factor_returns, ewma_covariance
from atlas.sectors.network import propagate


def test_factor_model_recovers_market_factor(small_market):
    px = small_market.prices.merge(small_market.companies[["entity_id", "sector"]], on="entity_id")
    px["market_cap"] = px["price"] * px["shares_outstanding"]
    px["size"] = np.log(px["market_cap"])
    sectors = sorted(px["sector"].unique())
    fm = estimate_factor_returns(px[px["date"] > px["date"].min()], sectors, styles=["size"])
    true = small_market.factor_returns["market"].reindex(fm.factor_returns.index)
    assert fm.factor_returns["market"].corr(true) > 0.9
    F = ewma_covariance(fm.factor_returns)
    assert np.all(np.linalg.eigvalsh(F) > 0)


def test_network_worked_example():
    W = np.zeros((3, 3))
    W[1, 0], W[2, 0], W[2, 1] = 0.4, 0.3, 0.5
    s = propagate(W, np.array([-0.10, 0, 0]), 0.8)
    assert s[1] == pytest.approx(-0.032) and s[2] == pytest.approx(-0.0368)


def test_vasicek_worked_example():
    assert vasicek_shrink(1.6, 0.4, 1.1, 0.04) == pytest.approx(1.2)
