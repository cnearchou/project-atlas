"""Section 8: risk management."""

import numpy as np
import pandas as pd
import pytest

from atlas.risk.attribution import euler_decomposition
from atlas.risk.garch import fit_gjr_garch, gjr_recursion
from atlas.risk.liquidity import days_to_liquidate
from atlas.risk.sizing import OptimiserInputs, capacity, drawdown_scale, kelly_cap, optimise, transaction_cost
from atlas.risk.stress import replay, reverse_stress
from atlas.risk.validation import bias_statistic, christoffersen, kupiec
from atlas.risk.var import evt_var_es, normal_var_es


def test_garch_worked_example_and_estimation(rng):
    assert gjr_recursion(np.array([-0.04]), 2e-6, 0.08, 0.0, 0.90, 0.0004)[-1] == pytest.approx(0.00049)
    om, a, g, b = 2e-6, 0.05, 0.08, 0.88
    s2, r = om / (1 - a - g / 2 - b), []
    for _ in range(4000):
        e = np.sqrt(s2) * rng.standard_t(6) / np.sqrt(1.5)
        r.append(e)
        s2 = om + (a + g * (e < 0)) * e * e + b * s2
    res = fit_gjr_garch(np.array(r), use_arch=False)
    assert abs(res.persistence - (a + g / 2 + b)) < 0.05 and res.gamma > 0


def test_var_examples(rng):
    v, e = normal_var_es(0.0221)
    assert v == pytest.approx(0.0364, abs=1e-4) and e == pytest.approx(0.0456, abs=1e-4)
    L = rng.standard_t(4, 20000) * 0.01
    ev = evt_var_es(L, 0.99)
    assert ev["var"] == pytest.approx(np.quantile(L, 0.99), rel=0.1) and ev["es"] > ev["var"]


def test_kupiec_and_christoffersen():
    b = np.zeros(500, bool)
    b[:45] = True
    assert kupiec(b)["reject"]
    b2 = np.zeros(500, bool)
    b2[::20] = True
    assert not kupiec(b2)["reject"] and not christoffersen(b2)["reject"]
    assert bias_statistic(np.ones(100) * np.r_[1, -1] .repeat(50), np.ones(100))["bias"] == pytest.approx(1.005, abs=0.01)


def test_euler_decomposition_sums_to_vol():
    S = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "b"], columns=["a", "b"])
    w = pd.Series([0.6, 0.4], index=["a", "b"])
    d = euler_decomposition(w, S)
    assert d["cr"].sum() == pytest.approx(np.sqrt(w @ S @ w))


def test_sizing_examples():
    assert kelly_cap(0.04, 0.32, 1.0) == pytest.approx(0.390625)
    assert days_to_liquidate(20e6, 110e6) == pytest.approx(0.909, abs=1e-3)
    assert transaction_cost(0.01, 0.002, 0.02, 1e7, 1e8, w_new=-0.01, borrow_fee=0.03)[()] > 0
    eq = pd.Series([1.0, 0.95, 0.91, 0.93, 0.97, 0.99])
    assert drawdown_scale(eq).tolist() == [1, 1, 0.5, 0.5, 1, 1]
    cap = capacity(0.04, lambda a: 1e-6 * np.sqrt(a))
    assert 1e-6 * np.sqrt(cap) == pytest.approx(0.02, rel=0.02)


def test_optimiser_respects_constraints(rng):
    n = 40
    idx = [f"S{i}" for i in range(n)]
    A = rng.normal(size=(n, n)) * 0.01
    S = pd.DataFrame(A @ A.T + np.eye(n) * 1e-3, idx, idx)
    sec = pd.DataFrame({"s1": [1.0] * 20 + [0.0] * 20, "s2": [0.0] * 20 + [1.0] * 20}, index=idx)
    fee = pd.Series(0.0, index=idx)
    fee.iloc[0] = 0.10                         # hard to borrow -> cannot be shorted
    mu = pd.Series(rng.normal(0, 0.01, n), idx)
    mu.iloc[0] = -0.05
    x = OptimiserInputs(mu=mu, Sigma=S, sector_exposure=sec, beta=pd.Series(1.0, idx), borrow_fee=fee)
    w = optimise(x, w_max=0.05, gross_max=1.0, net_max=0.02, exposure_eps=0.01, use_cvxpy=False)
    assert w.abs().max() <= 0.05 + 1e-6 and w.abs().sum() <= 1.0 + 1e-6
    assert abs(w.sum()) <= 0.02 + 1e-6 and (sec.T @ w).abs().max() <= 0.01 + 1e-6
    assert w.iloc[0] >= -1e-8


def test_stress_examples():
    B = pd.DataFrame({"market": [1.15], "growth": [0.8]})
    assert replay(B, {"market": -0.20, "growth": -0.15}).iloc[0] == pytest.approx(-0.35)
    F = np.array([[0.04, 0.0], [0.0, 0.01]])
    shock = reverse_stress(B, pd.Series([1.0]), F, 0.20)
    assert (B.values @ shock.values)[0] == pytest.approx(-0.20)
