"""Section 9 and Section A: hypothesis testing."""

import numpy as np
import pandas as pd
import pytest

from atlas.research.backtest import quantile_portfolios
from atlas.research.gate import production_gate
from atlas.research.hypotheses import register_all
from atlas.research.inference import (benjamini_hochberg, deflated_sharpe_ratio, effective_sample_size, fama_macbeth,
                                      nw_tstat_mean, ols)
from atlas.research.metrics import fundamental_law, ic_series, max_drawdown
from atlas.research.monitoring import cusum, psi
from atlas.research.registry import Registry


def test_benjamini_hochberg_example():
    assert benjamini_hochberg([0.001, 0.008, 0.020, 0.040, 0.300]).tolist() == [True, True, True, True, False]


def test_newey_west_and_ols(rng):
    x = rng.standard_normal(500)
    y = 2 + 0.5 * x + rng.standard_normal(500)
    r = ols(y, x)
    assert abs(r.params[1] - 0.5) < 0.1 and r.tstats[1] > 5
    m, se, t = nw_tstat_mean(0.03 + 0.12 * rng.standard_normal(240))
    assert np.isfinite(t)
    e = np.convolve(rng.standard_normal(1200), np.ones(12), "valid")   # overlapping 12-period sums
    assert effective_sample_size(e, 12) < len(e) / 4


def test_deflated_sharpe_penalises_many_trials():
    assert deflated_sharpe_ratio(0.3, 0.01, 2, 240) > deflated_sharpe_ratio(0.3, 0.01, 1000, 240)


def test_fama_macbeth_recovers_coefficient(rng):
    rows = []
    for d in range(60):
        x = rng.standard_normal(200)
        rows.append(pd.DataFrame({"date": d, "x": x, "y": 0.02 * x + 0.05 * rng.standard_normal(200)}))
    res = fama_macbeth(pd.concat(rows), "y", ["x"])
    assert res.loc["x", "mean"] == pytest.approx(0.02, abs=0.005) and res.loc["x", "t"] > 3


def test_planted_signal_detected_and_noise_rejected(rng):
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    ids = [f"S{i}" for i in range(200)]
    sig = pd.DataFrame(rng.standard_normal((120, 200)), dates, ids)
    noise = pd.DataFrame(rng.standard_normal((120, 200)), dates, ids)
    fwd = 0.01 * sig + 0.08 * pd.DataFrame(rng.standard_normal((120, 200)), dates, ids)
    ic_sig, ic_noise = ic_series(sig, fwd), ic_series(noise, fwd)
    assert nw_tstat_mean(ic_sig.values)[2] > 3 and abs(nw_tstat_mean(ic_noise.values)[2]) < 3
    bt = quantile_portfolios(sig, fwd)
    assert bt["monotonic"] and bt["long_short"].mean() > 0
    assert fundamental_law(0.03, 1000) == pytest.approx(0.949, abs=1e-3)
    assert 0 <= max_drawdown(bt["long_short"]) < 1


def test_gate_and_monitoring(rng):
    g = production_gate(t_nw=3.5, bh_survives=True, fm_t=3.2, split_ics=[0.02, 0.03], variant_ts=[3.1, 3.4],
                        net_mean=0.002, dsr=0.97)
    assert g.passed
    assert not production_gate(t_nw=3.5).passed          # unrun checks block the gate
    x = rng.standard_normal(5000)
    assert psi(x, rng.standard_normal(5000)) < 0.05 and psi(x, rng.standard_normal(5000) + 1) > 0.25
    ic = pd.Series(np.r_[np.full(24, 0.04), np.full(24, -0.02)])
    cs = cusum(ic, mu0=0.04, k=0.005, h=0.1)
    assert not cs["alarm"].iloc[:24].any() and cs["alarm"].iloc[-1]


def test_registry_preregisters_hypotheses(tmp_path):
    reg = Registry(tmp_path / "reg.csv")
    ids = register_all(reg)
    assert set(ids) == {"H1", "H2", "H3"}
    assert register_all(reg) == {}                         # registering twice does nothing
    reg.log(hypothesis_text="x", signal_definition="momentum_12_1", passed_gate=False)
    assert reg.n_trials("momentum") == 1 and len(reg.load()) == 4
