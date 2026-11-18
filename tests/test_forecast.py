"""Section 7: forecasting engine."""

import numpy as np
import pandas as pd
import pytest

from atlas.forecast.calibration import ScenarioCalibrator, blend_probabilities, ece, inverse_brier_weights, outcomes_from_returns
from atlas.forecast.cv import leakage_check, purged_kfold, walk_forward_splits
from atlas.forecast.expected import expected_factor_returns_path, grinold_alpha
from atlas.forecast.models import GBMModel, RidgeModel, daily_ic
from atlas.forecast.scenarios import fit_nu, t_probabilities
from atlas.forecast.simulation import simulate_joint


def test_walk_forward_and_kfold_are_purged():
    dates = pd.bdate_range("2015-01-01", periods=800)
    for h in (1, 21):
        for sp in walk_forward_splits(dates, h=h):
            assert leakage_check(sp.train_dates, sp.test_dates, dates, h)
        for sp in purged_kfold(dates, h=h):
            assert leakage_check(sp.train_dates[sp.train_dates < sp.test_dates.min()], sp.test_dates, dates, h)
            after = sp.train_dates[sp.train_dates > sp.test_dates.max()]
            if len(after):
                gap = np.searchsorted(dates, after.min()) - np.searchsorted(dates, sp.test_dates.max())
                assert gap > h


def test_t_probabilities_worked_example():
    p = t_probabilities([0.02], [0.32], 5, 0.15).iloc[0]
    assert p["p_bull"] == pytest.approx(0.311, abs=1e-3) and p["p_bear"] == pytest.approx(0.262, abs=1e-3)


def test_fit_nu_recovers_tails(rng):
    nu = 5.0
    e = rng.standard_t(nu, 20000) / np.sqrt(nu / (nu - 2))
    assert abs(fit_nu(e) - nu) < 1.0


def test_grinold_and_blending_examples():
    a = grinold_alpha(pd.Series([-1.2]), 0.04, pd.Series([0.25]))
    assert a.iloc[0] == pytest.approx(-0.012)
    w = inverse_brier_weights({"dcf": 0.20, "stat": 0.16})
    assert w["dcf"] == pytest.approx(0.444, abs=1e-3)
    m = {"dcf": pd.DataFrame({"p_bull": [0.198], "p_base": [0.481], "p_bear": [0.321]}),
         "stat": pd.DataFrame({"p_bull": [0.311], "p_base": [0.427], "p_bear": [0.262]})}
    b = blend_probabilities(m, w).iloc[0]
    assert b["p_bull"] == pytest.approx(0.261, abs=2e-3) and b["p_bear"] == pytest.approx(0.288, abs=2e-3)


def test_calibration_fixes_overconfidence(rng):
    n = 6000
    p_bull = rng.uniform(0.05, 0.5, n)
    p_bear = np.full(n, 0.2)
    u = rng.random(n)
    o = pd.DataFrame({"p_bull": (u < p_bull).astype(float), "p_bear": (u > 1 - p_bear).astype(float)})
    o["p_base"] = 1 - o["p_bull"] - o["p_bear"]
    raw_bull = np.clip(1.5 * p_bull, 0, 0.75)              # overconfident bull forecasts
    raw = pd.DataFrame({"p_bull": raw_bull, "p_bear": p_bear, "p_base": 1 - raw_bull - p_bear})
    cal = ScenarioCalibrator().fit(raw, o).transform(raw)
    assert np.allclose(cal.sum(axis=1), 1)
    assert ece(cal["p_bull_cal"], o["p_bull"]) < ece(raw["p_bull"], o["p_bull"])
    assert outcomes_from_returns([0.2, 0.0, -0.2])[["p_bull", "p_base", "p_bear"]].values.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_models_learn_planted_signal(rng):
    rows = []
    for d in pd.bdate_range("2018-01-01", periods=80):
        x = rng.standard_normal(100)
        rows.append(pd.DataFrame({"date": d, "x": x, "n": rng.standard_normal(100), "target": 0.3 * x + rng.standard_normal(100)}))
    df = pd.concat(rows, ignore_index=True)
    rid = RidgeModel(h=1).fit(df, ["x", "n"])
    assert rid.model.coef_[0] > 5 * abs(rid.model.coef_[1])
    gbm = GBMModel(max_iter=100).fit(df, ["x", "n"])
    df["pred"] = gbm.predict(df)
    assert daily_ic(df).mean() > 0.2
    c = rid.contributions(df.iloc[:5])
    assert np.allclose(c.sum(axis=1) + rid.model.intercept_ + (rid.mean_ * rid.model.coef_).sum(), rid.predict(df.iloc[:5]))


def test_regime_path_and_joint_simulation():
    means = pd.DataFrame({"market": [0.01, -0.02]}, index=["exp", "rec"])
    T = np.array([[0.95, 0.05], [0.2, 0.8]])
    ef = expected_factor_returns_path(means, np.array([0.0, 1.0]), T, 12)
    assert -0.24 < ef["market"] < 0.12            # drifts away from the recession mean
    B = pd.DataFrame({"market": np.ones(50)}, index=[f"S{i}" for i in range(50)])
    F = [np.array([[0.0004]]), np.array([[0.0016]])]
    R = simulate_joint(B, np.array([1.0, 0.0]), T, [np.zeros(1), np.zeros(1)], F, pd.Series(0.0004, index=B.index), n_paths=4000)
    C = np.corrcoef(R[:, :5].T)
    assert R.shape == (4000, 50) and C[0, 1] > 0.3        # common factor makes stocks co-move
