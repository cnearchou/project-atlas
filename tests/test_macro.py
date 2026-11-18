"""Section 2: macro."""

import numpy as np
import pandas as pd

from atlas.macro.regimes import GaussianHMM, next_period
from atlas.macro.transforms import adf_test
from atlas.macro.var_model import fit_var


def test_hmm_recovers_regimes(small_market):
    Y = small_market.macro[["growth", "credit_spread"]]
    Yz = ((Y - Y.mean()) / Y.std()).values
    m = GaussianHMM(2).fit(Yz, seed=0)
    probs = m.filter(Yz)
    rec = int(np.argmin(m.means[:, 0]))
    acc = ((probs[:, rec] > 0.5).astype(int) == small_market.macro["true_regime"].values).mean()
    assert acc > 0.8
    assert np.allclose(m.T.sum(axis=1), 1)


def test_next_period_worked_example():
    T = np.array([[0.9, 0.1], [0.3, 0.7]])
    assert np.allclose(next_period(np.array([0.5, 0.5]), T), [0.6, 0.4])


def test_var_stability_and_adf(rng):
    n = 400
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
    res = fit_var(pd.DataFrame(y, columns=["a", "b"]), 1)
    assert res.is_stable()
    assert abs(res.A[0][0, 0] - 0.5) < 0.1
    rw = pd.Series(np.cumsum(rng.standard_normal(n)))
    assert adf_test(pd.Series(y[:, 0]))["stationary_5pct"]
    assert not adf_test(rw)["stationary_5pct"]
