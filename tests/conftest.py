"""Shared fixtures. Small synthetic markets keep the whole suite fast (well under a few minutes)."""

import numpy as np
import pandas as pd
import pytest

from atlas.ingest.synthetic import make_market


@pytest.fixture(scope="session")
def small_market():
    return make_market(n_stocks=40, n_days=400, seed=3)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
