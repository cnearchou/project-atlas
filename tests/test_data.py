"""Section 0: shared data infrastructure. Includes the no-look-ahead test required by Build Phase 1."""

import numpy as np
import pandas as pd
import pytest

from atlas.data.cleaning import forward_fill_limited, impute_sector_median, robust_zscore, winsorise
from atlas.data.corporate_actions import total_returns
from atlas.data.point_in_time import PointInTimeTable, asof_join
from atlas.data.standardise import neutralise, rank_gauss
from atlas.data.storage import DataStore
from atlas.data.universe import apply_delisting_returns, eligible_universe


def test_point_in_time_never_returns_future_information(small_market):
    """Phase 1 exit criterion: a query as of date d never returns information known after d."""
    pit = PointInTimeTable(small_market.fundamentals)
    for d in pd.to_datetime(["2018-06-30", "2018-09-15", "2019-01-31"]):
        snap = pit.as_of(d)
        assert (snap["knowledge_date"] <= d).all()


def test_restatement_replaces_original_only_after_it_is_known():
    pit = PointInTimeTable(pd.DataFrame({
        "entity_id": ["A", "A"], "field": ["revenue", "revenue"],
        "event_date": ["2020-03-31", "2020-03-31"], "knowledge_date": ["2020-05-01", "2020-08-01"],
        "value": [100.0, 90.0]}))
    assert pit.latest_value("2020-06-01", "revenue")["A"] == 100.0
    assert pit.latest_value("2020-09-01", "revenue")["A"] == 90.0
    assert pit.latest_value("2020-04-01", "revenue").empty


def test_asof_join_respects_max_age():
    left = pd.DataFrame({"entity_id": ["A"] * 3, "date": pd.to_datetime(["2020-01-10", "2020-03-01", "2021-01-01"])})
    right = pd.DataFrame({"entity_id": ["A"], "knowledge_date": pd.to_datetime(["2020-02-01"]), "x": [1.0]})
    out = asof_join(left, right, max_age=pd.Timedelta(days=92))
    assert np.isnan(out["x"].iloc[0]) and out["x"].iloc[1] == 1.0 and np.isnan(out["x"].iloc[2])


def test_total_return_identity_and_split_has_no_fake_return(small_market):
    px = small_market.prices
    r = total_returns(px)
    assert np.nanmax(np.abs(r - px["ret"])) < 1e-10
    split_rows = px["split_ratio"] != 1
    if split_rows.any():
        assert r[split_rows].abs().max() < 0.3   # a 2-for-1 split must not look like -50%


def test_delisting_return_missing_defaults_to_minus_30pct():
    rets = pd.DataFrame({"entity_id": ["A", "A"], "date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "ret": [0.01, 0.0]})
    dl = pd.DataFrame({"entity_id": ["A"], "date": pd.to_datetime(["2020-01-02"]), "delisting_return": [np.nan]})
    out = apply_delisting_returns(rets, dl)
    assert out["ret"].iloc[1] == pytest.approx(-0.30)


def test_universe_uses_lagged_adv():
    n = 30
    df = pd.DataFrame({"entity_id": "A", "date": pd.bdate_range("2020-01-01", periods=n), "price": 10.0,
                       "shares_outstanding": 1e8, "volume": 1e6})
    df.loc[n - 1, "volume"] = 1e9      # a volume spike today must not make the stock eligible today
    ok = eligible_universe(df, min_adv_dollars=5e7)
    assert not ok.iloc[-1]


def test_cleaning_functions():
    x = pd.Series([1.0, 2, 3, 4, 100])
    z = robust_zscore(x)
    assert z.max() == 3.0 and abs(z.iloc[2]) < 1e-12
    assert winsorise(pd.Series(np.arange(101.0)), 0.05, 0.95).max() == 95.0
    df = pd.DataFrame({"sector": ["a", "a", "b"], "x": [1.0, np.nan, 5.0]})
    out = impute_sector_median(df, "x")
    assert out["x"].iloc[1] == 1.0 and out["x_missing"].iloc[1] == 1
    ff = pd.DataFrame({"entity_id": "A", "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-02-01"]), "v": [1.0, np.nan, np.nan]})
    out = forward_fill_limited(ff, ["v"], max_days=5)
    assert out["v"].iloc[1] == 1.0 and np.isnan(out["v"].iloc[2])


def test_rank_gauss_worked_example_and_neutralise():
    z = rank_gauss(pd.Series([5.0, 1, 3, 2, 4]))
    assert np.allclose(sorted(z.round(2)), [-1.28, -0.52, 0.0, 0.52, 1.28])
    rng = np.random.default_rng(1)
    sectors = pd.Series(rng.choice(["a", "b", "c"], 300))
    s = pd.Series(rng.standard_normal(300)) + sectors.map({"a": 2.0, "b": 0.0, "c": -2.0})
    r = neutralise(s, sectors)
    assert r.groupby(sectors).mean().abs().max() < 1e-10


def test_raw_layer_is_write_once(tmp_path):
    st = DataStore(tmp_path)
    st.write("raw", "t", pd.DataFrame({"a": [1]}))
    with pytest.raises(FileExistsError):
        st.write("raw", "t", pd.DataFrame({"a": [2]}))
    st.write("clean", "t", pd.DataFrame({"a": [1]}))
    st.write("clean", "t", pd.DataFrame({"a": [2]}))
    assert st.read("clean", "t")["a"].iloc[0] == 2
