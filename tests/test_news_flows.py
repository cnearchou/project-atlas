"""Sections 3 and 4: news and flows."""

import numpy as np
import pandas as pd
import pytest

from atlas.flows.positioning import crowding_scores, insider_scores, short_metrics, sqrt_impact
from atlas.news.events import bmp_test, detect_events, event_study, shrink_impacts
from atlas.news.indexes import fit_lambda, smooth
from atlas.news.pipeline import SentimentModel, assign_session, deduplicate, link_entities, novelty_scores
from atlas.news.politics import normalise_prices, resolution_moves


def _articles():
    return pd.DataFrame({
        "title": ["Chipmaker beats earnings with record growth", "Chipmaker beats earnings with record growth",
                  "New tariff war hits exporters"],
        "body": ["Quarterly revenue rose strongly and guidance was raised again"] * 2 + ["Duties on imports announced"],
        "published_utc": pd.to_datetime(["2024-01-02 15:00", "2024-01-02 15:30", "2024-01-05 22:00"], utc=True)})


def test_deduplicate_keeps_earliest():
    out = deduplicate(_articles())
    assert len(out) == 2 and out["published_utc"].iloc[0].hour == 15 and out["published_utc"].iloc[0].minute == 0


def test_session_alignment_after_close_and_weekend():
    s = assign_session(_articles()["published_utc"])
    assert s.iloc[0] == pd.Timestamp("2024-01-02")     # 10:00 New York, same day
    assert s.iloc[2] == pd.Timestamp("2024-01-08")     # Friday 17:00 New York -> Monday


def test_sentiment_novelty_and_entities():
    s = SentimentModel().score(["profit beats and record growth", "lawsuit and losses after downgrade"])
    assert s[0] > 0 > s[1]
    nov = novelty_scores(_articles())
    assert nov.iloc[1] < 0.05 and nov.iloc[0] == 1.0
    assert link_entities("Apple Inc reported iPhone sales", {"apple": "E1"}, {"E1": "Apple designs iPhone smartphones"}) == ["E1"]


def test_index_smoothing_and_lambda_fit(rng):
    idx = pd.bdate_range("2020-01-01", periods=400)
    x = pd.Series(rng.standard_normal(400), index=idx)
    I = smooth(x, 0.9)
    assert I.std() < x.std()
    nxt = (0.5 * smooth(x, 0.9)).shift(-1) + 0.01 * rng.standard_normal(400)
    best, table = fit_lambda(x, nxt.shift(1).shift(-1), train_end=idx[300])
    assert best in table["lambda"].values


def test_event_detection_and_event_study(rng):
    idx = pd.bdate_range("2020-01-01", periods=300)
    counts = pd.Series(rng.poisson(5, 300), index=idx)
    counts.iloc[-1] = 30
    ev = detect_events(pd.Series(0.0, index=idx), counts)
    assert ev["event_volume"].iloc[-1]
    m = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    r = 1.2 * m + pd.Series(rng.normal(0, 0.01, 300), index=idx)
    r.iloc[280] += 0.05
    res = event_study(r, m, idx[280])
    assert res["car"] > 0.03 and abs(res["beta"] - 1.2) < 0.3
    assert bmp_test(np.full(20, 1.0) + rng.normal(0, 0.5, 20)) > 3


def test_shrinkage_pulls_noisy_cells_to_type_mean():
    cells = pd.DataFrame({"event_type": ["war"] * 3, "sector": ["a", "b", "c"], "mean_car": [0.05, -0.01, 0.0],
                          "sd_car": [0.10, 0.01, 0.01], "n": [2, 100, 100]})
    out = shrink_impacts(cells)
    assert abs(out["shrunk_impact"].iloc[0] - out["mean_car"].mean()) < abs(0.05 - out["mean_car"].mean())


def test_politics_examples():
    assert normalise_prices(pd.Series([0.56, 0.48])).sum() == pytest.approx(1.0)
    mv = resolution_moves(0.12, 55)
    assert mv["win"] == pytest.approx(5.4) and mv["lose"] == pytest.approx(-6.6)


def test_flows_examples():
    assert sqrt_impact(6e6, 2e6, 0.025, 0.8) == pytest.approx(0.0346, abs=1e-4)
    df = pd.DataFrame({"entity_id": list("ABC"), "date": pd.Timestamp("2024-01-01"), "short_interest": [1e6, 5e6, 2e6],
                       "adv_20d": [1e6] * 3, "float_shares": [1e8] * 3, "momentum": [0.1, 0.5, 0.2]})
    sm = short_metrics(df)
    assert sm.loc[sm["entity_id"] == "B", "squeeze_score"].iloc[0] == sm["squeeze_score"].max()
    h = pd.DataFrame({"fund_id": ["f1", "f1", "f2", "f2", "f3"], "entity_id": ["A", "B", "A", "B", "C"],
                      "weight": [0.5, 0.5, 0.5, 0.5, 1.0]})
    cr = crowding_scores(h)
    assert cr["A"] == pytest.approx(1.0) and np.isnan(cr["C"])
    t = pd.DataFrame({"entity_id": ["A"] * 3, "insider_id": [1, 2, 3], "date": pd.to_datetime(["2024-01-10", "2024-01-12", "2024-01-15"]),
                      "value": [1e5, 2e5, 1e5], "plan_10b5_1": [False] * 3})
    ins = insider_scores(t, pd.Series({"A": 1e8}), "2024-01-31")
    assert ins.loc["A", "insider_cluster"] and ins.loc["A", "insider_score"] == pytest.approx(4e-3)
