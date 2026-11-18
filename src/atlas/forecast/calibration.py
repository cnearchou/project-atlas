"""Blending and calibration (Spec §7.7; Ref 3.9 and Technical Reference Part 1.6).

Brier score of a model over N resolved forecasts and the three outcomes (bull, base, bear):
    BS = (1/N) sum_n sum_c (p_{n,c} - o_{n,c})^2
(the one-outcome version (1/N) sum (p - o)^2 is used when a single event is scored).

Inverse-Brier blending of the DCF (Spec §1.5) and statistical (Spec §7.6) probabilities:
    w_m = (1 / BS_m) / sum_j (1 / BS_j),   p_blend = sum_m w_m p_m
Each model is scored over a rolling three-year window, using only forecasts whose 12 months have
elapsed, so the weights at date t use forecasts made before t - 12 months.

Isotonic calibration maps raw to calibrated probabilities (fitted on past out-of-sample forecasts).
Expected Calibration Error:
    ECE = sum_b (n_b / N) | p_bar_b - o_bar_b |
Overlapping 12-month forecasts are strongly autocorrelated, so every calibration statistic is
reported with the effective sample size n_eff (atlas.research.inference.effective_sample_size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from atlas.research.inference import effective_sample_size

OUTCOMES = ("p_bull", "p_base", "p_bear")


def outcomes_from_returns(realised_12m, threshold: float = 0.15) -> pd.DataFrame:
    """One-hot outcomes: bull if R > x, bear if R < -x, base otherwise (Section A.3)."""
    r = np.asarray(realised_12m, float)
    return pd.DataFrame({"p_bull": (r > threshold).astype(float), "p_base": ((r >= -threshold) & (r <= threshold)).astype(float),
                         "p_bear": (r < -threshold).astype(float)})


def brier(probs: pd.DataFrame, outcomes: pd.DataFrame) -> float:
    P, O = probs[list(OUTCOMES)].values, outcomes[list(OUTCOMES)].values
    ok = np.isfinite(P).all(axis=1) & np.isfinite(O).all(axis=1)
    return float(np.mean(np.sum((P[ok] - O[ok]) ** 2, axis=1)))


def binary_brier(p, o) -> float:
    p, o = np.asarray(p, float), np.asarray(o, float)
    return float(np.mean((p - o) ** 2))


def inverse_brier_weights(scores: dict[str, float]) -> dict[str, float]:
    """Example (Spec §7.7 note): Brier 0.20 and 0.16 -> weights 0.44 and 0.56."""
    inv = {k: 1.0 / v for k, v in scores.items() if v and np.isfinite(v)}
    tot = sum(inv.values())
    return {k: v / tot for k, v in inv.items()}


def blend_probabilities(models: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    out = sum(weights[m] * models[m][list(OUTCOMES)] for m in weights)
    return out.div(out.sum(axis=1), axis=0)


def rolling_brier_weights(history: pd.DataFrame, asof, window_years: int = 3, horizon_months: int = 12) -> dict[str, float]:
    """Weights from forecasts resolved before `asof`.

    history: columns model, forecast_date, p_bull, p_base, p_bear, o_bull, o_base, o_bear.
    Only forecasts with forecast_date <= asof - horizon are resolved, and only the last `window_years`
    of those are used.
    """
    asof = pd.Timestamp(asof)
    latest = asof - pd.DateOffset(months=horizon_months)
    h = history[(history["forecast_date"] <= latest) & (history["forecast_date"] > latest - pd.DateOffset(years=window_years))]
    scores = {}
    for m, g in h.groupby("model"):
        O = g[["o_bull", "o_base", "o_bear"]].rename(columns=lambda c: c.replace("o_", "p_"))
        scores[m] = brier(g, O)
    return inverse_brier_weights(scores) if scores else {}


class ScenarioCalibrator:
    """One isotonic map per outcome, followed by renormalisation so probabilities sum to one."""

    def fit(self, probs: pd.DataFrame, outcomes: pd.DataFrame) -> "ScenarioCalibrator":
        self.maps = {c: IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(probs[c], outcomes[c]) for c in OUTCOMES}
        return self

    def transform(self, probs: pd.DataFrame) -> pd.DataFrame:
        cal = pd.DataFrame({c: self.maps[c].predict(probs[c]) for c in OUTCOMES}, index=probs.index)
        s = cal.sum(axis=1).replace(0, np.nan)
        cal = cal.div(s, axis=0).fillna(1 / 3)
        cal.columns = [f"{c}_cal" for c in OUTCOMES]
        return cal


def ece(p, o, n_bins: int = 10) -> float:
    p, o = np.asarray(p, float), np.asarray(o, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - o[m].mean())
    return float(total)


def reliability_table(p, o, n_bins: int = 10, overlap_lag: int = 12) -> pd.DataFrame:
    """Reliability diagram data with the effective sample size of each bucket's errors."""
    p, o = np.asarray(p, float), np.asarray(o, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        rows.append({"bucket": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": int(m.sum()),
                     "n_eff": effective_sample_size(p[m] - o[m], overlap_lag),
                     "mean_forecast": p[m].mean(), "observed_rate": o[m].mean()})
    return pd.DataFrame(rows)
