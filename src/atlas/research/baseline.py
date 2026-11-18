"""Baseline comparison and ablation (Spec §9.7, Section A.2 H1).

Baseline: an equal-weighted composite of standardised, sector-neutralised momentum, value and quality,
    s_base = (z_MOM + z_VALUE + z_QUALITY) / 3.
Each month ΔIC_t = IC_full,t - IC_base,t; its mean must be positive with a Newey–West t above 3 (H1),
and the full model's net Sharpe must exceed the baseline's.

Ablation: each layer L is removed in turn, the model retrained, and
    ΔIC_L = IC_full - IC_{-L},   ΔBrier_L = Brier_{-L} - Brier_full
are measured. A layer that improves neither IC, calibration nor risk prediction is removed together
with any paid data it needs.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from atlas.data.standardise import neutralise, rank_gauss
from atlas.research.inference import nw_tstat_mean


def baseline_composite(cs: pd.DataFrame, cols=("momentum", "value", "quality"), sector_col: str = "sector") -> pd.Series:
    """One date's baseline score. cs indexed by entity with the three raw signals and sector."""
    zs = []
    for c in cols:
        z = rank_gauss(cs[c])
        zs.append(neutralise(z, cs[sector_col]) if sector_col in cs else z)
    return (sum(zs) / len(zs)).rename("baseline")


def compare_ic(ic_full: pd.Series, ic_base: pd.Series) -> dict:
    d = (ic_full - ic_base).dropna()
    m, se, t = nw_tstat_mean(d.values)
    return {"delta_ic_mean": m, "delta_ic_se": se, "t_nw": t, "n": len(d), "h1_pass": bool(m > 0 and t > 3)}


def ablation(layers: dict[str, list[str]], all_features: list[str],
             evaluate: Callable[[list[str]], dict]) -> pd.DataFrame:
    """`evaluate(features)` retrains on the feature list and returns {'ic': ..., 'brier': ...}."""
    full = evaluate(all_features)
    rows = []
    for layer, feats in layers.items():
        kept = [f for f in all_features if f not in feats]
        res = evaluate(kept)
        rows.append({"layer": layer, "n_features_removed": len(feats), "ic_without": res["ic"],
                     "delta_ic": full["ic"] - res["ic"],
                     "delta_brier": res.get("brier", float("nan")) - full.get("brier", float("nan"))})
    out = pd.DataFrame(rows)
    out["keep"] = (out["delta_ic"] > 0) | (out["delta_brier"] > 0)
    return out
