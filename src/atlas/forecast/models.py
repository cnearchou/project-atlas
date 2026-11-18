"""Model calculations (Spec §7.3; Ref 3.9).

Ridge regression (baseline):
    beta_hat = (X^T X + lambda I)^{-1} X^T y,   prediction y_hat = x^T beta_hat
    lambda is chosen by purged inner cross-validation (Spec §7.2), maximising the mean daily IC.

Gradient-boosted trees:
    F_m(x) = F_{m-1}(x) + nu h_m(x),  h_m fitted to the current residuals,
    depth 4–6, at least 500 samples per leaf, learning rate 0.01–0.05, early stopping on validation IC.
    LightGBM is used when installed; otherwise scikit-learn's HistGradientBoostingRegressor, which
    implements the same histogram-based algorithm, is used.

Quantile models: the same trees trained with the pinball loss
    L_tau(y, q) = tau (y - q) if y >= q, else (1 - tau)(q - y),  tau in {0.1, 0.5, 0.9}.

Explanations: phi_{i,k} with  y_hat_i = phi_0 + sum_k phi_{i,k}.
    * Ridge: exact linear SHAP values phi_{i,k} = beta_k (x_{i,k} - mean_k).
    * LightGBM: exact TreeSHAP via predict(pred_contrib=True).
    * scikit-learn trees: the `shap` package when installed, otherwise a local linear surrogate
      (a ridge fit to the tree's own predictions), which is clearly labelled as an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from atlas._optional import optional_import
from atlas.forecast.cv import purged_kfold


def daily_ic(df: pd.DataFrame, pred_col: str = "pred", y_col: str = "target") -> pd.Series:
    """Spearman IC per date: rank correlation between prediction and outcome across stocks."""
    def _ic(g):
        if len(g) < 5:
            return np.nan
        return g[pred_col].rank().corr(g[y_col].rank())
    return df.groupby("date")[[pred_col, y_col]].apply(_ic)


def pinball_loss(y, q, tau: float) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


# ---- Ridge -----------------------------------------------------------------------------------
@dataclass
class RidgeModel:
    alphas: tuple = (0.1, 1.0, 10.0, 100.0, 1000.0)
    h: int = 21
    alpha_: float | None = None
    model: Ridge | None = None
    cv_table: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str], y: str = "target", tune: bool = True) -> "RidgeModel":
        d = df.dropna(subset=[y])
        if tune and d["date"].nunique() > 50:
            for a in self.alphas:
                ics = []
                for sp in purged_kfold(d["date"], self.h, n_splits=4):
                    tr, va = d[d["date"].isin(sp.train_dates)], d[d["date"].isin(sp.test_dates)].copy()
                    if len(tr) < 100 or len(va) < 20:
                        continue
                    va["pred"] = Ridge(alpha=a).fit(tr[cols], tr[y]).predict(va[cols])
                    ics.append(daily_ic(va, y_col=y).mean())
                self.cv_table[a] = float(np.nanmean(ics)) if ics else np.nan
            valid = {a: v for a, v in self.cv_table.items() if np.isfinite(v)}
            self.alpha_ = max(valid, key=valid.get) if valid else 10.0
        else:
            self.alpha_ = self.alpha_ or 10.0
        self.model = Ridge(alpha=self.alpha_).fit(d[cols], d[y])
        self.cols, self.mean_ = cols, d[cols].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.cols])

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exact SHAP values for a linear model."""
        return (df[self.cols] - self.mean_) * self.model.coef_


# ---- Gradient-boosted trees -----------------------------------------------------------------
@dataclass
class GBMModel:
    """Gradient-boosted trees for the mean (squared loss) or a quantile (pinball loss)."""

    quantile: float | None = None
    max_depth: int = 5
    min_leaf: int = 500
    learning_rate: float = 0.03
    max_iter: int = 600
    seed: int = 0

    def fit(self, df: pd.DataFrame, cols: list[str], y: str = "target", valid_frac: float = 0.15) -> "GBMModel":
        d = df.dropna(subset=[y]).sort_values("date")
        self.cols = cols
        min_leaf = int(min(self.min_leaf, max(20, len(d) // 200)))  # scaled down for small samples
        lgb = optional_import("lightgbm")
        if lgb is not None:
            cut = d["date"].quantile(1 - valid_frac)
            tr, va = d[d["date"] <= cut], d[d["date"] > cut]
            params = dict(learning_rate=self.learning_rate, max_depth=self.max_depth, num_leaves=2 ** self.max_depth - 1,
                          min_child_samples=min_leaf, n_estimators=self.max_iter, random_state=self.seed, verbose=-1)
            if self.quantile is not None:
                params.update(objective="quantile", alpha=self.quantile)
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(tr[cols], tr[y], eval_set=[(va[cols], va[y])],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
            self.backend = "lightgbm"
        else:
            kw = dict(max_depth=self.max_depth, min_samples_leaf=min_leaf, learning_rate=self.learning_rate,
                      max_iter=self.max_iter, early_stopping=True, validation_fraction=valid_frac,
                      n_iter_no_change=50, random_state=self.seed)
            if self.quantile is not None:
                kw.update(loss="quantile", quantile=self.quantile)
            self.model = HistGradientBoostingRegressor(**kw).fit(d[cols], d[y])
            self.backend = "sklearn"
        self._train_sample = d[cols].sample(min(len(d), 5000), random_state=self.seed)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.cols])

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols]
        if self.backend == "lightgbm":
            c = self.model.predict(X, pred_contrib=True)
            return pd.DataFrame(c[:, :-1], index=df.index, columns=self.cols)
        shap = optional_import("shap")
        if shap is not None:
            vals = shap.TreeExplainer(self.model).shap_values(X)
            return pd.DataFrame(vals, index=df.index, columns=self.cols)
        # Approximation: linear surrogate fitted to the model's own predictions on training data.
        S = self._train_sample
        sur = Ridge(alpha=1.0).fit(S, self.model.predict(S))
        return (X - S.mean()) * sur.coef_


def top_drivers(contrib: pd.DataFrame, k: int = 5) -> pd.Series:
    """The k features with the largest absolute contribution for each row, e.g. 'bear_gap(-0.21)'."""
    def _row(r):
        top = r.abs().sort_values(ascending=False).index[:k]
        return "; ".join(f"{c}({r[c]:+.3f})" for c in top)
    return contrib.apply(_row, axis=1)


def to_score(pred: pd.Series, dates: pd.Series) -> pd.Series:
    """Standardise predictions across stocks on each date (mean 0, sd 1): the score z of Spec §7.4."""
    g = pred.groupby(dates)
    return (pred - g.transform("mean")) / g.transform("std")
