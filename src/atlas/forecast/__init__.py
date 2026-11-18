"""Section 7: Forecasting engine (Spec §7.1–§7.9; Ref 3.9).

panel        §7.1  training panel: features, missing indicators, forward residual-return targets
cv           §7.2  expanding walk-forward splits with purging and embargo
models       §7.3  ridge baseline, gradient-boosted trees, quantile models, SHAP-style explanations
expected     §7.4  Grinold conversion and regime-weighted expected factor returns E[f]
volatility   §7.5  blend of GARCH and implied volatility, horizon scaling
scenarios    §7.6  Student-t scenario probabilities (absolute and relative, Section A.3)
calibration  §7.7  Brier score, inverse-Brier blending, isotonic calibration, ECE, effective sample size
simulation   §7.8  joint Monte Carlo of regimes, factors, specific returns and network propagation
outputs      §7.9  assembly of the [forecasts] output table
"""
