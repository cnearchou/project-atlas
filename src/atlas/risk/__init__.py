"""Section 8: Risk management (Spec §8.1–§8.9; Ref 3.10).

garch        §8.1  GJR-GARCH(1,1) with Student-t likelihood (arch package or own MLE)
var          §8.2  VaR and ES: filtered historical simulation, Monte Carlo, EVT (peaks over threshold)
attribution  §8.3  Euler risk decomposition, factor vs specific risk; §8.9 P&L and signal attribution
liquidity    §8.4  days to liquidate and liquidity-adjusted VaR
stress       §8.5  historical, hypothetical and reverse stress tests
sizing       §8.6  Kelly bound, transaction and borrow costs, optimiser, drawdown rule, capacity
validation   §8.7  Kupiec, Christoffersen and bias statistics
execution    §8.9  broker connection (paper trading), implementation shortfall
outputs      §8.8  assembly of the [risk] output table
"""
