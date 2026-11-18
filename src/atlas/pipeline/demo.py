"""End-to-end run of Project Atlas on synthetic data (Spec §10.1 and the Build Plan, Section 11).

The demo follows the build phases in order and uses the same functions that would run on real data:

Phase 1   Data foundation (Section 0): prices -> total returns -> delisting returns -> universe ->
          point-in-time fundamentals joined as of each date.
Phase 4   Sector and factor model (Section 6): daily constrained WLS factor returns, EWMA factor
          covariance, specific risk.
Phase 3/6 Signals (Sections 1 and 5): momentum, reversal, value, quality, volatility, size and a
          deliberately useless noise signal that the research harness should reject.
Phase 2   Testing harness (Section 9): IC, Newey–West t, quantile backtests, Benjamini–Hochberg,
          deflated Sharpe ratio, production gate and research registry (H1–H3 pre-registered).
Phase 7   Macro (Section 2): HMM regimes from macro series and regime-weighted E[f] (Spec §7.4).
Phase 8   Forecast engine (Section 7): purged walk-forward ridge and gradient-boosted trees at 1-month
          and 12-month horizons, Grinold conversion, Student-t scenarios, historical Brier/ECE,
          isotonic calibration, DCF Monte Carlo for a sample of companies and inverse-Brier blending.
Phase 5/9 Risk and portfolio (Section 8): GJR-GARCH, FHS/EVT/Monte Carlo VaR and ES, Euler
          attribution, stress tests, costed optimisation, VaR backtests.
Phase 14  Monitoring (Section 9.5): rolling IC, CUSUM and PSI.

Layers that need data the synthetic market does not contain (news, flows, options, analyst
estimates) are switched off, exactly as the Minimum Viable Data Plan in Section A.4 prescribes.
Every result is written to data/outputs so it can be inspected after the run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from atlas._compat import MONTH_END
from atlas.config import load_settings
from atlas.data.corporate_actions import total_returns
from atlas.data.point_in_time import asof_join
from atlas.data.standardise import rank_gauss
from atlas.data.storage import DataStore
from atlas.data.universe import apply_delisting_returns, eligible_universe
from atlas.behaviour.signals import momentum_raw, monthly_returns
from atlas.forecast import calibration as cal
from atlas.forecast.cv import walk_forward_splits
from atlas.forecast.expected import expected_factor_returns, expected_factor_returns_path, grinold_alpha, regime_factor_means
from atlas.forecast.models import GBMModel, RidgeModel, daily_ic, to_score, top_drivers
from atlas.forecast.outputs import assemble_forecasts
from atlas.forecast.scenarios import fit_nu, quantile_disagreement, scenario_table, t_probabilities
from atlas.forecast.simulation import simulate_joint
from atlas.fundamentals.monte_carlo import DCFInputs, convert_to_12m_returns, run_dcf_monte_carlo, scenario_probabilities
from atlas.ingest.synthetic import make_market
from atlas.macro.regimes import GaussianHMM, label_states
from atlas.research import metrics
from atlas.research.backtest import quantile_portfolios
from atlas.research.baseline import compare_ic
from atlas.research.gate import production_gate
from atlas.research.hypotheses import register_all
from atlas.research.inference import benjamini_hochberg, deflated_sharpe_ratio, effective_sample_size, nw_tstat_mean
from atlas.research.monitoring import cusum, psi, rolling_ic, status
from atlas.research.registry import Registry
from atlas.risk import validation
from atlas.risk.attribution import euler_decomposition, factor_specific_split
from atlas.risk.garch import fit_gjr_garch, standardised_residuals
from atlas.risk.liquidity import days_to_liquidate, lvar
from atlas.risk.outputs import assemble_risk, max_position
from atlas.risk.sizing import OptimiserInputs, kelly_cap, optimise_to_target_vol
from atlas.risk.stress import historical_stress, reverse_stress
from atlas.risk.var import evt_var_es, fhs_var_es, mc_portfolio_var_es
from atlas.sectors.factor_model import ewma_covariance, estimate_factor_returns, specific_variance
from scipy.stats import norm

log = logging.getLogger("atlas.demo")
FEATURES = ["momentum", "reversal", "value", "quality", "volatility", "size", "noise"]


def _say(msg: str, verbose: bool):
    log.info(msg)
    if verbose:
        print(msg, flush=True)


def run_demo(n_stocks: int = 200, n_days: int = 1260, seed: int = 7, out_dir: str | Path | None = None,
             verbose: bool = True, fast: bool = False) -> dict:
    """Run every layer once. `fast=True` shrinks simulation sizes for tests."""
    t0 = time.time()
    S = load_settings()
    out = Path(out_dir) if out_dir else S.path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    store = DataStore(out / "store")
    reg = Registry(out / "research_registry.csv")
    rng = np.random.default_rng(seed)
    results: dict = {}

    # ===== Phase 1: data foundation (Section 0) ==================================================
    _say("Phase 1  Data foundation (Section 0)", verbose)
    mkt = make_market(n_stocks=n_stocks, n_days=n_days, seed=seed)
    px = mkt.prices.copy()
    px["ret_total"] = total_returns(px)                    # Spec §0.3
    id_err = float((px["ret_total"] - px["ret"]).abs().max())
    px["ret"] = px["ret_total"]
    px = apply_delisting_returns(px, mkt.delistings)      # Spec §0.5
    px["market_cap"] = px["price"] * px["shares_outstanding"]
    px["eligible"] = eligible_universe(px, S["universe"]["min_price"], S["universe"]["min_market_cap"],
                                       S["universe"]["min_adv_dollars"]).values
    px = px.merge(mkt.companies[["entity_id", "sector"]], on="entity_id")
    for name, df in {"prices": mkt.prices, "fundamentals_pit": mkt.fundamentals}.items():
        if not store.exists("raw", name):
            store.write("raw", name, df)
    store.write("clean", "daily_panel", px)
    _say(f"  {px['entity_id'].nunique()} companies, {px['date'].nunique()} days; total-return identity error "
         f"{id_err:.1e}; {len(mkt.delistings)} delistings merged", verbose)

    # Point-in-time fundamentals: one row per (entity, quarter) known from its filing date (Spec §0.4).
    fw = mkt.fundamentals.pivot_table(index=["entity_id", "event_date", "knowledge_date"], columns="field",
                                      values="value").reset_index()
    fw = fw.sort_values(["entity_id", "event_date"])
    fw["revenue_ttm"] = fw.groupby("entity_id")["revenue"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    fw["ni_ttm"] = fw.groupby("entity_id")["net_income"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    fw["ebit_ttm"] = fw.groupby("entity_id")["ebit"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    fw["gp_ttm"] = fw.groupby("entity_id")["gross_profit"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    fw["rev_growth"] = fw.groupby("entity_id")["revenue_ttm"].pct_change(4)
    daily = asof_join(px, fw.drop(columns="event_date"), max_age=pd.Timedelta(days=200))
    daily = daily.sort_values(["entity_id", "date"]).reset_index(drop=True)

    # ===== Signals (Sections 1, 5 and 6 styles) ================================================
    _say("Phase 3  Signals: momentum, reversal, value, quality, volatility, size, noise", verbose)
    daily["log_mcap"] = np.log(daily["market_cap"])
    daily["book_to_price"] = daily["equity"] / daily["market_cap"]
    daily["earnings_yield"] = daily["ni_ttm"] / daily["market_cap"]
    daily["roa"] = daily["ni_ttm"] / daily["total_assets"]
    daily["gross_profitability"] = daily["gp_ttm"] / daily["total_assets"]
    daily["vol_63"] = daily.groupby("entity_id")["ret"].transform(lambda s: s.rolling(63, min_periods=40).std())

    mret = monthly_returns(daily)                                   # month x entity
    rwide = daily.pivot_table(index="date", columns="entity_id", values="ret")
    mom = momentum_raw(mret)
    month_ends = daily.groupby(daily["date"].dt.to_period("M"))["date"].max().values
    snap = daily[daily["date"].isin(month_ends)].copy()
    snap["month"] = snap["date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    mom_long = mom.stack().rename("momentum").reset_index()
    mom_long.columns = ["month", "entity_id", "momentum"]
    mom_long["month"] = mom_long["month"].dt.normalize()
    rev_long = (-mret).stack().rename("reversal").reset_index()
    rev_long.columns = ["month", "entity_id", "reversal"]
    rev_long["month"] = rev_long["month"].dt.normalize()
    snap = snap.merge(mom_long, on=["month", "entity_id"], how="left").merge(rev_long, on=["month", "entity_id"], how="left")
    snap["value"] = (snap.groupby("date")["book_to_price"].transform(rank_gauss)
                     + snap.groupby("date")["earnings_yield"].transform(rank_gauss)) / 2
    snap["quality"] = (snap.groupby("date")["roa"].transform(rank_gauss)
                       + snap.groupby("date")["gross_profitability"].transform(rank_gauss)) / 2
    snap["volatility"] = snap["vol_63"]
    snap["size"] = snap["log_mcap"]
    snap["noise"] = rng.standard_normal(len(snap))              # no information by construction
    snap = snap[snap["eligible"]].copy()

    # ===== Phase 4: factor model (Section 6.3) ==================================================
    _say("Phase 4  Factor model: daily constrained WLS, EWMA covariance, specific risk (Section 6.3)", verbose)
    styles = ["size", "value", "momentum", "volatility"]
    fm_panel = daily[["date", "entity_id", "ret", "sector", "market_cap", "log_mcap", "book_to_price", "vol_63"]].copy()
    fm_panel = fm_panel.merge(snap[["date", "entity_id", "momentum"]], on=["date", "entity_id"], how="left")
    fm_panel["momentum"] = fm_panel.groupby("entity_id")["momentum"].ffill(limit=23)
    g = fm_panel.groupby("entity_id")
    # Exposures are measured at the previous close so they never use the return they explain.
    fm_panel["size"] = g["log_mcap"].shift(1)
    fm_panel["value"] = g["book_to_price"].shift(1)
    fm_panel["volatility"] = g["vol_63"].shift(1)
    fm_panel["momentum"] = g["momentum"].shift(1)
    fm_panel["market_cap"] = g["market_cap"].shift(1)
    fm_panel = fm_panel[fm_panel["date"] > fm_panel["date"].min() + pd.Timedelta(days=100)]
    sectors = sorted(mkt.companies["sector"].unique())
    fm = estimate_factor_returns(fm_panel, sectors, styles)
    f_ret, resid = fm.factor_returns, fm.residuals
    F_daily = ewma_covariance(f_ret, S["factor_model"]["cov_half_life_days"])
    d2_daily = specific_variance(resid.iloc[-500:])
    true_mkt = mkt.factor_returns["market"].reindex(f_ret.index)
    results["factor_market_corr_with_truth"] = float(f_ret["market"].corr(true_mkt))
    _say(f"  {len(f_ret)} daily regressions; estimated market factor vs true: corr = "
         f"{results['factor_market_corr_with_truth']:.3f}", verbose)

    # Residual momentum (Spec §5.2) from monthly sums of factor-model residuals.
    resid_m = resid.resample(MONTH_END).sum(min_count=10)
    resid_m.index = resid_m.index.normalize()
    rm = resid_m.shift(1).rolling(11, min_periods=9).sum().stack().rename("momentum_resid").reset_index()
    rm.columns = ["month", "entity_id", "momentum_resid"]
    snap = snap.merge(rm, on=["month", "entity_id"], how="left")
    FEAT = [*FEATURES, "momentum_resid"]

    # ===== Phase 2/3: research harness on single signals (Section 9) =============================
    _say("Phase 2  Research harness: IC, Newey–West t, quantile backtests, BH, DSR, gate (Section 9)", verbose)
    register_all(reg)
    fwd1 = mret.shift(-1)
    fwd1.index = fwd1.index.normalize()
    sig_rows, ls_series = [], {}
    for f in FEAT:
        wide = snap.pivot_table(index="month", columns="entity_id", values=f)
        ic = metrics.ic_series(wide, fwd1)
        bt = quantile_portfolios(wide, fwd1, n_q=5, cost_per_turnover=0.002)
        m, se, t = nw_tstat_mean(ic.values)
        p = 2 * (1 - norm.cdf(abs(t))) if np.isfinite(t) else 1.0
        sr = bt["long_short_net"].mean() / bt["long_short_net"].std()
        ls_series[f] = bt["long_short_net"]
        sig_rows.append({"signal": f, "ic_mean": m, "ic_sd": ic.std(), "t_nw": t, "p_value": p,
                         "sharpe_net": metrics.sharpe(bt["long_short_net"]), "sr_period": sr,
                         "net_mean": bt["long_short_net"].mean(), "n_months": len(ic)})
    sig = pd.DataFrame(sig_rows).set_index("signal")
    sig["bh_survives"] = benjamini_hochberg(sig["p_value"].values, S["testing"]["fdr_q"])
    sr_var = float(sig["sr_period"].var())
    sig["dsr"] = [deflated_sharpe_ratio(r.sr_period, sr_var, len(sig), r.n_months) for r in sig.itertuples()]
    sig["passed_gate"] = [production_gate(t_nw=r.t_nw, bh_survives=r.bh_survives, net_mean=r.net_mean, dsr=r.dsr,
                                          fm_t=r.t_nw, split_ics=None).passed for r in sig.itertuples()]
    for r in sig.itertuples():
        reg.log(hypothesis_text=f"Single-signal test of {r.Index}", signal_definition=r.Index, universe="synthetic",
                period=f"{snap['month'].min():%Y-%m}..{snap['month'].max():%Y-%m}", n_variants_tried=len(sig),
                ic_mean=r.ic_mean, ic_sd=r.ic_sd, t_nw=r.t_nw, sharpe=r.sharpe_net, dsr=r.dsr, passed_gate=r.passed_gate)
    sig.to_csv(out / "signal_tests.csv")
    results["signal_tests"] = sig
    if verbose:
        print(sig[["ic_mean", "t_nw", "sharpe_net", "bh_survives", "dsr"]].round(3).to_string())

    # ===== Phase 7: macro regimes and expected factor returns (Section 2.5, 7.4) =================
    _say("Phase 7  Macro: HMM regimes and regime-weighted expected factor returns (Sections 2.5, 7.4)", verbose)
    Y = mkt.macro[["growth", "inflation", "credit_spread"]]
    Yz = (Y - Y.mean()) / Y.std()
    hmm = GaussianHMM(2).fit(Yz.values, seed=seed)
    probs = pd.DataFrame(hmm.filter(Yz.values), index=Y.index, columns=label_states(hmm))
    rec_col = int(np.argmin(hmm.means[:, 0]))
    agree = float(((probs.iloc[:, rec_col] > 0.5).astype(int) == mkt.macro["true_regime"]).mean())
    results["regime_accuracy"] = agree
    f_month = f_ret.resample(MONTH_END).sum()
    f_month.index = f_month.index.normalize()
    reg_now = probs.idxmax(axis=1)
    reg_now.index = reg_now.index.normalize()
    rmeans = regime_factor_means(f_month, reg_now, n0=60)   # strong shrinkage: factor timing is weak
    # The market term uses r_f + ERP in production (Spec §2.7). The synthetic market has no index cash
    # flows for an implied ERP, so the long-run average market return is used as its stand-in.
    mkt_lr_month = float(f_month["market"].mean())
    p_next = probs.iloc[-1].values @ hmm.T
    order = [list(probs.columns).index(s) for s in rmeans.index]
    ef_1m = expected_factor_returns(rmeans, p_next[order], market_override=mkt_lr_month)
    T_ord = hmm.T[np.ix_(order, order)]
    ef_12m = expected_factor_returns_path(rmeans, probs.iloc[-1].values[order], T_ord, 12, market_override=12 * mkt_lr_month)
    _say(f"  regime labels {list(probs.columns)}; filtered regime matches truth {agree:.0%} of months; "
         f"E[market, 1m] = {ef_1m['market']:.2%}", verbose)

    # ===== Phase 8: forecasting engine (Section 7) ==============================================
    _say("Phase 8  Forecast engine: purged walk-forward ridge + boosted trees, 1m and 12m (Section 7)", verbose)
    month_end_days = pd.DatetimeIndex(sorted(snap["date"].unique()))
    tgt1 = resid.rolling(21, min_periods=15).sum().shift(-21)        # forward 1-month residual (Spec §7.1)
    tgt12 = resid.rolling(252, min_periods=200).sum().shift(-252)     # forward 12-month residual
    panel = snap[["date", "month", "entity_id", "sector", *FEAT]].copy()
    for name, T_ in (("y1", tgt1), ("y12", tgt12)):
        t_long = T_.reindex(month_end_days).stack().rename(name).reset_index()
        t_long.columns = ["date", "entity_id", name]
        panel = panel.merge(t_long, on=["date", "entity_id"], how="left")
    for c in FEAT:
        panel[f"{c}_missing"] = panel[c].isna().astype(float)
        panel[c] = panel.groupby("date")[c].transform(rank_gauss).fillna(0.0)
    panel["target"] = panel.groupby("date")["y1"].transform(rank_gauss)
    panel["target12"] = panel.groupby("date")["y12"].transform(rank_gauss)
    cols = FEAT + [f"{c}_missing" for c in FEAT if panel[f"{c}_missing"].std() > 0]
    model_cols = [c for c in cols if c != "noise" and c != "noise_missing"] + ["noise"]  # noise kept as a check

    dates = np.array(sorted(panel["date"].unique()))
    oos = []
    gbm = None
    for i, sp in enumerate(walk_forward_splits(dates, h=1, min_train=24, test_size=1)):
        tr = panel[panel["date"].isin(sp.train_dates)].dropna(subset=["target"])
        te = panel[panel["date"].isin(sp.test_dates)].copy()
        rid = RidgeModel(h=1).fit(tr, model_cols, tune=(i % 6 == 0))
        te["pred_ridge"] = rid.predict(te)
        if gbm is None or i % (12 if fast else 6) == 0:                  # retrain trees twice a year
            gbm = GBMModel(max_iter=150 if fast else 400).fit(tr, model_cols)
        te["pred_gbm"] = gbm.predict(te)
        te["pred"] = 0.5 * to_score(pd.Series(te["pred_ridge"], index=te.index), te["date"]) + \
            0.5 * to_score(pd.Series(te["pred_gbm"], index=te.index), te["date"])
        oos.append(te)
    oos = pd.concat(oos)
    ic_full = daily_ic(oos.dropna(subset=["target"]), "pred", "target")
    oos["baseline"] = (oos["momentum"] + oos["value"] + oos["quality"]) / 3          # Spec §9.7 baseline
    ic_base = daily_ic(oos.dropna(subset=["target"]), "baseline", "target")
    h1 = compare_ic(ic_full, ic_base)
    ic1 = float(ic_full.mean())
    m1, se1, t1 = nw_tstat_mean(ic_full.values)
    results["oos_ic_1m"], results["oos_t_1m"], results["H1"] = ic1, t1, h1
    _say(f"  1-month OOS IC = {ic1:.4f} (NW t = {t1:.2f}); baseline IC = {ic_base.mean():.4f}; "
         f"H1 ΔIC t = {h1['t_nw']:.2f} -> {'PASS' if h1['h1_pass'] else 'not passed'}", verbose)
    reg.log(hypothesis_text="H1 test on synthetic data", signal_definition="H1", ic_mean=ic1, t_nw=h1["t_nw"],
            passed_gate=h1["h1_pass"], universe="synthetic")

    # 12-month model (ridge) for scenario forecasts; the 12-month IC is measured at 12 months (Spec §7.4).
    oos12 = []
    for sp in walk_forward_splits(dates, h=12, min_train=24, test_size=1):
        tr = panel[panel["date"].isin(sp.train_dates)].dropna(subset=["target12"])
        if tr["date"].nunique() < 12:
            continue
        te = panel[panel["date"].isin(sp.test_dates)].copy()
        r12 = RidgeModel(h=12).fit(tr, model_cols, y="target12", tune=False)
        te["pred12"] = r12.predict(te)
        oos12.append(te)
    oos12 = pd.concat(oos12) if oos12 else pd.DataFrame(columns=[*panel.columns, "pred12"])
    ic12_series = daily_ic(oos12.dropna(subset=["target12"]), "pred12", "target12") if len(oos12) else pd.Series(dtype=float)
    ic12 = float(ic12_series.mean()) if ic12_series.notna().sum() >= 3 else ic1 * 0.5
    n_eff12 = effective_sample_size(ic12_series.dropna().values, 12) if ic12_series.notna().sum() > 3 else np.nan
    _say(f"  12-month OOS IC = {ic12:.4f} over {ic12_series.notna().sum()} months (overlapping; n_eff ~ {n_eff12:.1f})", verbose)

    # ---- Calibration history: statistical 12m forecasts scored once their 12 months elapsed (Spec §7.7)
    nu = fit_nu(np.concatenate([(resid[c] / resid[c].rolling(63, min_periods=40).std().shift(1)).dropna().values
                                for c in resid.columns[:50]]))
    hist = oos12.dropna(subset=["y12"]).copy()
    if len(hist):
        hist["z12"] = to_score(hist["pred12"], hist["date"])
        rv = resid.rolling(126, min_periods=60).std().reindex(pd.DatetimeIndex(hist["date"].unique()))
        rv_long = rv.stack().rename("rv").reset_index()
        rv_long.columns = ["date", "entity_id", "rv"]
        hist = hist.merge(rv_long, on=["date", "entity_id"], how="left").dropna(subset=["rv"])
        hist["alpha12"] = ic12 * hist["rv"] * np.sqrt(252) * hist["z12"]
        P = t_probabilities(hist["alpha12"].values, hist["rv"].values * np.sqrt(252), nu, S["scenarios"]["relative_threshold"])
        O = cal.outcomes_from_returns(hist["y12"].values, S["scenarios"]["relative_threshold"])
        bs_stat = cal.brier(P, O)
        clim = pd.DataFrame(np.tile(O.mean().values, (len(O), 1)), columns=O.columns)   # climatology reference
        bs_clim = cal.brier(clim, O)
        ece_bull = cal.ece(P["p_bull"], O["p_bull"])
        n_eff_cal = effective_sample_size((P["p_bull"] - O["p_bull"]).values, 12)
        # Absolute scenarios (Section A.3): total 12-month return against +/-15%. The expected factor
        # part uses only market history available at each forecast date (expanding mean).
        tot = rwide.reindex(columns=resid.columns)
        fwd_tot12 = np.expm1(np.log1p(tot).rolling(252, min_periods=200).sum().shift(-252))
        vol_tot = tot.rolling(126, min_periods=60).std()
        mkt_exp = f_ret["market"].expanding(120).mean() * 252
        hd = pd.DatetimeIndex(hist["date"])
        hist["r12_total"] = fwd_tot12.stack().reindex(pd.MultiIndex.from_arrays([hd, hist["entity_id"]])).values
        hist["vol_total"] = vol_tot.stack().reindex(pd.MultiIndex.from_arrays([hd, hist["entity_id"]])).values * np.sqrt(252)
        hist["mu_abs"] = hist["alpha12"] + mkt_exp.reindex(hd).values
        ha = hist.dropna(subset=["r12_total", "vol_total", "mu_abs"])
        P_abs = t_probabilities(ha["mu_abs"].values, ha["vol_total"].values, nu, S["scenarios"]["absolute_threshold"])
        O_abs = cal.outcomes_from_returns(ha["r12_total"].values, S["scenarios"]["absolute_threshold"])
        bs_abs = cal.brier(P_abs, O_abs)
        calibrator = cal.ScenarioCalibrator().fit(P_abs, O_abs)   # same event as the absolute output columns
        ece_abs = cal.ece(P_abs["p_bull"], O_abs["p_bull"])
    else:
        bs_stat = bs_clim = ece_bull = n_eff_cal = bs_abs = ece_abs = np.nan
        calibrator = None
    results.update(brier_stat_rel=bs_stat, brier_climatology=bs_clim, ece_rel_bull=ece_bull, brier_stat_abs=bs_abs,
                   ece_abs_bull=ece_abs)
    _say(f"  relative-scenario Brier = {bs_stat:.4f} (climatology {bs_clim:.4f}); ECE(bull) = {ece_bull:.4f}; "
         f"n_eff = {n_eff_cal:.0f}; absolute Brier = {bs_abs:.4f}; Student-t nu = {nu:.2f}", verbose)

    # ---- Final models on all labelled data, scored on the latest cross-section ------------------
    last_date = dates[-1]
    train_all = panel[panel["date"] < dates[-1 - 1]].dropna(subset=["target"])
    rid = RidgeModel(h=1).fit(train_all, model_cols)
    gbm = GBMModel(max_iter=150 if fast else 400).fit(train_all, model_cols)
    lab12 = panel.dropna(subset=["target12"])
    r12 = RidgeModel(h=12).fit(lab12, model_cols, y="target12", tune=False) if len(lab12) else rid
    now = panel[panel["date"] == last_date].set_index("entity_id")
    z1 = 0.5 * to_score(pd.Series(rid.predict(now), index=now.index), pd.Series(last_date, index=now.index)) + \
        0.5 * to_score(pd.Series(gbm.predict(now), index=now.index), pd.Series(last_date, index=now.index))
    z1 = (z1 - z1.mean()) / z1.std()
    z12 = to_score(pd.Series(r12.predict(now), index=now.index), pd.Series(last_date, index=now.index))
    # Quantile models of the 12-month residual return (pinball loss, Spec §7.3), cross-checked in §7.6.
    q = {tau: GBMModel(quantile=tau, max_iter=100 if fast else 200).fit(lab12, model_cols, y="y12").predict(now)
         for tau in (0.1, 0.5, 0.9)}
    contrib = rid.contributions(now)
    drivers = top_drivers(contrib)

    # Volatility: GJR-GARCH per stock (Spec §8.1), scaled to the horizon (Spec §7.5).
    ids = list(now.index)
    garch = {}
    for eid in ids:
        r = rwide[eid].dropna().iloc[-750:]
        try:
            garch[eid] = fit_gjr_garch(r.values, use_arch=True)
        except Exception:  # noqa: BLE001 - a failed fit falls back to EWMA volatility
            garch[eid] = None
    sig_d = pd.Series({e: (g.next_sigma if g else rwide[e].dropna().iloc[-63:].std()) for e, g in garch.items()})
    sig_1m = pd.Series({e: (g.horizon_vol(21) if g else sig_d[e] * np.sqrt(21)) for e, g in garch.items()})
    sig_12m = pd.Series({e: (g.horizon_vol(252) if g else sig_d[e] * np.sqrt(252)) for e, g in garch.items()})
    d_spec = np.sqrt(d2_daily.reindex(ids).fillna(d2_daily.mean()))
    alpha1 = grinold_alpha(z1, ic1, d_spec * np.sqrt(21))
    alpha12 = grinold_alpha(z12, ic12, d_spec * np.sqrt(252))
    B_now = fm.exposures[f_ret.index[-1]].reindex(ids).fillna(0.0)
    mu12 = B_now[ef_12m.index.intersection(B_now.columns)] @ ef_12m[ef_12m.index.intersection(B_now.columns)] + alpha12
    scen = scenario_table(mu12, sig_12m, alpha12, d_spec * np.sqrt(252), nu,
                          S["scenarios"]["absolute_threshold"], S["scenarios"]["relative_threshold"])
    flags = quantile_disagreement(alpha12.values, (d_spec * np.sqrt(252)).values, nu, q[0.1], q[0.9])
    results["quantile_flags"] = int(flags.sum())

    # DCF Monte Carlo for a sample of companies and inverse-Brier blending (Spec §1.5, §7.7).
    latest_f = daily[daily["date"] == daily["date"].max()].set_index("entity_id")
    has_fund = latest_f.dropna(subset=["revenue_ttm", "ebit_ttm"]).index
    sample = [e for e in ids if e in has_fund][: (5 if fast else 25)]
    sector_margin = (latest_f["ebit_ttm"] / latest_f["revenue_ttm"]).groupby(latest_f["sector"]).median()
    dcf_rows = {}
    for eid in sample:
        row = latest_f.loc[eid]
        x = DCFInputs(revenue0=row["revenue_ttm"], margin0=float(np.clip(row["ebit_ttm"] / row["revenue_ttm"], 0.01, 0.5)),
                      g0=float(np.nan_to_num(row["rev_growth"], nan=0.05)),
                      m_sector=float(sector_margin.get(row["sector"], 0.15)),
                      net_debt=row["total_debt"] - row["cash"], shares=row["shares_outstanding"],
                      financial=False)
        mc = run_dcf_monte_carlo(x, n=1000 if fast else 4000, seed=seed)
        fac12 = float(mu12[eid] - alpha12[eid])
        r12d = convert_to_12m_returns(mc.values, row["price"], kappa=0.3, factor_return_12m=fac12,
                                      resid_vol_12m=float(d_spec[eid] * np.sqrt(252)), nu=nu)
        p = scenario_probabilities(r12d, S["scenarios"]["absolute_threshold"])
        dcf_rows[eid] = {"dcf_mean": mc.mean, "price": row["price"], **{f"{k}_dcf": v for k, v in p.items()}}
    dcf = pd.DataFrame(dcf_rows).T
    # No DCF forecasts have resolved yet in the synthetic history, so the weights fall back to equal
    # weights; with a live track record they come from rolling_brier_weights().
    w_blend = {"dcf": 0.5, "stat": 0.5}
    blended = scen[["p_bull", "p_base", "p_bear"]].copy()
    if len(dcf):
        dd = dcf[["p_bull_dcf", "p_base_dcf", "p_bear_dcf"]].astype(float).rename(columns=lambda c: c.replace("_dcf", ""))
        both = cal.blend_probabilities({"dcf": dd, "stat": scen.loc[dd.index, ["p_bull", "p_base", "p_bear"]]}, w_blend)
        blended.loc[both.index] = both.values
    cal_cols = calibrator.transform(blended) if calibrator is not None else pd.DataFrame(index=blended.index)
    forecasts = assemble_forecasts([
        pd.DataFrame({"score": z12, "alpha": alpha12, "mu": mu12, "sigma": sig_12m}),
        pd.DataFrame({"q10": q[0.1], "q50": q[0.5], "q90": q[0.9]}, index=now.index),
        blended, cal_cols, scen[["p_bull_rel", "p_base_rel", "p_bear_rel"]],
        drivers.rename("top_5_shap_drivers").to_frame()], last_date, "12m")
    forecasts.to_csv(out / "forecasts.csv", index=False)
    dcf.to_csv(out / "dcf_sample.csv")
    store.write("outputs", "forecasts", forecasts)
    results["forecasts"] = forecasts
    _say(f"  forecasts table: {len(forecasts)} rows; mean P(bull/base/bear) = "
         f"{forecasts['p_bull'].mean():.1%}/{forecasts['p_base'].mean():.1%}/{forecasts['p_bear'].mean():.1%}", verbose)

    # ===== Phase 5/9: risk and portfolio construction (Section 8) ================================
    _say("Phase 5  Risk: optimiser, VaR/ES (FHS, EVT, Monte Carlo), attribution, stress tests (Section 8)", verbose)
    F_1m = F_daily * 21
    B_cols = list(f_ret.columns)
    B = B_now[B_cols]
    Sigma = pd.DataFrame(B.values @ F_1m @ B.values.T + np.diag((d_spec ** 2 * 21).values), index=ids, columns=ids)
    mkt_ret = f_ret["market"]
    beta_ts = pd.Series({e: np.cov(rwide[e].reindex(mkt_ret.index).iloc[-252:].fillna(0), mkt_ret.iloc[-252:])[0, 1]
                         / mkt_ret.iloc[-252:].var() for e in ids})
    adv = latest_f.reindex(ids)["price"] * daily[daily["date"] > daily["date"].max() - pd.Timedelta(days=30)].groupby("entity_id")["volume"].mean().reindex(ids)
    aum = 5e7   # a small fund in paper trading (Section 11, Phase 15)
    sector_cols = [c for c in B.columns if c.startswith("sector_")]
    inp = OptimiserInputs(mu=alpha1, Sigma=Sigma, sector_exposure=B[sector_cols], beta=beta_ts,
                          sigma_daily=sig_d, adv_dollars=adv, spread=pd.Series(0.001, index=ids))
    w, lam = optimise_to_target_vol(inp, S["mandate"]["target_volatility"] / np.sqrt(12), risk_aversion=2.0,
                                    w_max=S["mandate"]["max_position"], gross_max=S["mandate"]["gross_exposure_max"],
                                    net_max=S["mandate"]["net_exposure_max"], exposure_eps=0.02, aum=aum,
                                    cost_amortisation=3.0)
    w = w.where(w.abs() > 1e-4, 0.0)
    port_vol_ann = float(np.sqrt(w @ Sigma @ w) * np.sqrt(12))
    eul = euler_decomposition(w, Sigma)
    split = factor_specific_split(w, B, F_1m, d_spec ** 2 * 21)
    port_hist = (rwide[ids].iloc[-750:].fillna(0.0) @ w.values)
    pg = fit_gjr_garch(port_hist.values)
    zp = standardised_residuals(port_hist.values, pg)
    var95, es95 = fhs_var_es(zp, pg.next_sigma, 0.95)
    evt = evt_var_es(-port_hist.values, 0.99)
    # Monte Carlo VaR from the joint simulation (Spec §7.8): two regimes with scaled factor covariance.
    reg_cov = [F_daily * 0.8, F_daily * 1.8]
    reg_mu = [np.zeros(len(B_cols)), np.zeros(len(B_cols))]
    sim = simulate_joint(B, probs.iloc[-1].values, hmm.T, reg_mu, reg_cov, d_spec ** 2, n_paths=2000 if fast else 10000, nu=nu)
    mc_var, mc_es = mc_portfolio_var_es(sim, w.values, 0.95)
    stress = historical_stress(B, w)
    F_ann = F_daily * 252
    rev = reverse_stress(B, w, F_ann, loss=0.20)
    # VaR backtest (Spec §8.7) on rolling one-day parametric-t VaR forecasts of the current portfolio.
    s_hist = pd.Series(pg.sigma, index=port_hist.index)
    q95 = -np.quantile(zp, 0.05)
    breaches = (-port_hist.values) > q95 * s_hist.values
    kup = validation.kupiec(breaches[-500:], 0.05)
    chr_ = validation.christoffersen(breaches[-500:])
    bias = validation.bias_statistic(port_hist.values[-500:], s_hist.values[-500:])
    results.update(port_vol_ann=port_vol_ann, var95=var95, es95=es95, evt99=evt, mc_var95=mc_var, kupiec=kup,
                   christoffersen=chr_, bias=bias, gross=float(w.abs().sum()), net=float(w.sum()))
    _say(f"  gross {w.abs().sum():.2f}, net {w.sum():+.3f}, {int((w != 0).sum())} positions, predicted vol "
         f"{port_vol_ann:.1%}/yr; factor share of risk {split['factor_risk_pct']:.0%}", verbose)
    _say(f"  1-day VaR95 FHS {var95:.2%} / ES95 {es95:.2%}; MC VaR95 {mc_var:.2%}; EVT VaR99 {evt['var']:.2%}; "
         f"Kupiec LR {kup['lr']:.2f} ({'reject' if kup['reject'] else 'ok'}); bias {bias['bias']:.2f}", verbose)

    # Per-stock risk table (Spec §8.8).
    rows = {}
    for e in ids:
        g = garch[e]
        r_e = rwide[e].dropna().iloc[-750:]
        if g is not None:
            z = standardised_residuals(r_e.values, g)
            v, s_ = fhs_var_es(z, g.next_sigma, 0.95)
        else:
            v, s_ = np.nan, np.nan
        try:
            ev = evt_var_es(-r_e.values, 0.99)
        except Exception:  # noqa: BLE001
            ev = {"var": np.nan, "es": np.nan}
        pos_val = abs(w[e]) * aum
        rows[e] = {"sigma_garch": sig_d[e], "var95_fhs": v, "es95_fhs": s_, "var99_evt": ev["var"], "es99_evt": ev["es"],
                   "days_to_liquidate": float(days_to_liquidate(pos_val, adv[e])),
                   "lvar": lvar(v * pos_val if np.isfinite(v) else 0.0, pos_val, 0.001, adv[e], sig_d[e])}
    risk_df = pd.DataFrame(rows).T
    fs = pd.DataFrame({e: factor_specific_split(pd.Series({e: 1.0}), B, F_daily, d_spec ** 2) for e in ids}).T
    risk_df["factor_risk_pct"] = fs["factor_risk_pct"].astype(float)
    risk_df["idio_risk_pct"] = fs["idio_risk_pct"].astype(float)
    st = historical_stress(B).reindex(ids)
    risk_df["max_position"] = max_position(pd.Series(kelly_cap(alpha12, sig_12m.reindex(ids)), index=ids),
                                           adv, aum, S["mandate"]["max_position"])
    risk_table = assemble_risk([risk_df, st], last_date)
    risk_table.to_csv(out / "risk.csv", index=False)
    pd.DataFrame({"weight": w, "alpha_1m": alpha1, "mcr": eul["mcr"], "cr_pct": eul["cr_pct"]}).to_csv(out / "portfolio.csv")
    stress.to_csv(out / "stress_tests.csv")
    rev.to_csv(out / "reverse_stress.csv")
    store.write("outputs", "risk", risk_table)

    # Costed backtest of the model's long–short portfolio (Spec §8.6, §9.4).
    pred_wide = oos.pivot_table(index="date", columns="entity_id", values="pred")
    pred_wide.index = pd.DatetimeIndex(pred_wide.index).to_period("M").to_timestamp(how="end").normalize()
    bt_model = quantile_portfolios(pred_wide, fwd1, n_q=5, cost_per_turnover=0.002)
    ls = bt_model["long_short_net"]
    dsr_model = deflated_sharpe_ratio(ls.mean() / ls.std(), sr_var, len(sig) + 1, len(ls))
    gate = production_gate(t_nw=t1, bh_survives=bool(t1 > 3), net_mean=float(ls.mean()), dsr=dsr_model,
                           fm_t=t1, split_ics=[ic_full.iloc[: len(ic_full) // 2].mean(), ic_full.iloc[len(ic_full) // 2:].mean()])
    results.update(model_sharpe_net=metrics.sharpe(ls), model_dsr=dsr_model, model_gate=gate)
    _say(f"  model long–short: net Sharpe {metrics.sharpe(ls):.2f}, max DD {metrics.max_drawdown(ls):.1%}, DSR {dsr_model:.2f}", verbose)
    if verbose:
        print("  Production gate for the combined model:\n" + gate.report())

    # ===== Phase 14: monitoring (Section 9.5) ====================================================
    _say("Phase 14 Monitoring: rolling IC, CUSUM, PSI (Section 9.5)", verbose)
    cs = cusum(ic_full, mu0=max(ic1, 0.0))
    train_cut = dates[len(dates) // 2]
    psis = {c: psi(snap.loc[snap["date"] <= train_cut, c], snap.loc[snap["date"] > dates[-7], c])
            for c in ["momentum", "value", "quality", "volatility", "size"]}
    mon = pd.DataFrame([{"signal": "combined_model", "date": last_date, "ic_12m": rolling_ic(ic_full).iloc[-1],
                         "cusum": cs["cusum"].iloc[-1], "alarm": bool(cs["alarm"].iloc[-1]), "ece": ece_bull,
                         "psi_max": max(psis.values()),
                         "status": status(rolling_ic(ic_full).iloc[-1], bool(cs["alarm"].iloc[-1]), max(psis.values()),
                                          ece_bull if np.isfinite(ece_bull) else 0.0)}])
    mon.to_csv(out / "monitoring.csv", index=False)
    results["monitoring"] = mon
    if verbose:
        print(mon.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    results["runtime_seconds"] = time.time() - t0
    results["out_dir"] = str(out)
    _write_summary(out, results)
    _say(f"Done in {results['runtime_seconds']:.0f}s. Outputs written to {out}", verbose)
    return results


def _write_summary(out: Path, r: dict) -> None:
    """A short Markdown report of the run, saved next to the CSV outputs."""
    sig = r["signal_tests"]
    lines = ["# Project Atlas demo run (synthetic data)", "",
             "Every number below comes from synthetic data with known planted effects; it shows that the",
             "pipeline runs and that the tests behave sensibly, not that any strategy works on real markets.", "",
             "## Single-signal tests (Section 9)", "", sig[["ic_mean", "t_nw", "sharpe_net", "bh_survives", "dsr", "passed_gate"]].round(3).to_markdown()
             if hasattr(sig, "to_markdown") and _has_tabulate() else sig.round(3).to_string(), "",
             "## Forecast engine (Section 7)", "",
             f"- 1-month out-of-sample IC: {r['oos_ic_1m']:.4f} (Newey–West t = {r['oos_t_1m']:.2f})",
             f"- H1 (full model vs baseline): ΔIC t = {r['H1']['t_nw']:.2f}, pass = {r['H1']['h1_pass']}",
             f"- Relative-scenario Brier score: {r['brier_stat_rel']:.4f} (climatology {r['brier_climatology']:.4f})",
             f"- Regime accuracy of the HMM against the true synthetic regime: {r['regime_accuracy']:.0%}", "",
             "## Risk (Section 8)", "",
             f"- Gross / net exposure: {r['gross']:.2f} / {r['net']:+.3f}; predicted volatility {r['port_vol_ann']:.1%} a year",
             f"- 1-day VaR95 (FHS) {r['var95']:.2%}, ES95 {r['es95']:.2%}, Monte Carlo VaR95 {r['mc_var95']:.2%}",
             f"- Kupiec LR {r['kupiec']['lr']:.2f} (reject = {r['kupiec']['reject']}), bias statistic {r['bias']['bias']:.2f}",
             f"- Model long–short net Sharpe {r['model_sharpe_net']:.2f}, deflated Sharpe {r['model_dsr']:.2f}",
             f"- Production gate passed: {r['model_gate'].passed}", "",
             f"Runtime: {r['runtime_seconds']:.0f} seconds."]
    (out / "demo_summary.md").write_text("\n".join(lines))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
