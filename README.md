# Project Atlas

**A Multi-Layer Quantitative Equity Forecasting and Risk System**

Project Atlas is a research codebase for a market-neutral long/short equity strategy. It turns
fundamentals, macro data, news, flows, sentiment and sector structure into calibrated bull / base /
bear probabilities for every company, sizes positions under explicit risk limits, and tests every
component statistically before it is allowed to influence a decision.

The code implements the design set two documents, included in `docs/`:

| Document | Referred to in the code as |
|---|---|
| `docs/Project_Atlas_Specification.docx`, the full system specification | **Spec §x.y** (for example *Spec §7.6*) |
| `docs/Project_Atlas_Technical_Reference.docx`, the APIs, formulas and algorithms | **Ref x.y** (for example *Ref 3.9*) |

Every module opens with a docstring that names the section it implements and writes out the
formulas used, and the notes inside the code explain each step in the order given in the
specification. A reader can open any section of the Word document and find the matching code.

> **Status and disclaimer.** This is research and educational code. It runs end to end on synthetic
> data and has a test suite, but it has not been validated on real market data and it is not
> investment advice. Nothing here should be used to trade real money.

---

## How the code maps to the specification

| Spec section | Package | What it contains |
|---|---|---|
| A. Mandate, edge and scenarios | `config/settings.yaml`, `atlas.research.hypotheses` | Mandate limits, scenario thresholds, pre-registered hypotheses H1–H3 |
| 0. Shared data infrastructure | `atlas.data`, `atlas.ingest` | Storage layers, security master, corporate actions, point-in-time tables, universe, cleaning, rank-Gaussian standardisation; clients for WRDS, SEC EDGAR, FRED, GDELT, congress.gov, Polymarket, Kalshi, FINRA; synthetic market generator |
| 1. Fundamentals | `atlas.fundamentals` | Statements and ratios, Piotroski / Altman / Beneish scores, AR(1) driver projection, DCF and residual income, DCF Monte Carlo, reverse DCF, SUE, options (Black–Scholes, IV, SVI, risk-neutral density) |
| 2. Macro | `atlas.macro` | Stationarity transforms and ADF, surprises and surprise index, Kalman-filter nowcast (dynamic factor model), VAR with impulse responses, Gaussian HMM regimes, macro betas, implied equity risk premium |
| 3. News, geopolitics, elections, laws | `atlas.news` | MinHash deduplication, session alignment, entity linking, topics, FinBERT sentiment (with lexicon fallback), novelty, decaying indexes, uncertainty index, event detection, event studies with the BMP test, shrinkage, election and bill-passage models |
| 4. Flows and positioning | `atlas.flows` | ETF flow pressure, short interest and squeeze score, 13F crowding, dealer gamma, index inclusion, square-root impact, insider score |
| 5. Behaviour and sentiment | `atlas.behaviour` | Momentum (raw, volatility-scaled, residual), reversal, 52-week high, revisions, attention, market sentiment index |
| 6. Sector structure | `atlas.sectors` | Text-based peers, factor model (constrained WLS, EWMA covariance, specific risk), shrunk betas, sector aggregates, supply-chain network propagation |
| 7. Forecasting engine | `atlas.forecast` | Training panel, purged walk-forward cross-validation, ridge and gradient-boosted trees, quantile models, explanations, Grinold conversion, regime-weighted expected factor returns, Student-t scenarios, Brier blending, isotonic calibration, joint Monte Carlo |
| 8. Risk management | `atlas.risk` | GJR-GARCH, VaR/ES (filtered historical simulation, EVT, Monte Carlo), Euler attribution, liquidity-adjusted VaR, stress tests, Kelly bound, cost model with borrow fees, optimiser, drawdown rule, capacity, Kupiec / Christoffersen tests, execution and P&L attribution |
| 9. Hypothesis testing | `atlas.research` | Newey–West, Fama–MacBeth, Benjamini–Hochberg, deflated Sharpe ratio, stationary bootstrap, research registry, signal metrics, quantile backtests, production gate, CUSUM and PSI monitoring, baseline and ablation |
| 10. Integration | `atlas.pipeline` | Daily schedule as a dependency graph, end-to-end demo, command-line interface |

```
project-atlas/
├── config/settings.yaml        every tunable number of the specification, with its section
├── docs/                       the specification and the technical reference (Word)
├── src/atlas/                  the package (one sub-package per section, see the table above)
├── scripts/                    run_demo.py, fetch_free_data.py
├── tests/                      pytest suite, one file per layer
├── notebooks/                  exploration only
├── data/                       local storage layers (ignored by git)
└── ci/tests.yml                continuous integration (moved to .github/workflows/ before the first push)
```

---

## Getting started

Python 3.10 or newer is required. On a Mac (including M1/M2 machines):

```bash
cd project-atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core: numpy, pandas, scipy, scikit-learn, pyyaml, requests
pytest -q                        # about 30–60 seconds
python scripts/run_demo.py       # full pipeline on synthetic data, about 1–2 minutes
```

The demo prints one block per build phase and writes its results to `data/outputs/`:

| File | Content |
|---|---|
| `forecasts.csv` | The `[forecasts]` table of Spec §7.9: score, α, μ, σ, quantiles, raw / calibrated / relative scenario probabilities, top five drivers |
| `risk.csv` | The `[risk]` table of Spec §8.8: GARCH volatility, VaR/ES, factor share of risk, liquidity, stress losses, maximum position |
| `portfolio.csv` | Optimised weights with marginal and component risk |
| `signal_tests.csv`, `research_registry.csv` | Spec §9.1–§9.4 test results and the registry of every experiment |
| `monitoring.csv` | The monitoring table of Spec §9.6 |
| `demo_summary.md` | A short report of the run |

The synthetic market (`atlas.ingest.synthetic`) has known, planted effects: a momentum effect, a
quality effect, volatility clustering, delistings, splits, dividends, filing lags and macro regimes.
It also contains a pure-noise signal. A correct pipeline should find the planted effects, reject the
noise signal and recover the true market factor and regimes. The demo reports all of these, and some
tests check them automatically. The numbers say nothing about real markets.

### Optional libraries

The specification names several specialist libraries. None of them is needed to run the code: each
module imports them only when available and otherwise uses its own NumPy/SciPy implementation of the
same method (see `atlas/_optional.py`).

```bash
pip install -e ".[stats]"     # statsmodels, linearmodels, arch, hmmlearn
pip install -e ".[ml]"        # lightgbm, shap
pip install -e ".[optim]"     # cvxpy
pip install -e ".[storage]"   # pyarrow, duckdb
pip install -e ".[nlp]"       # transformers (FinBERT), sentence-transformers, datasketch
pip install -e ".[data]"      # wrds, fredapi, edgartools, ib_async
pip install -e ".[all]"
```

### Command line

```bash
atlas demo [--stocks 200] [--days 1260] [--seed 7] [--fast]
atlas schedule                 # the daily job order of Spec §10.2
atlas register-hypotheses      # pre-register H1–H3 in the research registry
```

---

## Moving to real data

The order follows the Minimum Viable Data Plan (Spec §A.4) and the Build Plan (Spec §11):

1. Copy `.env.example` to `.env` and fill in `SEC_USER_AGENT` (name and email) and `FRED_API_KEY`.
2. `python scripts/fetch_free_data.py --tickers AAPL MSFT --fred` downloads SEC XBRL facts in
   point-in-time form and FRED/ALFRED vintages into `data/raw/`.
3. With a university WRDS account, `atlas.ingest.wrds_client.WRDSClient` pulls CRSP prices and
   delistings, Compustat, I/B/E/S and OptionMetrics.
4. The same functions used in `atlas.pipeline.demo` are then run on the real tables, one layer at a
   time, and each layer is kept only if the ablation study (Spec §9.7) shows that it adds value.

### Data licensing and secrets

* **Licensed data is never committed.** CRSP, Compustat, I/B/E/S and OptionMetrics data may not be
  redistributed. The whole `data/` directory (except its README) is in `.gitignore`.
* **API keys stay in `.env`**, which is also ignored by git. On GitHub, keys for automated jobs belong
  in repository secrets, never in code.
* The broker wrapper (`atlas.risk.execution.IBBroker`) refuses to connect to a live trading port
  unless this is enabled explicitly, and checks every order against pre-trade limits.

---

## Running it in the cloud

The daily pipeline can run on a small cloud machine or a scheduled GitHub Actions workflow instead of
a laptop. Data must then live in private storage, not in the repository, and secrets must be
supplied through the platform's secret store. The test workflow in `ci/tests.yml` runs the full test
suite and the synthetic demo on every push once it is moved to `.github/workflows/tests.yml`.

---

## Design principles (Spec §11)

* **Point-in-time everywhere.** Every feature is computed only from information known at the time
  (`atlas.data.point_in_time`), and a test checks that queries as of a date never see later data.
* **Test before trusting.** Every signal passes the production gate of Spec §9.4, and failures are
  recorded in the registry.
* **Simple first.** Ridge before boosted trees, OLS betas before Kalman betas.
* **Costs are part of the model.** Spreads, square-root impact and borrow fees enter the optimiser.
* **Everything is recorded.** Settings live in one file, experiments in the registry, code in git.

## Collaboration

Project Atlas is an open, ongoing project, and collaborators are welcome. Students and practitioners
interested in quantitative finance, statistics, machine learning or software engineering are invited
to help improve it, whether by reviewing the methodology, fixing bugs, adding tests or building one of
the unfinished layers.

Areas where help is especially welcome:

* **Real-data ingestion:** connecting the news (Spec §3), flows (Spec §4) and options (Spec §1.6)
  layers to live free data sources.
* **Validation:** checking formulas against the specification and looking for look-ahead bias.
* **Testing on real data:** running the research harness (Spec §9) on CRSP/Compustat or free data.
* **Performance:** speeding up the factor model and Monte Carlo simulations.
* **Front end:** a Streamlit app with company, portfolio and research pages (Spec §10.3).

**How to get involved**

1. Open an [Issue](https://github.com/cnearchou/project-atlas/issues) to introduce yourself, suggest
   an improvement or ask which task to pick up.
2. Fork the repository, create a branch, and open a pull request. Each change should cite the
   specification section it implements, include a test, and pass `pytest`.
3. Licensed data and API keys are never committed (see *Data licensing and secrets* above).

Feedback on the design itself is just as valuable as code, so questions and critiques are welcome too.

## Licence

MIT, see `LICENSE`. The licence covers the code only; it gives no rights to any vendor's data.
