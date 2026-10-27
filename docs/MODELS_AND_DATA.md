# Models and data: what trains what, and how it all fits together

Status as of 2026-10-27 (commit `d33f693` and later). Every model here runs on this computer from free
data; the only paid or external calls are the optional AI agents (free Gemini / OpenRouter models first).
**No model in this app is validated for trading.** Each one carries its own status, shown on the pages.

## 1. The data

| Data | Source | Stored in | Known-at-the-time rule |
|---|---|---|---|
| Daily prices: adjusted close, high, low, volume (stocks, ETFs, sector funds, SPY, BIL, SGOV, USO, UUP, HYG, IEF) | Yahoo Finance, downloaded once, then only new sessions | `~/.tradingagents/cache/optimizer/market_data.sqlite3` (SQLite) | only closes up to the decision date; adjusted closes are total-return |
| Market series: VIX, VIX3M, 13-week bill (^IRX), 10-year yield (^TNX) | Yahoo Finance | same | same |
| Dividends and splits | Yahoo Finance | same | used to rebuild the prices stocks actually traded at |
| Company fundamentals (revenue, margins, cash flow, assets, shares...) | SEC EDGAR company facts | cache folder | each figure from the day after it was **filed** |
| Economy: CPI, unemployment, payrolls, financial conditions, yield curve, credit spread, fed funds | FRED and ALFRED (vintages) | cache `macro/` | the vintage published **before** each date: no later revisions |
| S&P 500 membership history, with removed stocks | github.com/fja05680/sp500 | cache | members on each date; delisted stocks kept until their last price (survivorship **partial**) |
| GICS sector of each S&P 500 member | Wikipedia S&P 500 list | cache | today's classification |
| News headlines | Yahoo Finance | `portfolio.sqlite3` (`stock_news`) | first-seen time recorded |
| Company news: every 8-K since 2006 (132,000+ for the S&P 500: results, executive changes, deals, announcements) and its press-release text | SEC EDGAR | `cache/optimizer/sec_news.sqlite3` (`stocks/news_archive.py`) | SEC's acceptance time; after 16:00 New York it counts from the next session |
| Wall Street analysts: dated rating changes and price targets (from about 2012), each report's EPS estimate, actual and surprise (from about 2002) | Yahoo Finance | price store snapshots; rebuilt per date by `stocks/consensus.py` | an action from the session after its day; targets in today's shares; Yahoo's lists as of today (firms that stopped covering may be missing) |
| Your extra companies | Settings > Companies the model scores | `portfolio.sqlite3` (`stock_watchlist`) | scored like members |
| Your account: holdings, fills, budget | Settings, Portfolio tab | `portfolio.sqlite3` (append-only ledger) | |
| **Training sets** of the decision models | built from the above | `~/.tradingagents/cache/optimizer/training.duckdb` (DuckDB) | one copy per distinct set, with a fingerprint |
| Every model prediction and its outcome | the daily decision | `portfolio.sqlite3` (`model_predictions`, `model_outcomes`), `results/model_history/model_predictions.csv` | saved before the outcome is known |

Excluded on purpose: SOXL and TQQQ (`my_universe.json` → `"exclude"`). No paid data feeds.

## 2. The models

### A. The daily stock decision (the "internal model": best buy, runners-up, sells)

Code: `tradingagents/stocks/daily.py` (`fit_bundle`, `score_today`, `entries`, `review_lot`).
Fitted weekly (or when the code or data changes), on **S&P 500 members on decision dates every 5 sessions
since 2006**: about 380,000 rows of 54 features (prices and momentum, volatility, trend, volume, relative
strength against SPY and the sector, market series, SEC fundamentals as filed, economy as published).

| Part | Algorithm | Trained on (label) | Output |
|---|---|---|---|
| Ranker | XGBoost learning-to-rank (pairwise), groups = dates | each stock's 20-session return over cash from the next close, graded into fifths per date | a score, turned into the **rank** among today's stocks |
| Expected-net map | 10 bins, shrunk toward the average (k = 200) | out-of-fold ranker percentiles → the barrier policy's net trade P&L | expected net after costs (and its spread) |
| Trade quality | logistic regression + sigmoid calibration | out-of-fold rows: did the policy's trade make money after costs? | P(profit after costs) |
| Position risk | XGBoost classifier | did the position end down ≥ 1.5 × its own 20-session volatility? | P(large loss) |

"Out-of-fold" = yearly cross-fitting with purging: a row is scored by a ranker that never saw its year,
and the second-stage parts learn only from earlier folds.

**How they combine into a decision** (`daily.entries`):
1. Order today's members and your extra companies by ranker score (percentile among members; ties share a rank
   and are listed alphabetically).
2. A candidate passes the **model gates** when P(large loss) ≤ 15% *and the risk model tells stocks apart*
   (otherwise "risk gate not assessed"), expected net ≥ +0.5%, P(profit) ≥ 50%. The +0.5% and 50% thresholds
   were not selected on development data in any recorded study.
3. Then the **account gates**: not already held, at most 5 names, at most 25% each, within the trading budget,
   at most 10% of the whole IRA in one stock.
4. The first that passes is the **best buy**; the next four are the runners-up. Entry = the next session's close
   (no limit; `positions.execution_spec`);
   target and stop = ±1 twenty-session volatility (`stop = price·e^(−σ√20)`, `target = price·e^(+σ√20)`),
   sold at the next close after a close beyond either, or after 20 sessions.
5. Held stocks: EXIT at the stop, target or deadline; REDUCE above the 25% cap; else HOLD.
6. **Validation gate**: real ENTER only after a decision study beats SPY after costs *and* a paper (shadow)
   account has 20+ trades ahead of SPY. Until then: `WAIT · model BUY, not validated` (today: not validated).

Verdict labels (`stocks/verdict.py`): BUY = top fifth by rank + the three model gates; AVOID = bottom fifth or a
negative expected net; SELL (held) = an exit rule, or bottom fifth with a negative expected net.

Known weakness (measured): P(large loss) is 7.1% for every stock and P(profit) 53–54%: those parts have no
stock-specific view; the rank does the work. Decision study (2024–26): +5.0% vs SPY +43% and cash +9.9%.

**Saving and retraining** (`stocks/training_store.py`): each training table is fingerprinted; the same data
is never fitted twice (the earlier fit is reused), each set is saved in DuckDB with its bundle, and a model
refuses to score any date on or before the last date it trained on.

**Learning from results** (`stocks/feedback.py`): every daily prediction (≈500 stocks) is saved, scored 20
sessions later, exported to CSV, and once 300 have matured over 10 days, used to recalibrate P(profit) and
expected net (isotonic / a shrunk line).

### B. Stock signal models (ticker page signals, screener, entry-plan dip levels)

Code: `tradingagents/stocks/pipeline.py`, `stocks/models.py`, `stocks/signals.py`. Same panel and data as A,
trained on resolved rows, run on request ("Compute signals for all").

| Model | Algorithm | Label |
|---|---|---|
| Expected return | Elastic Net and XGBoost quantile (5/50/95%) | 20-session log return over cash |
| Relative | XGBoost regression | return over SPY |
| Trend | XGBoost classifier | up / down / sideways / unclear |
| Events | XGBoost classifiers | trades above the 20-day high / below the 20-day low (intraday), large loss at day 20, deep drawdown |
| Severity | XGBoost quantile | worst drawdown from today's close (5th, 50th percentile) |

Combined only as **context**: warnings when a probability is ≥ 1.2× (or 1.5×) its base rate *and* in today's
top fifth; the median and deep dip give the "patient" limit prices. Walk-forward status: untested for the
current version.

### C. Trade outcome models (policy layer, "insufficient evidence")

Code: `stocks/policies.py`, `stocks/outcome_models.py`, `stocks/decision_layer.py`, `stocks/policy_study.py`.
Data: S&P 500 members every 10 sessions since 2006, same features; labels from triple-barrier simulations of
three fixed policies (buy at next close; 5-session limit at 0.5 σ below, filled only when the low goes 0.1%
through; hold from now), with gaps, dividends, costs, delistings and no-fills.

| Model | Algorithm | Output |
|---|---|---|
| Outcome probabilities | LightGBM multiclass (and a linear baseline), isotonic calibration on a separate later slice | P(target first), P(stop first), P(expiry), P(no fill) |
| Expected excess | LightGBM regression | expected return over cash |
| Bad case | LightGBM quantile 10% | 10th-percentile return |

Combined by the decision layer: the action with the highest expected excess that clears a no-trade margin
(chosen on 2014–24 data) and a −15% bad-case budget; held stocks judged from now on. Status: **not validated**
(2025–26 holdout: probabilities worse than base rates, −0.94% vs cash per decision), so it **abstains** on
every stock and gives the AI agents nothing.
Its would-be buys (its own choice while it abstains, at most five a day) are still recorded in the Track
record tab and judged like the internal model's picks; a limit that never filled is not judged.

### D. Sector rotation (Trade page)

Code: `research/rotation.py`, `research/sector_model.py`. Data: the 11 SPDR sector funds, SPY and six market
drivers (10-year yield, yield curve, oil, dollar, junk bonds vs Treasuries, VIX), daily since 2007.

| Part | Algorithm | What it gives |
|---|---|---|
| Stages and alerts | relative-rotation arithmetic (63-session relative strength, 10-session momentum) | Improving / Leading / Weakening / Lagging; early "rotating in/out" alerts |
| Causes | ridge regression of each sector's daily return over SPY on the six drivers (252 sessions) | which driver moved each sector, with how much the drivers explain (R²) |
| Next leader | **LightGBM LambdaRank** (+ LightGBM regression for size), sector as a category, decision dates every 5 sessions from 2008 | the 11 sectors ranked for the next 20 sessions |
| Probable winner | the internal stock model's (A) top-ranked stock in each GICS sector | one stock per sector |

LambdaRank status: **not validated** (2025–26 holdout rank correlation +0.08, interval −0.04 to +0.18; it beat
plain momentum there but showed no skill in 2010–24).

### E. ETF forecasts (Forecasts tab)

Code: `research/engine.py`, `research/models.py`, `research/forecast.py`. Data: the ETF list (sectors, broad
markets, bonds, gold...; SOXL/TQQQ excluded) with market features. Models: Elastic Net, momentum, XGBoost
(regression, ranking, quantile), Chronos-2 (a pretrained time-series model, run locally), and "stacked"
versions that add Chronos-2's forecast as inputs; each with a walk-forward record. Combined view: the models'
forecasts side by side and averaged. Used as context, and as the "ETF forecast agrees / disagrees" line on the
rotation alerts.

### F. The AI agents (TradingAgents, TauricResearch v0.7.0)

LLM agents, not trained here: four analysts (market, sentiment, news, fundamentals) fetch data through tools
(the same price store, SEC, FRED, Yahoo news, StockTwits/Reddit), then a bull/bear debate, a research manager,
a trader, a risk debate (aggressive/conservative/neutral) and a portfolio manager write a rating.
Models, tried in order until one answers (each only when its key is set and the provider is in the
spending policy's fallbacks): **Gemini 3.8 Flash** → the free **OpenRouter** models (Nemotron 3 Ultra,
Gemma 4 31B) → **Gemini 3.5 Flash-Lite**. OpenRouter comes before the other Gemini model because
that one shares the Google key's quota and outages. The same order applies to the market brief, the stock
ratings and the moat notes (`optimizer.market._models`); roles you move to another provider in Settings >
Agent models keep their seats on the first attempt.
Combined: **context only** (the internal model decides). They receive model evidence only if it is validated
(none is today). The "Copy the prompt" box reproduces the same agents' API calls for ChatGPT.

### G. Every model side by side (Models tab, ticker page)

Code: `stocks/model_zoo.py`, `stocks/seq_model.py` (Mamba); study: `experiments/model_zoo_study.py`,
`experiments/mamba_study.py`. Each model is fitted on all history to the last close and ranks today's members
(the Models tab: rank 100 = its top pick, plus a 20-session forecast for models that predict a return). None of
them decides anything. Every run is saved (`model_zoo_runs`).

Tested on identical purged yearly folds (rank IC with the barrier trade's net P&L; 2025 on was looked at before,
so it is development evidence; the forward record is the test):

| Model | 2012-24 IC | 2025 on IC | 2025 on top fifth - all |
|---|---|---|---|
| XGBoost ranker, 54 features (current) | +0.022 | +0.001 | -0.06% |
| XGBoost ranker, all data (+news, analysts, competitors) | +0.032 | -0.006 | -0.46% |
| LightGBM LambdaRank, all data | +0.020 | +0.021 | +0.69% |
| LightGBM regression | +0.030 | -0.003 | +0.06% |
| Ridge, all stocks | +0.015 | -0.018 | +0.08% |
| Ridge, one per stock (market, sector, competitors, economy, analysts, news) | +0.015 | +0.018 | +0.46% |
| Mamba (64-day sequences + 54 features) | +0.018 | +0.006 | +0.18% |
| Chronos-2, zero-shot (2023-24 / 2025 on) | -0.006 | +0.019 | +0.11% |

Data added to the current model: SEC news +0.027 / +0.031; analysts +0.021 / +0.001; competitors +0.026 / +0.015.
No 2025-on interval excludes zero: nothing is shown better than the current model yet. Forecasts in return units
(regressions, Chronos-2) are not calibrated: the per-stock ridge in particular gives large numbers (+12% for NVDA
on 2026-10-26) and should be read as an order, not a size.

## 3. How it all fits together

```
Yahoo prices, SEC filings, FRED vintages, S&P 500 membership
        │
        ├─► Panel: 54 features per stock per date ──► training.duckdb (fingerprinted sets)
        │       │
        │       ├─► A. Daily decision models ──► rank, expected net, P(profit), P(large loss)
        │       │        └─► gates + account rules + validation gate ──► Best buy, runners-up, sells (Trade tab)
        │       │                 └─► every prediction saved ──► scored after 20 sessions ──► recalibration, Track record
        │       ├─► B. Signal models ──► warnings, dip limits (ticker page, screener) — context
        │       └─► C. Trade outcome models ──► decision layer ──► abstains until validated
        │
        ├─► Sector funds + drivers ──► D. stages, alerts, causes; LambdaRank next leader
        │                                   └─► + A's top stock per sector = probable winner
        ├─► ETF list ──► E. ETF forecasts (Forecasts tab; agree/disagree on rotation alerts)
        └─► AI agents (F) ──► rating and full report (context; Track record judges it too)
```

Order of authority: the **validation gate** decides whether anything is a real buy (today: nothing is); the
internal model (A) decides the order and the levels; everything else (B, C, D, E, F) is evidence shown
beside it with its own status.

## 4. Reading the outputs you see (Trade page)

What each number means and how far to trust it. The worked example is the decision of 2026-10-26. The audit
behind the corrections is in `docs/MODEL_AUDIT_2026-10.md`.

| Shown | Where it comes from | What to make of it |
|---|---|---|
| **Rank** | the ranker's score among today's index members; tied scores share a rank | The ranker kept only its 10-tree minimum, so its scores move in steps and stocks tie. On 2026-10-26, 413 of 499 members were in a tie; GEN and DVA had bitwise-identical scores. Tied stocks are listed alphabetically: that order is not a preference. |
| **Why a stock ranks high** | TreeSHAP contributions of the ranker (baseline plus contributions equals the score) | GEN on 2026-10-26: liabilities/assets +0.0080 of a 0.0131 score. Its other positive pushes total +0.0074 and its negatives −0.0023. Across members, that one feature carried 35% of the attribution. It is a learned factor tilt, not a view on the company, and not causal. |
| **Expected net** | the out-of-fold map from the members-only percentile to the barrier trade's net P&L, in 10 bins | Every stock in a bin shares the number. The top bin (+0.54%) is *below* the 0.7–0.8 bin (+0.65%): the map does not rise with the score. It is a historical score-band average, not a forecast for the stock. |
| **vs SPY** | the same map, net minus SPY's total return over the identical entry/exit dates | Negative in every bin (−0.32% to −0.77%). A positive absolute and a negative active figure can coexist. |
| **P(profit)** | trade-quality model (logistic + sigmoid) | 53–54% for everyone, and worse than the base rate out of sample. Shown, but it no longer breaks ties. |
| **P(large loss)** | position-risk model (loss ≥ 1.5 × the stock's own volatility) | One value (7.1%, the training base rate) for every stock. The gate now says **"risk gate not assessed"** and blocks, rather than letting everyone pass. The `quant-upgrades` branch's 8%-fall model varies by stock but is not validated. |
| **Buy** | the tested policy (`positions.execution_spec`) | Buy at the next session's close, whatever it is, with no limit price. The last close is a dated reference, not an order price. |
| **Stop / target** | ± one 20-session EWMA volatility *from your fill* | Shown as percentages from the fill. Dollar levels assume a fill at the last close and are illustrations only. A close beyond either level triggers a sale at the next close (so a gap is sold below the stop); otherwise the stock is sold after 20 sessions. |
| **Untested: patient dip** | the signal model's median drawdown (a path minimum) | A different, never-tested policy. "About an even chance" is of the price *trading* there, not of a fill, and returns after such fills are unknown. |
| **No stock passes** | the gates | Each top-ranked stock is listed with what blocks it. Today that is the uninformative risk gate. |
| **WAIT · not validated** | the validation gate | ENTER needs a study that beats SPY after costs *and* 20+ forward paper trades ahead of SPY. |
| **Agents' rating** | AI agents (section F) | Context only; it never changes the verdict. Every run, failed ones included, is recorded in `agent_runs`. |

## 5. Where each status comes from

| Model | Evidence | Today |
|---|---|---|
| A. Daily decision | `research_trials` `decision_study:h20` + the shadow account | not validated (study untested for this code; earlier study failed) |
| B. Signals | `stock_study:h20` | untested for the current version |
| C. Trade outcome | `policy_study:h20` (trial 33) | not validated |
| D. Sector LambdaRank | `sector_model:h20` | not validated |
| E. ETF forecasts | walk-forward trials per model | some rank ETFs better than chance (IC ≈ 0.1); none beat buy-and-hold after costs |
| F. AI agents | Track record tab (prospective) | first verdicts after 20 sessions |

Reproduce: `.venv/Scripts/python.exe -m experiments.policy_study`, the "Run the sector model" button (or
`tradingagents.research.sector_model.run`), "Make today's decision" (fit models if needed).
