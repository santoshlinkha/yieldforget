# Portfolio app: system reference

**What this is.** A complete technical description of the personal portfolio
application built on top of the open-source TradingAgents project, as of
2026-10-24 (branch `portfolio-app`). It covers what the app does, every model, what
data is used and saved, how forecasts are made and evaluated, every metric, the
measured results, and the known limits. It is written to be reviewed: by the owner,
and by other AI systems asked for critique (see `REVIEW_PROMPT.md`).

**The account.** One Robinhood **Roth IRA** (a cash account: purchases need settled
cash; sales settle T+1; no margin, no short selling; options at most level 2 if
approved). About **$5,844**, kept as two bookkeeping sleeves inside the one IRA:
**Investing** (long-term, about $4,844) and **Trading** ($1,000 assigned initially,
then changing only with its own profit and loss and explicit transfers). The app
plans and records; **it never places orders**. The owner trades by hand.

**Ground rules the whole system follows.**
1. No look-ahead: every feature uses only information available at its date.
2. Every experiment is recorded, whatever it found, so multiple testing is visible.
3. Paid AI is opt-in per action, cached, budgeted and logged; numbers never come from
   a language model, only from fitted models or arithmetic.
4. Records are append-only: corrections are new records pointing at old ones.
5. Data is downloaded once and then only what is new.

---

## 1. Architecture

```
TradingAgents (upstream)   multi-agent LLM analysts (market, news, fundamentals, sentiment)
        |
tradingagents/dataflows    vendors: Yahoo Finance (prices, profiles, statements, analysts, news),
        |                  SEC EDGAR (company facts, filings), FRED/ALFRED (economic series),
        |                  local market store (SQLite + pickles): incremental cache
        |
tradingagents/optimizer    ETF/stock portfolio optimizer, risk levels, plan, daily orders (T+1),
        |                  auto-invest, paper tracker, backtest, market overview
tradingagents/research     ETF return forecasting: features, models, walk-forward, risk models,
        |                  neural (LSTM) and pretrained (Chronos-2) challengers, ticker research
        |                  (moat, quality checklist, dividends/splits), compute (CPU/GPU)
tradingagents/stocks       stock-only research: point-in-time fundamentals and macro, event targets,
        |                  batch forecasts, news sentiment, recommendations, outcomes
tradingagents/ledger       SQLite books: sleeves, events, settlement, recommendations, decisions,
        |                  AI gateway (budget/cache), forecast and stock-run records, migrations
tradingagents/webapp       local web app (127.0.0.1:8765), stdlib HTTP server + static pages
cli/main.py                command line: web, suggest, track, book ..., stocks ...
experiments/               studies that produce the evidence (walk-forward, ablations, risk)
```

Hardware: Windows 11 laptop, i5-11400H, 16 GB RAM, NVIDIA RTX 3050 Ti (4 GB). Python
3.13, XGBoost 3.4.1 (CUDA build), scikit-learn 1.9.1, PyTorch 2.11 (CUDA 12.8), arch 8.0,
transformers 5.20. Everything numerical runs locally.

---

## 2. Data sources and what is saved

| Source | What | How it is cached | Point-in-time handling |
|---|---|---|---|
| Yahoo Finance (yfinance 1.7) | Daily OHLCV (auto-adjusted for splits and dividends), quote profiles, financial statements (4 annual + quarterly), analyst recommendations, price targets, EPS/revenue estimates, upgrades/downgrades, earnings dates, dividends, splits, shares outstanding, news | Local market store: bars as one pickle per ticker, only new sessions fetched; snapshots in SQLite with TTLs (news 3 h; statements, analyst histories, actions 7 days; profiles daily) | Prices are as-of. Statements and analyst lists are **today's** versions (restatements and deletions invisible), so they are used only for display, not for model training, except where noted |
| SEC EDGAR company facts (XBRL) | Revenue, gross profit, cost of revenue, operating and net income, operating cash flow, capex, assets, liabilities, equity, shares outstanding, with **filing dates** | Compact per-stock JSON (`sec_pit2/`), refetched weekly; rate-limited (8/s) with retries | A fact counts from the **day after** its filing date; restatements are new facts with later filing dates and never rewrite earlier history |
| SEC EDGAR filings | Latest 10-K business section (Item 1, incl. Competition) | Saved per accession number, fetched once per filing | Used for the AI moat reading only |
| FRED / ALFRED | CPI, unemployment, payrolls, NFCI (all vintages); 10y-2y spread, Baa-10y spread, fed funds (unrevised) | JSON per series, refreshed weekly | Revised series: each observation's latest vintage published **before** the date, counted from the day after publication. Unrevised: usable the day after the observation |
| S&P 500 member list | Ticker, name, GICS sector, CIK, date added | CSV, refreshed weekly | Today's members only (survivorship), but each member enters training only from its date added |
| News (Yahoo) | Headlines, publisher, link, publication time | `stock_news` table, append-only, with first-seen time | Collected prospectively; no historical news exists in the system |

**Saved records (SQLite `portfolio.sqlite3`, migrations 1-4, append-only by triggers):**
accounts, events (contributions, fills, transfers, adjustments, voids), settlement and
reservations, settings and their history, recommendations (official plans, with code
version, settings and prices used), decisions, orders, AI usage (tokens, latency, cost),
AI result cache, research_trials (every experiment), forecast_runs/forecasts (ETF and
stock forecast runs with model file hashes), stock_runs, stock_forecasts, stock_news,
stock_ideas, stock_idea_decisions, stock_outcomes.

Other files: `paper_tracker.json` (the Day-1 paper record), model artifacts (pickles with
SHA-256 in the run record), `results/` (saved analysis reports, market briefs, moat readings),
`experiments/output/` (study predictions and summaries).

---

## 3. The portfolio optimizer (Investing and Trading sleeves)

**Universe.** About 40 long-lived ETFs (US total market, sectors, international, bonds,
gold, commodities, leveraged), the S&P 500 (cached member list), and disruptive-theme
stocks and small-cap funds: about 566 tickers, plus holdings. Each is scored cheaply
(expected growth above cash per unit of crash risk); the best 60 plus holdings go to the
optimizer.

**Risk dial (1-10).** A stress-loss limit on the whole portfolio:

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Limit | 1% | 2.5% | 5% | 10% | 15% | 20% | 30% | 40% | 50% | none |

**Stress loss**, two models:
- *Worst case* (default): sum of weight x each fund's own worst historical peak-to-trough
  drop (young funds: a long-lived stand-in, or a 50% floor under 10 years of history).
- *Crash scenarios* (optional): each past crash in SPY, TLT, GLD or DBC replayed on the mix;
  absent funds estimated by downside beta.

**Solver.** Maximize expected growth minus a risk penalty, subject to
sum(weight x stress loss) <= limit, per-asset caps (35% of the portfolio; at most 25% of
the loss budget per asset), long only. Augmented Lagrangian with accelerated projected
gradient (FISTA). Optional **growth target**: bisection for the least risk that reaches a
target growth, never past the dial's cap.

**Expected returns.** The standard rule (vol_prior): 0.3 Sharpe x volatility, less half
the variance, shrunk for short histories. Optionally a saved forecast model's output
(relative mode: only differences between assets move money, scaled by the model's
measured within-date calibration, clipped at 2 sigma). The default is the standard rule.

**Orders.** The next session's orders under T+1: buys use settled cash only; sales of the
cash fund (SGOV) fund the following session's buys; trades under 0.5% of the account
(or $1) are skipped. No day trading. **Auto-invest** splits a daily budget into Robinhood
recurring buys (every market day, week, 2 weeks or month; $1 minimum assumed).

**Records.** Official recommendations saved per session (before 3:45 pm New York for that
day, else the next session), with code version, settings and prices; decisions and fills
recorded separately. The **paper tracker** follows two portfolios from Day 1
("followed the plan" vs "kept as is") under the same T+1 rules.

**Measured (walk-forward 2008-2026, ETFs, monthly rebalances).** The worst-case model never
broke its limit in 274 one-year holds; the crash-scenario model broke it by up to 1.5x.
Growth estimates ran 0.3-1.3 points a year above what was realized. The optimizer's Sharpe
(0.52-0.60) is close to 60/40's (0.55): it controls drawdown, it does not add return. Every
setting trailed holding VTI or SPY over 2012-2026.

---

## 4. ETF return forecasting (`tradingagents/research`)

**Target.** Log return **above cash** (BIL) over 5, 20 or 60 trading sessions, from the
decision date's close.

**Features (22), each from data up to the decision date's close, ranked across assets
within each date into [-0.5, 0.5]:** momentum 1/3/6/12-1 months, 1-week reversal,
volatility 1m/3m, downside volatility, beta to SPY (1y), distance from the 52-week high and
the 200-day average, 1-year drawdown, high-low range, dollar volume, its trend, Amihud
illiquidity. Market levels (standardized on training rows): VIX, VIX/VIX3M slope, 10-year
yield's 3-month change, 13-week bill rate, SPY 3-month return and 1-month volatility.

**Models.** zero, train_mean, vol_prior (the incumbent rule), momentum (12-1 rank),
Elastic Net (penalty chosen on the last fifth of training dates), XGBoost (pseudo-Huber),
XGBoost ranking (pairwise within date, mapped to returns by a line on held-back data),
XGBoost quantiles (5/50/95). All wrapped by `Probabilistic`: quantiles from standardized
held-back residuals (or direct quantiles), P(gain), and a bootstrap "model uncertainty"
(spread of the mean over 5 refits on resampled data). Challengers: a small LSTM (about
10k weights, 60 daily returns + features) and **Chronos-2** (Amazon, 120M parameters,
Apache-2.0, zero-shot on the last 1,024 log prices; contamination risk unknown).

**Walk-forward.** Yearly refits from 2012; training rows only where the label **ended**
before the test block (purge, embargo = horizon); shared folds for all assets; all
uncertainty computed over dates (moving-block bootstrap, 12-date blocks); Holm-adjusted
p-values across the models in a trial.

**Results (ETF universe, 2012-2026).**

| Horizon | Model | IC (95% CI) | Within-date calibration | 5-95% coverage |
|---|---|---|---|---|
| 20 | standard rule | 0.067 (0.023, 0.110) | -0.70 | 89% |
| 20 | Elastic Net | 0.111 (0.007, 0.219) | 1.01 | 89% |
| 20 | XGBoost | 0.113 (0.043, 0.192) | -0.28 | 88% |
| 20 | LSTM | 0.114 (0.042, 0.182) | 1.11 | 89% |
| 20 | Chronos-2* | 0.101 (0.047, 0.158) | -0.06 | 87% |
| 60 | standard rule | 0.118 (0.045, 0.193) | -0.89 | 90% |
| 60 | XGBoost | 0.212 (0.111, 0.326) | 0.92 | 88% |
| 60 | XGBoost ranking | 0.220 (0.129, 0.317) | 0.03 | 83% |
| 60 | Chronos-2* | 0.207 (0.127, 0.306) | 0.33 | 90% |
| 5 | XGBoost quantiles | 0.092 (0.065, 0.124) | 0.75 | 87% |

Findings: learned models rank ETFs better than chance and better than the standard rule;
**no model times the market** (date-average forecast vs outcome correlation about 0 or
negative); P(gain) is worse than the base rate; in the portfolio test after 10 bp costs
**no forecast beat the standard rule** (the best roughly tied with 5x the trading).

**Chronos-2 stacking ("+chronos").** Chronos-2's median forecast and 5-95 width as two
extra inputs. Before its release: XGBoost ranking IC 0.220 -> 0.236 at 60 sessions (top
fifth net +1.61% -> +2.15% per period); plain XGBoost slightly worse at 20. After its
release (10-12 dates, the only uncontaminated test) adding it made every 60-session model
slightly worse. It is an option, not a default.

**Live record.** Every saved run is scored once its horizon has passed (IC per run date,
top fifth vs all). Chronos-2's forecasts are recorded at each new close when the app
starts: the clean test of a model that may have seen past prices.

**Combined forecast.** Each model's latest forecasts at a horizon are shown under each
ticker with an evidence-weighted combination: weight = walk-forward IC where Holm p < 0.05;
untested runs, runs that give every ticker the same forecast, and Chronos-2 get weight 0.

**Stock universe for the ticker page (S&P 500, price features only, survivorship-biased).**
60 sessions: XGBoost ranking IC 0.066 (0.038, 0.093), top fifth +1.28%/period net, Holm p <
0.001. 20 sessions: nothing significant; trained on all data, Elastic Net and XGBoost
regressors gave every stock the same forecast (only market-wide splits survived). Stock
pages use a stock model at 60 sessions only.

**Risk models.** Volatility (QLIKE, lower is better): GARCH(1,1)-t best at every horizon
(20 sessions: 0.328 vs EWMA 0.391 vs 10-year sample 0.698). Covariance: the 10-year sample
did best for the optimizer (Ledoit-Wolf, factor and EWMA on 1 year underestimated next
month's variance by 30-40%).

---

## 5. Ticker research (ticker page, any symbol)

- **Wall Street**: consensus (strong buy to strong sell counts over 4 months), price
  targets, EPS and revenue estimates with revisions, upgrades/downgrades history, earnings
  estimates vs reported. From Yahoo; today's versions.
- **Quality checklist**:
  1. *Economic moat (numbers)*: pricing power (gross margin level, floor and stability
     against sector-specific bars), ROIC every year (operating income x (1 - own tax rate) /
     average invested capital, excess cash beyond 2% of revenue deducted), operating
     margin, positive FCF every year, steady revenue growth; banks/insurers get ROE, ROA,
     net margin, book value per share growth. Verdict: strong (4-5 of 5), some (2-3), few.
  2. *Moat sources (AI, paid, optional)*: the latest 10-K's business section read by the
     configured model; switching costs, network effects, cost advantage, intangibles,
     efficient scale, each rated with a **verbatim quote verified against the filing**
     (unverified quotes downgrade the rating to "unclear"); durability and named threats.
  3. *Financial strength*: debt/equity (flag > 2), interest coverage (flag < 3), current
     ratio (flag < 1.0), a high-rate downturn test (operating income -30%, all debt
     refinanced +2 points; flag < 1.5x, thin < 3x).
  4. *Profitability and cash flow*: FCF (TTM, margin, conversion), ROIC vs 15%, gross /
     operating / net margin trends (expanding / stable within 1 point / shrinking).
  5. *Valuation*: forward P/E vs the company's own year-end P/Es (prices restored from
     dividend-adjusted closes), EV/EBITDA, FCF yield, and a **reverse DCF**: the 10-year
     FCF growth today's EV implies at a 9% discount rate and 2.5% terminal growth, judged
     against the revenue record and analysts' next-year estimate.
- **Dividends, splits, share count**: dividend history (split-adjusted), TTM yield, years
  of raises (by typical payment, not calendar totals), 5-year growth, cuts and
  suspensions; splits; share count restated for splits, with dilution (>= +2%/yr) and
  buybacks (<= -2%/yr); funds excluded from share-count analysis.
- **Forecasts**: the standard rule's yearly figure and each horizon's model forecast,
  flagged when the model never saw this kind of asset.
- **News**: chronological headlines.

---

## 6. Stock research module (`tradingagents/stocks`)

### 6.1 Scope and inputs
Stocks only. Tickers from a pasted list, a CSV (ticker/symbol column), the saved
watchlist, and the account's holdings. Each is classified: S&P 500 members are stocks;
others must be Yahoo `quoteType == EQUITY`. Funds, mutual funds, crypto, futures, options,
indexes, currencies and unknown symbols are turned away with the reason; one failed lookup
never stops the rest. Market, sector and economic series remain contextual inputs.

### 6.2 Targets (defined before training; thresholds frozen and versioned)
Decision after the close of date t. sigma_h = daily EWMA volatility at t (lambda 0.94) x sqrt(h).
- **Tradable return**: enter at the **next** close (t+1), exit h sessions later; log, above
  cash (BIL) and separately above SPY. An after-close forecast never trades at that close.
- **Terminal return** (path events): log(close[t+h] / close[t]).
- **Drawdown from entry**: min(low[t+1..t+h]) / close[t] - 1 (distinct from terminal loss).
- **Large loss**: terminal return <= -1.5 sigma_h. **Deep drawdown**: drawdown <= -(exp(1.5 sigma_h) - 1).
- **Breakouts**: barriers = highest high and lowest low of the 20 sessions through t. Up
  breakout = any high in t+1..t+h above the upper barrier; down = any low below the lower.
  Both may occur; a single bar crossing both is recorded (`both_same_bar`) without inventing
  an order. **Failed breakout**: crossed, but closed the period back inside.
- **Trend state** (exactly one of four): *up*: r >= 0.5 sigma_h and path efficiency
  (|r| / sum |daily log returns|) >= 0.30; *down*: mirror; *sideways*: |r| <= 0.35 sigma_h and
  the whole high-low range <= 1.25 sigma_h (so a volatile round trip is not sideways);
  *unclear*: everything else. Events overlap trend states freely.
- **Current trend** (detection, not forecast): price above a rising 50-day average above the
  200-day (+1), mirror (-1), else 0. A feature and a display only.
- Threshold version at the time of writing: `thr-14adb1ddde`.

Observed rates (20 sessions, 2006-2026, S&P 500): up breakout 58%, down 41%, large loss 6%,
deep drawdown 12%; trend mix: unclear 52%, up 20%, sideways 18%, down 10%.

### 6.3 Features (four groups, used for ablation)
- **technical** (26): the ETF price features plus 20- and 50-day average slopes, position in
  the 20-day and 52-week ranges, strength vs SPY and vs the stock's sector ETF over 1 and 3
  months, 1-week volume vs 3 months, current trend.
- **market** (6): as in section 4.
- **fundamental** (14): revenue growth (annual and latest quarter vs a year earlier), gross,
  operating, net and FCF margins, ROA, liabilities/assets, dilution over a year (split-adjusted),
  earnings/sales/FCF/book yields on the market value at the date (raw price, undoing later
  dividends and splits, x shares then reported), filing age. From SEC facts as known at the date.
- **macro** (8): CPI inflation (12 months), 3-month change in unemployment, payroll growth,
  NFCI, 10y-2y, Baa-10y spread and its 3-month change, fed funds. As first published.
Cross-sectional features are ranked within each date; market and macro stay levels
(standardized on training rows). Missing values count as 0 (the middle). XBRL facts start
2009-2011, so fundamentals are missing for early years (coverage about 70-80% overall).
**Estimated trading cost**: Abdi-Ranaldo spread from daily high/low/close over the 21
sessions before the date, plus 2 bp slippage a side; shown separately, subtracted only in
"net" figures. Analyst histories are **not** used in stock training (Yahoo's lists are
today's versions; not defensibly point-in-time). News is **not** used in training (no history).

### 6.4 Models
Pooled across stocks (one model per target and horizon, not per stock):
- Returns: momentum (benchmark), Elastic Net, XGBoost quantiles (each via `Probabilistic`);
  XGBoost regression for the return above SPY.
- Events (up/down breakout, large loss, deep drawdown): base rate (null), L2 logistic
  (penalty by validation log loss), XGBoost logistic (eta 0.05, early stopping).
- Trend state: multinomial logistic and XGBoost softprob.
- Severity: XGBoost quantiles on the drawdown from entry (5th, 50th percentiles).
- Volatility: EWMA vs GJR-GARCH(1,1) with Student-t errors (study only).
- Training universe: S&P 500, each stock from its date added; models are saved per
  (cutoff, horizon, features, thresholds, code version) and reused for later batches that day.

### 6.5 Batch pipeline
Inputs checked -> shared data fetched once (bars incremental; SEC facts and Yahoo actions
4 at a time, rate-limited, retried, cached) -> per horizon: panel, models fitted or loaded,
every S&P 500 stock forecast at the cutoff (so requested stocks get a percentile) -> news
collected and summarized -> evidence attached from the latest study -> saved. A stock with
stale or missing prices, no trading volume, or under a year of history **abstains** with the
reason; failures are isolated per ticker. Per-ticker progress is reported. Measured: a
first batch of 5 tickers at one horizon took 202 s including training; a second batch the same
day reuses the models.

Each forecast (per stock and horizon) saves: expected return above cash (evidence-weighted
combination of Elastic Net and XGBoost quantiles), median, 5/95%, P(gain), each model's mean
and their spread, return above SPY, estimated round-trip cost, net expected return and its
percentile among the S&P 500, trend probabilities (up/down/sideways/unclear), breakout
probabilities with the barrier levels, large-loss and deep-drawdown probabilities, drawdown
5th/50th percentiles, current trend, filing age, the training base rates, the evidence status
per target (backed / weak / none / untested), news summary, reasons, and source.

### 6.6 News
Collected per stock with publication and first-seen times; identical headlines are one
record and near-duplicates (80% shared words within 3 days) are linked. *About* (named in
the first 40 characters) vs *mentions*. *Fact* vs *opinion* vs *unclear* by rule. Event tags
by rule: earnings, guidance, financing, buybacks/dividends, litigation, regulation,
management, M&A, analyst actions, products/contracts. Sentiment = P(positive) - P(negative)
from **FinBERT (ProsusAI/finbert) run locally** (fallback: a Loughran-McDonald word list,
named in the record); averaged over headlines *about* the stock in 7 days. No headline means
"sentiment unknown", never neutral. Optional paid reading for a shortlist (default 5) through
the AI gateway: structured events with facts separated from interpretation; citations must
point at real headline numbers.

### 6.7 Recommendations (reviewable, never executed)
For the Trading sleeve by default, from a run at 20 sessions: **buy candidate** when net
expected return > 0, return evidence *backed* by the final test, P(large loss) <= 15%, and no
data flag; at most 5 names; equal dollars, each <= 25% of the sleeve, together within
**settled** cash; fractional shares; orders under $1 dropped. **Watch** when positive but not
backed or flagged. **Avoid / consider reducing** when negative: **never a short order**; a sale
is at most what is held. Earliest execution: the session after the cutoff; hold through the
horizon (no day trades). Before sizing, the whole IRA (stocks and ETFs) is priced and each
candidate's correlation and beta to it, and the IRA's beta to SPY after adding it, are shown
(funds are not looked through to sectors). Decisions are separate records; fills stay in the
ledger.

### 6.8 Three evaluation systems, kept separate
1. **Historical strategy simulation**: the walk-forward study (section 7), using only
   information available at each decision date.
2. **Prospective Day-1 record**: each saved recommendation's buys vs putting the same
   dollars into SPY on the same dates (frozen static policy), at the same execution
   assumptions, once matured. Old recommendations are never regenerated.
3. **Actual performance**: the ledger's fills and values.
Matured forecasts get their realized outcomes in `stock_outcomes`, scored in a live scorecard
(IC per run date, 5-95% coverage, Brier vs base rate per event).

### 6.9 Next steps, justified by the measured evidence
1. **Test the ranking objective (XGBoost rank within date) and cross-sectionally ranked
   labels** under the point-in-time design: the market component dominates stock returns, so
   regressors collapse to market-wide splits; a within-date ranking target removes it. The
   earlier, looser S&P 500 study found IC 0.066 at 60 sessions with it.
2. **Point-in-time index membership** (members later removed) to remove survivorship bias.
3. **Use GJR-GARCH-t volatility** for sigma_h and the forecast scale (it beat EWMA).
4. **Keep breakout probabilities as risk context**; test whether they add anything beyond
   distance-to-barrier and volatility (a two-feature logistic baseline).
5. **Do not add a sequence model (PatchTST, VSN+LSTM) or a pretrained model yet**: no return
   model has passed the final test, the target is dominated by noise, and on ETFs the LSTM and
   Chronos-2 did not beat boosted trees. Revisit when a simple model shows confirmed skill.
6. **Fix tree counts** (no early stopping in deployment, or a fixed number from the study) if
   identical CPU/GPU results are wanted.

---

## 7. Validation methods and every metric

**Folds.** Chronological, shared by all assets; training uses only rows whose labels ended
before the test block (purge, embargo = horizon); preprocessing (standardization, penalty
choice, early stopping) fitted inside each training fold on its last fifth of dates (also
purged). Stock study: blocks of 2 years from 2012; **development 2012 to June 2024**, where
models are compared, the best chosen and ensemble weights learned; **final test July 2024
on**, scored once with everything frozen.

**Units of uncertainty.** Dates, not rows: hundreds of correlated stocks on one date are one
market state. Intervals by moving-block bootstrap over dates.

| Metric | Definition | Good sign |
|---|---|---|
| IC | Spearman rank correlation of forecast and outcome across assets on each date, averaged | > 0, interval above 0 |
| p (Holm) | Two-sided p of the IC t-stat, Holm-adjusted across the models in a trial | < 0.05 |
| R² out of sample | 1 - SSE(model) / SSE(zero forecast) | > 0 |
| Calibration (pooled / within date) | Slope of outcome on forecast, all rows / within each date | about 1 |
| Timing correlation | Date-average forecast vs date-average outcome | > 0 (it is about 0) |
| Pinball loss, coverage | Quantile loss at 5/50/95%; share of outcomes inside 5-95% | coverage about 90% |
| Brier, log loss | For P(gain) and every event, against the training base rate (null) | below the null |
| Brier skill | 1 - Brier / Brier(null), with a date-block bootstrap interval | > 0, interval above 0 |
| AUC | Ranking of events by probability | > 0.5 |
| Reliability table | Forecast probability tenths vs observed frequency | on the diagonal |
| Alerts | Top tenth by probability on each date: precision, recall, missed, false alarms, lift | lift > 1 |
| Failed breakouts | Share of alerted, realized breakouts that closed back inside | low |
| Crash lead time | Sessions from the forecast to the first breach of the drawdown threshold | longer |
| Trend scores | Multinomial log loss vs the training mix; accuracy vs always guessing the commonest state; recall per state | above the nulls |
| Severity | Share of drawdowns below the forecast 5th percentile (target 5%); pinball loss | about 5% |
| Top fifth vs all | Per date, equal-weight top fifth by forecast minus all, gross and net of costs on turnover | > 0 |
| Portfolio | Long-only top fifth by net expected return, non-overlapping holds, vs all stocks, momentum, trend rule: mean, annualized, total, max drawdown, turnover, by period | above benchmarks |
| QLIKE | realized/forecast variance - log(ratio) - 1 | lower |
| Ablation | IC of each feature-group set on development | added groups help |

Every run (ETF, stacking, stock, ablation variant) is stored in `research_trials`.

---

## 8. Stock study results (2026-10-24, trials `stock_study:h5/h20/h60`, rescored)

Universe: today's S&P 500 (503 stocks), each from its date added; all four feature groups;
thresholds `thr-14adb1ddde`. Development 2012 to June 2024 (314 dates at 5 sessions, 157 at
20 and 60); **final test July 2024 to October 2026 (59 dates at 5 sessions, 30 at 20 and 60)**.
Intervals: 95% moving-block bootstrap over dates. "Ranked" = dates on which the model told
stocks apart at all (a model that gives every stock the same number has no rank that day).

**Returns (IC, development / final test).**

| Horizon | Momentum | Elastic Net | XGBoost quantiles | Equal ensemble |
|---|---|---|---|---|
| 5 | 0.015 (-0.005, 0.036) / **0.042 (0.017, 0.089)**, Holm p 0.57 | 0.011, ranked 15 of 314 / -0.015, ranked 2 of 59 | 0.011 (-0.031, 0.046), ranked 169 / -0.034, ranked 9 | 0.011 (-0.009, 0.031) / 0.040 (0.012, 0.088) |
| 20 | -0.019 (-0.044, 0.011) / -0.015 (-0.035, 0.024) | 0.003 (-0.021, 0.022) / -0.007 (-0.042, 0.015) | 0.024 (-0.019, 0.064), ranked 96 of 157 / ranked 0 of 30 | 0.003 / -0.007 |
| 60 | 0.016 (-0.018, 0.068) / -0.034 (-0.054, 0.006) | 0.014 (-0.013, 0.036) / 0.002 (-0.051, 0.028) | 0.022 (-0.012, 0.063), ranked 104 / ranked 0 of 30 | 0.016 / -0.010 |

Return above SPY (relative model): IC 0.001 to 0.025 in development, -0.036 to 0.001 in the
final test. **No stock return model has skill that survives the final test and the multiple-testing
 correction.** The tree models often collapse to market-wide splits only (the same
forecast for every stock on a date): early stopping on a noisy target keeps very few trees.
No development-period model was significant, so the ensemble weights are equal.

**Events (ensemble of logistic and boosted trees; Brier skill vs the base rate, AUC, lift of
the top tenth).**

| Horizon | Up breakout | Down breakout | Large loss | Deep drawdown |
|---|---|---|---|---|
| 5 dev / test | +0.300 (0.263, 0.333), AUC 0.82 / **+0.281 (0.234, 0.338)**, AUC 0.82, lift 2.6 | +0.254 / **+0.200 (0.132, 0.338)**, AUC 0.82, lift 2.9 | -0.047 / +0.003 (-0.016, 0.021) | +0.005 / +0.026 (-0.027, 0.107) |
| 20 dev / test | +0.230 (0.197, 0.254) / **+0.230 (0.218, 0.287)**, AUC 0.77 | +0.183 / **+0.197 (0.186, 0.278)**, AUC 0.76 | -0.103 / -0.006 (-0.009, 0.003) | -0.050 / +0.019 (0.009, 0.029) |
| 60 dev / test | +0.143 / **+0.190 (0.144, 0.207)**, AUC 0.75 | +0.114 / **+0.107 (0.097, 0.146)**, AUC 0.70 | -0.083 / -0.002 | +0.007 / -0.081 (-0.156, 0.030) |

Base rates (final test): up breakout 30% / 52% / 71% at 5 / 20 / 60 sessions; down 22% /
43% / 58%; large loss 5-7%; deep drawdown 9-13%. Of alerted breakouts that happened, 43-55%
failed (closed back inside the range). Deep-drawdown alerts led the first breach by about 3,
10 and 31 sessions on average, but their skill is not reliable (development negative).
**Breakout probabilities are the only stock forecasts with confirmed skill**; they are
probably largely mechanical (distance to the barrier relative to volatility), so they
describe risk, not return.

**Trend states**: no skill. The boosted trees predict "unclear" (the commonest state) almost
always; multinomial logistic is worse than the base mix in development.

**Severity** (drawdown 5th percentile): 4.6-8.4% of outcomes fell below it (target 5%):
roughly calibrated, slightly too narrow at 5 sessions in the final test.

**Portfolio** (long-only top fifth by net expected return, non-overlapping holds, per period,
above cash): 20 sessions development +1.41% vs all stocks +1.17% (difference +0.24%,
interval -0.02% to +0.51%), final test +0.07% vs +0.80% (difference -0.73%); momentum was best
in the final test. 60 and 5 sessions: no difference. Turnover about 0.5-0.7 per period.
**No robust portfolio edge.**

**Ablation** (20 sessions, development): Elastic Net IC about 0 with any feature set;
boosted trees 0.004 (technical only), 0.055 (+ market), 0.104 (+ fundamental), 0.058
(0.003, 0.083) with all four groups. The tree figures are computed only on dates on which the
model ranked anything, and their intervals are mostly undefined: **not reliable evidence that
any group adds value**.

**Volatility** (20 sessions, 40 stocks, 6,720 forecasts, QLIKE, lower is better): GJR-GARCH(1,1)
with Student-t errors 0.351, EWMA 0.408. GJR-GARCH-t is better.

**Compute** (measured): panel build about 28-30 s per horizon (503 stocks, 2006-2026); a
first batch of 5 tickers at one horizon took 202 s with training, and reuses its models
thereafter. Boosted-tree event model on 181k rows: CPU 3.0 s, GPU 1.5 s; on 90k rows: CPU
1.8 s, GPU 2.4 s (the GPU is used automatically only from 200k rows). CPU and GPU fits are
**not numerically identical**: early stopping chose different numbers of trees (53 vs 58;
165 vs 245), giving probability differences of 0.01-0.03 on average and up to 0.13. Runs
record the device used.

**What the app does with this.** A stock forecast's evidence status per target comes from the
final test: returns are *none* at every horizon today, breakouts are *backed*. Recommendations
therefore list every stock as *watch*, never as a buy candidate, until a return model passes
the final test. The forecasts and probabilities are still saved and shown, labelled.

**The earlier S&P 500 study** (price features only, today's members with no entry dates,
yearly refits, XGBoost ranking objective) found IC 0.066 (0.038, 0.093) at 60 sessions. The
point-in-time study here did not include the ranking objective; testing it under the stricter
design is the first next experiment.

**Scoring fix.** The first scoring of this study stored the Brier-skill interval high-first; it
was rescored from the saved predictions (no refitting) and recorded as new trials.

---

## 9. Compute, costs and controls

- `cpu`, `gpu` or `auto` (GPU only for boosted-tree jobs of 200,000+ rows, and for
  Chronos-2 and the LSTM); one GPU job at a time (lock file); a GPU failure reruns on the CPU;
  never a cloud fallback. Measured: XGBoost 200k x 40, 300 rounds: CPU 2.4 s, GPU 1.6 s;
  Chronos-2, 64 series: CPU 4.8 s, GPU 1.0 s, medians agree to 6e-8.
- AI gateway: cache key = task + provider + model + schema version + information cutoff +
  input hash; concurrent identical requests wait for one; daily/monthly dollar and request
  budgets (defaults $1/day, $10/month); **a model with no configured price cannot be called**;
  `TRADINGAGENTS_NO_PAID_AI=1` refuses all; every request's tokens and cost logged. Page loads,
  polling, replays and tests never pay. Currently no price is configured, so paid features
  (AI ratings, market brief, moat sources, news reading) are blocked until one is set.
- Data: prices, statements, filings, economic data and news are free; downloads are
  incremental and cached (section 2).

---

## 10. How to run

```
.venv\Scripts\python.exe -m cli.main web                 # the web app, http://127.0.0.1:8765
.venv\Scripts\python.exe -m cli.main suggest             # today's plan and orders
.venv\Scripts\python.exe -m cli.main stocks run AAPL MSFT --horizons 5,20,60
.venv\Scripts\python.exe -m cli.main stocks run --watchlist --holdings --csv list.csv
.venv\Scripts\python.exe -m cli.main stocks show [--run ID] [--horizon 20]
.venv\Scripts\python.exe -m cli.main stocks ideas --horizon 20
.venv\Scripts\python.exe -m cli.main stocks decide IDEA TICKER accepted
.venv\Scripts\python.exe -m cli.main stocks outcomes     # record matured outcomes; scorecard; Day-1 record
PYTHONPATH=. .venv\Scripts\python.exe experiments\stock_study.py 5 20 60
```
API (no stock page yet): `POST /api/stocks/run`, `GET /api/stocks[?run=]`,
`POST /api/stocks/ideas`, `POST /api/stocks/decide`, `POST /api/stocks/watchlist`,
`GET /api/stocks/outcomes`, `POST /api/stocks/news-ai` (paid).

---

## 11. Known limits

- **Survivorship**: the stock universe is today's S&P 500. Entering each stock from its date
  added removes the pre-inclusion run-up, but stocks later removed are missing.
- **One path of history**, mostly a US bull market (2012-2026).
- **No intraday data**: breakout ordering within a bar is unknown (recorded, not invented);
  execution is modelled at the next close, not the open.
- **Costs** are estimated from daily bars (Abdi-Ranaldo); market impact is ignored (order
  sizes are tens to hundreds of dollars).
- **Fundamentals** begin 2009-2011 (XBRL); some tags are missing for some companies.
- **Analyst data** (Yahoo) is today's version and is excluded from stock training.
- **News** exists only from when collection began; no news backtest is possible yet.
- **Chronos-2** may have seen the test prices in pretraining; only the live record is clean.
- **Paid AI** features are blocked until a model price is configured.
- **The app never trades**; recommendations are research for the owner to review.

---

## 12. Daily stock decisions, capital settings and challengers (added later on 2026-10-24)

Full detail: `docs/DECISIONS.md`. In brief:

- **Capital settings** (Settings > Capital; `book capital`; `/api/capital`): Trading capital $1,000 (net
  funding target, not a daily budget), Investing capital $4,800, daily investment $27.61 (planned
  purchase per session, bounded by settled cash). Validated, versioned, snapshotted into every record;
  changes produce reviewable transfer proposals and never move money or retrain models.
- **Execution contract** and one **position engine** for labels, simulation and the shadow account:
  next-close entry, close-confirmed levels filled at the next close, exact deadlines, no same-day exits,
  T+1, dividends as cash, executable prices.
- **Foundations fixed** after verification: point-in-time membership (fja05680/sp500; survivorship
  partial), members-only ranking with requested-ticker invariance, unknown capex, split-consistent
  valuation, expm1 barriers, no zero cash returns, per-series FRED lags, as-of pricing, missing risk
  blocks approval, limits including holdings, evidence bound to the exact design. The original stock
  study's evidence no longer applies (new thresholds version `thr-016cb9da59`).
- **Decisions**: ENTER / PAPER ENTER / WAIT / HOLD / REDUCE / EXIT / REVIEW with size, execution date,
  expected net P&L and spread, P(profit), P(large loss), levels, deadline, reasons and blocks.
- **Decision study**: XGBoost learning-to-rank IC 0.020 (0.005, 0.036) in development; trade-quality
  probabilities worse than the base rate; the complete policy trails cash-flow-matched SPY (best
  development policy +210% vs SPY +358% over 2013 to mid-2024; exploratory +5.0% vs SPY +43% and cash
  +9.9%). **Not validated**: the app is in experimental mode (WAIT; PAPER ENTER feeds the shadow account).
- **Challengers** on 150 stocks: no ranking skill for PatchTST, Chronos-2, TimesFM 2.5; Kronos-mini
  0.041 (0.016, 0.067) in development but likely contaminated and 0.014 (not significant) after.

## 13. Screener, signals, model verdicts, AI agents and the evidence packet (added 2026-10-24 to 2026-10-25)

### 13.1 Quality and Wall Street screen (`stocks/screen.py`; Screener tab; `stocks screen`)

Every S&P 500 stock (503) plus the watchlist; built on request only, every fetch cached.
- **Quality score (0-100)**: share of the ticker page's tests passed: the moat checks (`research/moat.py`:
  pricing power, returns on capital, profitability, cash generation, steady growth, with sector bars and
  bank-specific checks) plus the checklist's financial-strength items (debt to equity, interest coverage
  flag < 3, current ratio flag < 1.0, a high-rate downturn test) and profitability items (FCF, ROIC > 15%,
  gross/operating/net margin trends) (`research/quality.py`). Pass = 1, "check" = 0.5, flag = 0; tests
  without data left out. **Quality rank**: by quality score, ties share a rank.
- **Wall Street**: this month's analyst counts as a consensus score 1 (strong buy) to 5 (strong sell), the
  number of analysts, upside to the mean price target. Yahoo, as of today: not point-in-time.
- **Valuation** (forward P/E vs its own history, EV/EBITDA, FCF yield, reverse-DCF implied growth): shown,
  not scored.
- **Combined score / rank**: mean of three within-screen percentiles (quality, consensus with stronger buys
  higher, target upside) x 100; a stock missing one is ranked on the rest. Untested for returns.

### 13.2 Stock signals (`stocks/signals.py`; ticker page and the Screener's "Signals and action" view)

From the latest saved stock-model forecast per stock (horizon 20 sessions; `stocks/pipeline.py`): P(trend
up / down / sideways / unclear), P(breakout above the prior 20-session range), P(breakdown below it: any low
in t+1..t+20 below the lowest low of the last 20 sessions), P(large loss), P(deep drawdown), drawdown
quantiles (dd_q05, dd_q50).
- **Lift** = probability / base rate in training; **peer rank** among today's stocks.
- **Warnings**: breakout likely / breakdown risk need lift >= 1.2 and today's top fifth; crash warning lift
  >= 1.5, P >= 10%, top fifth; deep drawdown lift >= 1.5, P >= 20%, top fifth. The top-fifth rule exists
  because a market-wide move raised every stock's breakdown lift (234 of 499 stocks flagged without it).
- **Flat-model detection**: a 10th-90th percentile spread across stocks below 2 points means the model does
  not tell stocks apart, and it raises no warning. Today the large-loss model gives every stock about 6.9%
  and the deep-drawdown model about 17%: both flat. Trend up/down are flat too; only sideways varies.
- Every signal's evidence status is **untested** for the current thresholds version.
- Example (ANET, 2026-10-23): close $172.37, 20-day low $168.24 (2.4% below), P(breakdown) 78% vs 42% usual
  (top 8%): largely a consequence of sitting near the range floor with about 3% daily moves.

### 13.3 The verdict (`stocks/verdict.py`): BUY / WATCH / HOLD / SELL / AVOID

**Decided only by the internal 20-session quant model** (the owner's instruction: AI ratings must not
decide). The daily decision (`stocks/daily.py`) saves every ranked stock's score: rank (ties broken by
P(profit), then ticker), rank_pct, score_pct, expected net after costs (from the binned score-to-P&L map),
net sd, P(profit) (trade-quality model), P(large loss) (risk model), cost.
- Not held: **BUY** = the daily decision says ENTER, or top fifth by rank AND expected net > 0 AND
  P(profit) >= 50% AND P(large loss) <= 15%. **AVOID** = bottom fifth or expected net < 0. **WATCH** otherwise.
- Held: **SELL** = the daily decision says EXIT/REDUCE, or bottom fifth with expected net < 0. **HOLD**
  otherwise (top fifth: a candidate to add to; middle; bottom fifth: weakest, no sell signal).
- **Context only** (shown, never decides): breakdown/crash warnings, quality score, Wall Street, agents' rating.
- Today: 100 BUY, 307 WATCH, 96 AVOID among S&P 500 stocks; #1 is TPR (the top five tie on score).
- **Funds**: a fund's score is its rank among the 41 funds the ETF forecast engine covers, by the combined
  20-session forecast (`research/engine.combine_runs`: return above cash, P(gain); no large-loss figure).
- **Stocks outside the S&P 500** that are held or watched (ARR, OKLO) are scored by the same stock model and
  ranked against that day's members (members keep their ranks).
- Known properties: many tied scores (32 stocks share one value); the binned map is **not monotone** (top bin
  +0.58%, next bin +0.70%); expected net is positive for every stock (+0.29% to +0.70%: market drift), so
  "expected loss" never fires and AVOID comes from rank only. Decision candidates were re-ordered by score
  (they had been sorted by the binned expected net, which left out the #1 stock). Validation (section 12):
  ranking IC 0.020 in development; trading on it did not beat SPY.

### 13.4 Entry plan (`signals.entry_plan`; ticker page)

Limit-entry levels, each with the model's chance of being reached within 20 sessions: the 20-day low
(P(breakdown)), the median dip close x (1 + dd_q50) (about 50%), the deep dip close x (1 + dd_q05) (about
5%), and "buy at the next close". Each entry E gets the barrier policy's exits: stop E x (1 + expm1(-sigma_h)),
take-profit E x (1 + expm1(+sigma_h)), deadline 20 sessions (`positions.POLICIES["barrier_20"]`, the exit rule
the decision study tested). Experimental. Example AVGO: next close $321.68 (stop $289.05, target $358.00);
median dip $307.13; 20-day low $320.23 (83%); deep dip $265.63.

### 13.5 Decisions page

- **AI decisions card**: shortlist = holdings, stocks the agents analyzed in the last 30 days, model-verdict
  BUYs (model-rank order), the quant decision's PAPER ENTERs; the latest economic brief (an LLM summary of
  macro data and headlines); analyst toggles; an editable AI budget. "Agents' view (not used)" (rating mapped:
  Buy/Overweight -> BUY, Hold -> HOLD or WATCH, Underweight/Sell -> SELL or AVOID) beside "Model verdict",
  with agreement and a link to the run's report.
- **Quant model decision** (section 12): candidates in model-rank order; all WAIT (experimental mode; the
  Trading sleeve's $1,000 sits in SGOV, so settled cash is $0), PAPER ENTER feeding the shadow account.

### 13.6 AI agents (the TradingAgents graph): providers, roles, cost controls, records

- **Graph**: market (technical), sentiment, news & economy and fundamentals analysts (each a tool-calling
  loop over local data tools), a bull and a bear researcher (one round), a research manager (structured
  ResearchPlan), a trader (TraderProposal), aggressive / conservative / neutral risk analysts (one round), a
  portfolio manager (PortfolioDecision: rating Buy/Overweight/Hold/Underweight/Sell, executive summary,
  thesis, optional price target and horizon), and a reflector that writes lessons to a memory log.
- **Providers**: default Google **Gemini 3.8 Flash** on a verified free-tier key (no charge; Google may use
  free-tier content); fallbacks Gemini 3.8 Flash then **Gemini 3.5 Flash-Lite** on provider failure or a model
  without a price; budget refusals stop. The free 3.8 Flash quota ran out after about two full runs a day.
  OpenAI key present (gpt-6-luna $0.10/$0.50, gpt-6-sol $2/$10 per M tokens); Groq (gpt-oss-120b $0.15/$0.60;
  free plan about 8K tokens/minute, too small for analyst requests); the DeepSeek key was rejected as invalid.
- **Per-role models** (`llm_clients/roles.py`; Settings > Agent models): 13 roles; currently bear,
  conservative, neutral, research manager and portfolio manager on Groq gpt-oss-120b and the rest on Gemini,
  so opposing debate sides use different models; fallbacks keep roles assigned to other providers.
- **Gateway** (`ledger/ai_gateway.py`): every model priced or refused; budget reserved per run and checked
  before each request ($5/day, $50/month, 600/6,000 requests, counted at paid rates even on the free tier);
  usage recorded per request under the provider that answered; results cached by inputs (ticker, day,
  analysts, models, roles). Page loads never trigger paid AI.
- **Token reductions** in tool output (`agents/tools.py`): prices to the cent, no all-zero columns, last 60
  sessions; indicators to 2 decimals without weekend placeholders; all indicators in one call. A one-analyst
  run fell from 98k to 51k input tokens and from 15 to 12 requests.
- **Records**: every request and reply is logged (`llm_clients/request_log.py`, `requests.jsonl` per run;
  ticker page "what was sent"; `stocks calls`; `docs/AGENT_API_IO.txt`). Each run writes
  `0_FINAL_DECISION.md` with the model's verdict (deciding) and the agents' view (not used).

### 13.7 Manual chat route and the evidence packet (`agents/manual.py`, `agents/packet.py`)

"Copy the prompt" -> paste into ChatGPT / Gemini / Claude -> paste the answer back -> parsed, saved as a
report, the rating recorded (one rating per ticker and day), the final decision written. The prompt
(`evidence-packet-v1`) is grounded instructions followed by one JSON evidence packet read from local caches
only (no downloads):
- instrument and type (Yahoo quoteType), cutoff and timezone, latest session;
- one price snapshot `PX1` computed after cutting at the cutoff (EMA 10, SMA 50/200, RSI 14 Wilder, MACD
  12/26/9, Bollinger 20/2, ATR 14, 20-day high/low, 250-session high and drawdown, 20-day realized
  volatility, volume); comparisons at 5/20/60/250 sessions; the last 10 bars;
- stocks: SEC facts filed before the cutoff (period end, filed, available-from, form, value) and derived
  ratios; ETFs: cached-profile fund facts flagged "not point-in-time", a leverage flag, holdings / NAV /
  premium listed as missing;
- news: stored headlines with a known publication time on or before the cutoff's end of day (New York),
  duplicates merged, relevance-ranked, with ID, publisher, published, first seen, URL, "headline only" or summary;
- macro: FRED/ALFRED as known at the cutoff (vintages or lags), units, observation dates, 1- and 12-month
  comparisons in percentage points or percent change, a true 12-month CPI figure;
- the checks actually run (date order, missing closes, OHLC consistency, price basis, coverage), discrepancy
  flags (precedence: local store > OHLCV cache > profile), one missing-data list, portfolio context "unknown";
- size limits that drop whole items (old bars, the 250-session comparison, quarterly items, news beyond five);
- no app verdicts, ranks or holdings (model forecasts optional, off by default).
The answer contract keeps rating / executive_summary / investment_thesis / price_target / time_horizon and
adds optional `evidence_ids` and `insufficient_evidence` (the five-value rating cannot abstain; abstention is
recorded). Measured: TPR prompt 11,541 -> 4,953 tokens, TQQQ 7,047 -> 3,197 (o200k_base). Not yet evaluated on
answer quality. Agent (API) runs still assemble their input through tools, not the packet.

### 13.8 Open problems found while building this layer

1. Verdicts come from a model that has not beaten SPY; about 100 stocks are BUY on any day.
2. A coarse, non-monotone score-to-P&L map; heavy score ties; expected net positive for every stock.
3. Crash and deep-drawdown models are flat across stocks today; trend up/down nearly flat.
4. Signals' evidence status is "untested" for the current thresholds version.
5. Leveraged ETFs (TQQQ, SOXL) rank top among funds and are held, while the account rule says "no leverage"
   (meant as no margin): an unresolved policy question.
6. Analyst and profile data are today's values (not point-in-time); the screen is a snapshot.
7. LLM ratings disagree with the model (TPR: model #1 BUY, ChatGPT Sell); there is no evaluation yet of whose
   calls are better.
8. Free-tier quotas limit agent runs to a few a day; Groq's per-minute token cap blocks analyst requests.

## 14. Inventory: every model, its inputs, outputs, training and evidence

| # | Model | Where | Inputs (data, features) | Output | Training / method | Evidence |
|---|---|---|---|---|---|---|
| 1 | Standard rule (default expected return) | `optimizer` | 10+ years of daily adjusted closes per fund | expected growth = shrunk Sharpe x volatility, per year | closed form; no fitting | the plan's default; models below did not beat it in the portfolio after costs |
| 2 | Portfolio optimizer | `optimizer/plan.py` | expected growth, worst historical drop per fund (or past crashes replayed), risk level 1-10, growth target | target mix, orders, recurring buys | maximize expected growth subject to a stress-loss cap (worst-case or scenarios), per-fund cap of a quarter of the budget | 2008-2026 backtest: scenarios mode exceeded its cap by up to 1.5x, so worst-case is default |
| 3 | ETF forecast models (Elastic Net, XGBoost, XGBoost ranking, XGBoost quantiles, momentum) | `research/engine.py`, `forecast.py` | 41 funds; PRICE_FEATURES (momentum 1/3/6/12-1 m, reversal, volatility, downside vol, beta, distance from 52-week high, MA200 gap, drawdown, range, dollar volume, volume trend, Amihud), MARKET (VIX, VIX slope, 10y change, 3m bill, SPY momentum and vol), MACRO (CPI 12m, unemployment change, payrolls, NFCI, curve, credit spread and change, fed funds) | log return above cash over 5/20/60 sessions, quantiles, P(gain) | walk-forward with purging; within-date ranking; ensemble weighted by significant walk-forward IC (Holm) | ranked funds better than chance; did not beat the standard rule after costs |
| 4 | "+chronos" variants and stacking | `research/foundation.py` | the above plus Chronos-2 forecast features | as above | XGBoost on stacked inputs | as above; Chronos contamination risk |
| 5 | Chronos-2 (pretrained) | `research/engine.py` | price history | return forecast | zero-shot (amazon/chronos-2) | given no weight: pretraining may include the prices; scored only on a live record |
| 6 | Volatility: GJR-GARCH vs EWMA | `research` | daily returns | volatility forecast | GJR-GARCH(1,1) vs RiskMetrics EWMA (0.94) | GJR-GARCH beat EWMA on volatility |
| 7 | Stock event models: BaseRate, Logit, XGBEvent | `stocks/models.py`, `pipeline.py` | S&P 500 panel (point-in-time membership): technical group (the price features plus MA slopes, range positions, relative to SPY and sector, volume change, current trend), market, fundamental (SEC point-in-time: growth, margins, ROA, leverage, dilution, yields, filing age), macro (ALFRED vintages) | P(breakout up / down), P(large loss), P(deep drawdown) over 5/20/60 sessions | pooled across stocks; walk-forward folds shared across stocks; purged by label end | breakouts showed skill in the stock study; under the current thresholds: untested |
| 8 | Trend, Severity, Relative models | `stocks/models.py` | same panel | trend class probabilities, drawdown quantiles, return relative to the market | multinomial / quantile / regression | no return skill in the stock study |
| 9 | Stock return models (Elastic Net, XGBoost quantiles) | `stocks/pipeline.py` | same panel | mean / median / 5-95% return, P(gain), net after costs | walk-forward | no return skill; P(gain) is about 60-66% for every stock (drift) |
| 10 | Decision ranker (XGBoost learning-to-rank) | `stocks/adapters.py`, `daily.py` | same panel; date groups | within-date score -> rank | LambdaMART-style ranking by date, fitted weekly, cached by code hash | IC 0.020 (0.005, 0.036) development, 0.037 exploratory; heavy score ties |
| 11 | Score-to-net map | `adapters.score_to_net` | ranker percentile | expected net P&L, sd, 10% quantile | 10 bins, shrunk | top decile realized -0.5% in the study; not monotone today |
| 12 | Trade-quality model | `adapters.TradeQuality` | panel + score | P(trade profitable after costs) | logistic / XGBoost, optional sigmoid calibration, cross-fitted | worse than the base log loss (0.69) |
| 13 | Risk model | `daily` bundle | panel | P(large loss at the 20-session exit close; terminal, not the worst point while held) | classifier | flat across stocks today (about 7%) |
| 14 | Position engine and policies | `stocks/positions.py` | prices, dividends | trades for fixed 5/20/60 and barrier 20/60 policies | rule-based; T+1, next-close fills, no same-day exits | best development policy +210% vs SPY +358%; exploratory +5% vs SPY +43% |
| 15 | Challengers: PatchTST, Chronos-2, TimesFM 2.5, Kronos-mini | `stocks/challengers.py` | price sequences, 150 stocks | return scores | compact PatchTST trained locally; others pretrained | no skill except Kronos-mini in development (0.041), likely contaminated; 0.014 after |
| 16 | News sentiment (FinBERT) | `stocks/news.py` | stored headlines and summaries | sentiment score, relevance (about / mentions / other), events | pretrained FinBERT, local GPU or CPU | collected prospectively; not used in training |
| 17 | Moat and quality checklists | `research/moat.py`, `quality.py` | Yahoo annual statements, profile, prices, dividends, analysts | pass / check / flag per test, verdict counts | fixed rules with sector bars | untested for returns |
| 18 | Moat sources (AI reading of the 10-K) | `research/moat_ai.py` | the latest 10-K text | switching costs, network effects, cost advantages, intangibles, efficient scale, with quotes | one paid LLM call per filing, cached | qualitative |
| 19 | Market brief | `optimizer/market.py` | macro data and headlines | summary and trends | one LLM call per day, cached | qualitative |
| 20 | TradingAgents multi-agent graph | `tradingagents/graph` | tool data (prices, indicators, news, statements, macro), memory of past lessons | rating Buy..Sell with thesis | LLM agents (13 roles), one debate round each | shown as the agents' view; not used for decisions; no evaluation yet |
| 21 | Verdict and entry plan | `stocks/verdict.py`, `signals.py` | items 10-13 (stocks), item 3 (funds), item 7 (dips, breakdown) | BUY / WATCH / HOLD / SELL / AVOID; limit entries with stops and targets | fixed rules on model outputs | inherits items 10-13: experimental |

## 15. Fixes from the audit validation (2026-10-25)

An outside audit was checked against the code. Each confirmed finding was fixed as follows. No recorded
result was rewritten: old trials, old shadow snapshots and old reports stay as they were, and corrected
methods apply only to new runs.

| Finding | Fix | Where |
|---|---|---|
| X1: re-running a cutoff appended its shadow orders again (39 pending orders, $9,350 against $1,000) | orders are taken once per cutoff; pending orders count against the name limit and reserve cash; shadow version 2 restarts the account and records why | `stocks/shadow.py` |
| A lot bought on the latest session got that session as its deadline | deadlines past the loaded prices come from the exchange calendar | `positions.Book._next` |
| C: a verdict said BUY while the evidence gate said not validated | `verdict()` returns `verdict` (the model's view), `action` (what the gate allows: BUY becomes WAIT and SELL becomes HOLD unless validated) and `label`; pages show the action | `stocks/verdict.py`, pages |
| C: a stock with no risk estimate could pass | a missing or NaN risk estimate fails the large-loss check | `verdict.py` |
| J: a final decision used the latest model run, even for an older analysis day | it uses the newest decision with data through the analysis day | `server._write_decision` |
| J: a pasted answer could not be tied to the evidence it was given | each manual prompt saves its packet under a content ID (`packet_id`, first line of the prompt); the answer echoes it; the run folder keeps `evidence_packet.json` and `provenance.json`; cited IDs missing from the packet are flagged | `agents/packet.py`, `server.manual_*` |
| J: an abstaining answer showed as its placeholder rating | it shows as ABSTAIN | `verdict.rating_on_record`, `ai_decision` |
| J: a second run for a day overwrote the first | the earlier run moves to `archive/<time>` and stays listed as superseded | `ai_views.fresh_report_folder` |
| G: "large loss during the position" | the label is terminal: "at the exit close"; "Chance it fills" is "Chance the price reaches it" | pages, `daily.py`, `signals.py` |
| F: the legacy stock scorer averaged log returns and annualized at the minimum spacing (1.3x to 1.9x too high) | baskets compound from simple returns less cost; annualized on actual spacing | `stocks/evaluate.portfolio` |
| F: the ETF backtest traded at the close it planned from | trades one session later (`lag=1`; `lag=0` keeps the old figure for comparison) | `optimizer/backtest.py` |
| D: a past `--as-of` could see later prices and reuse a model fitted later that week | prices are cut at the cutoff after executable prices are computed on the full history; fits use only trades closed and outcomes known by the cutoff; a later bundle is never reused for an earlier cutoff | `stocks/daily.py` |
| B: validation was not bound to the model it validated | a study must match the policy's horizon and version, features, thresholds and model code; the prospective record must be the current shadow version and policy and beat SPY at the account level with a 20-session block-bootstrap interval above zero | `daily.evidence`, `shadow.py`, `experiments/decision_study.py` |
| H: recorded splits were ignored when rebuilding lots | a split scales open lots and divides entry prices; shadow lots record their split basis and take a later split once | `daily.lots_from_ledger`, `shadow._apply_splits` |
| H: dividends | credited on the ex-date in the shadow account (the cash really arrives on the pay date); documented as an approximation | `shadow.py` |
| I: many trials, selection on development | new decision studies record the deflated Sharpe ratio of daily excess over SPY, charged for the rule grid; recorded, not yet a gate | `walkforward.deflated_sharpe`, `study.summary` |

Consequence: every decision study recorded before 2026-10-25 lacks the model-code binding, so the
current model reads as untested until a new full study is run. That is deliberate: it is the
conservative reading of evidence that cannot be tied to the code now running.

## 16. Entry plans (added 2026-10-25)

The agents now say how an entry could be made, and the app checks it. Nothing is executed: the app
recommends and records, you place the order.

**Flow (no extra model calls).**

1. The **Trader** proposes an `entry_plan` inside its existing structured answer (`TraderProposal.entry_plan`).
2. The three **risk analysts** read the trader's plan in its proposal text and are asked to challenge its price, trigger, timing and expiry.
3. The **Portfolio Manager** approves, revises or rejects it (`PortfolioDecision.entry_plan`, `entry_plan_review`).

Only these five prompts carry the entry instructions; the analysts, researchers and research manager do not.
A pasted chat answer may include the same optional `entry_plan` in its JSON; it is recorded as the manager's
plan, with no separate trader proposal.

**Plan shape** (all optional; answers without it parse as before):

```json
{"status": "ready|conditional|wait|unavailable", "entry_type": "market|limit|conditional|null",
 "entry_price": null, "entry_range_low": null, "entry_range_high": null, "currency": null, "trigger": null,
 "earliest_execution": "YYYY-MM-DD", "expires_at": "YYYY-MM-DD", "evidence_ids": [], "rationale": null}
```

**Reference quote** (`agents/entry_plan.reference_quote`, local store only):

- It is the last close on or before the cutoff, **as traded**.
- On the store's latest session the adjusted close equals the traded close.
- For an earlier session, the later dividends and splits are undone from cached corporate actions.
- The basis is "verified" only when those actions were fetched after the store's last session.
- Agents see it as one line ("Entry reference: …"). The chat packet carries it as evidence item `REF1`.

**Local feasibility check** (`entry_plan.validate`).

Blocks (the plan is kept but cannot be used):
- the action is not Buy or Overweight (long-only: no short, no entry from a Sell);
- the reference is missing or its basis is unverified;
- the reference close is more than one session older than the cutoff;
- the type is unknown;
- a limit has no price, or a conditional has no trigger;
- a price is not positive, a range is unordered, or the entry is outside its range;
- the currency differs from the quote's;
- the price is more than 20% from the reference;
- the price matches the adjusted close rather than the traded one;
- execution is earlier than the session after the reference close, or not on a trading session;
- the plan expires before it starts;
- the stock was sold on the planned entry day (a day trade);
- the name limit is reached;
- settled cash after reservations is below the minimum order;
- the whole-IRA concentration cap would be exceeded.

Warnings: more than 8% from the reference, a buy limit above the last close, no currency, no expiry, an
expiry longer than 20 sessions, an old quote, already held, stop or target on the wrong side, and always
"no exit on the entry session".

Sizing uses the capital settings and the stock rules:

```
min(position cap, settled cash after reservations / free name slots)
```

It shows the stop-loss, the target, reward to risk and the loss at the stop. States are: feasible,
feasible if triggered, wait (with its reconsideration condition), no plan, blocked, expired.

**Persistence** (migration 6, table `entry_plans`, append-only):
- one row per stage (trader, portfolio_manager), linked by `parent_id`, with the review;
- the cutoff, the run folder, the source and the side;
- the plan as given, the reference quote, the check, the capital settings version and hash, and the rules;
- the evidence: cited IDs, packet ID, and the model decision as of that day.

When the plans are read, the newest final plan is current, earlier ones are marked superseded by the
next, and a plan past its expiry is expired. Each agent run also writes `entry_plan.json` beside its
reports.

**UI.** The ticker page has a new card, "Agents' entry plan", for the current plan. It shows:
- the manager's action and review, and the trader's action;
- the reference price with its session and basis;
- the evidence (packet, model run, unknown cited IDs);
- the final plan: status, type, price or range, condition (or "Reconsider when" for a wait), earliest
  execution and the first allowed day, expiry, evidence and rationale;
- the app's check with its blocks and warnings, the size, the stop and target, and reward to risk.

The trader's proposal is shown separately, below the final plan. Earlier plans sit behind a fold, marked
superseded or expired. A missing value reads "Insufficient data", never zero. End-to-end example:
`tests/test_webapp.py::test_an_entry_plan_from_a_pasted_answer_is_checked_kept_and_shown`. The plan there
is blocked because the test account has no settled cash.

**Measured token impact** (o200k_base tokenizer, an approximation for the Gemini and Groq tokenizers):

| Item | Before | After | Added |
|---|---|---|---|
| Trader output schema | 426 | 806 | +380 |
| Portfolio Manager output schema | 392 | 826 | +434 |
| Trader prompt (instruction + reference line) | | | +120 |
| Each risk analyst prompt (instruction + reference + plan in the trader's text) | | | +128 (x3 per round) |
| Portfolio Manager prompt | | | +167 |
| Model output (typical plan, trader + manager) | | | about +200 |
| **Whole agent run** | | | **about +1,485 input, +200 output** |
| Chat prompt instructions | 391 | 540 | +149 |
| Chat packet `REF1` item | | | +56 |

The schemas were trimmed for this: no docstring or field descriptions on `EntryPlan`, since structured
output sends both on every call. `docs/AGENT_API_IO.txt` is built from a recorded run, so it shows the
old prompts until the next agent run is recorded.
