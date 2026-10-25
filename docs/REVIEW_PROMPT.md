# Review request: a personal quantitative portfolio and stock-research system

You are reviewing a working personal investment-research system. Act as **a senior
quantitative researcher and a senior ML engineer who has seen many backtests fail in live
trading**. Your job is to find what is wrong, weak, unproven or dangerous, and to say
precisely how to fix or test it. Praise is not useful; specific, checkable criticism is.

## Who it is for and what is at stake

One person's **Robinhood Roth IRA, about $5,844**: a cash account (buys need settled cash,
sales settle T+1, no margin, no shorting, options at most level 2). It is split into two
bookkeeping sleeves: **Investing** (about $4,844, long-term ETFs and stocks) and **Trading**
($1,000 initially, then its own P&L and transfers). The system plans, forecasts and
records; **it never places orders**; the owner trades by hand. Contributions are about $27.61
per trading day (the Roth IRA limit is $7,500 a year under age 50).

The owner wants to know: (1) whether any of the forecasting is good enough to influence real
money, (2) whether the evidence is trustworthy (leakage, multiple testing, survivorship,
statistical power), (3) what to fix first, and (4) which next experiment is most worth its
cost.

## How to answer

Use exactly these sections, in this order:

1. **Verdict (at most 6 lines).** Is any forecast fit to influence allocations? Which, at
   what horizon, under what conditions? If none, say so and say what evidence would change it.
2. **Top issues (a table of 10 to 15 rows)**, most severe first, with columns: Severity
   (critical / high / medium / low), Area, Issue (specific; cite the section or function),
   Why it matters (what could go wrong, in money or in false confidence), Fix (concrete),
   How to verify the fix (a test or a measurement).
3. **Leakage and timing audit.** For each item in the checklist below, answer *pass*,
   *fail* or *uncertain*, with one or two sentences of reasoning grounded in the reference
   or the code excerpts.
4. **Target definitions.** Are the trend, breakout, sideways, large-loss and drawdown
   targets well defined, non-degenerate and useful? Propose better definitions or thresholds
   if warranted, and say how to choose them without peeking at the final test.
5. **Statistical validity.** Comment on: folds and purging; development vs final-test design
   and its power (count the test dates); bootstrap block length; Holm correction scope and the
   total number of trials; ablation design; whether the portfolio test is realistic (costs,
   turnover, overlapping holds, T+1, next-close execution, a $1,000 sleeve).
6. **Models.** Are the model choices sensible for the data size and signal-to-noise? Is
   anything over- or under-engineered? Comment on the pooled design, ranking within date,
   missing-as-zero, the ensemble weighting, probability calibration (no isotonic/Platt step),
   the Chronos-2 stacking and contamination handling, GJR-GARCH vs EWMA. Use the inventory in section 14.
7. **Portfolio integration and risk.** Are the recommendation rules (net expected return > 0,
   evidence backed, P(large loss) <= 15%, at most 5 names, <= 25% of the sleeve each, settled
   cash only, no shorts, hold through the horizon) sound for a $1,000 Trading sleeve inside a
   Roth IRA? What would you change? How should stock picks interact with the ETF optimizer?
8. **Decision layer and AI (sections 13-14).** Is a verdict that comes only from the 20-session ranker
   (BUY = top fifth with gates; AVOID = bottom fifth) defensible given section 12's evidence, the score ties
   and the non-monotone score-to-P&L map? Are the signal warnings (lift plus top-fifth plus flat-model test)
   sound? Is the entry plan (dip quantiles, barrier exits) a reasonable use of the drawdown model? How should
   the LLM agents' ratings be evaluated against the model before either is trusted, and is the evidence
   packet (13.7) adequate for grounded LLM analysis? Should leveraged ETFs be allowed under the account rules?
9. **Data.** What free or cheap data would most improve point-in-time correctness or signal
   (point-in-time index membership, analyst revisions with timestamps, earnings calendars,
   options-implied volatility, short interest, insider transactions)? Be specific about sources
   and their point-in-time properties; say when you are not sure a source is point-in-time.
10. **Next experiments, ranked** by expected value of information per hour of work, each with
   a hypothesis, the exact test, and the decision it would change. Include whether a compact
   sequence model (PatchTST, VSN+LSTM) or a pretrained model (Chronos-2 fine-tuning, TimesFM)
   is worth testing next given the measured results, and why.
11. **Engineering.** Reproducibility, caching, failure isolation, append-only records, cost
    controls: anything fragile or missing?
12. **Questions for the owner** that would change your advice.

Rules for your answer: be concrete and cite sections (e.g. "6.2 Trend state") or function
names. Do not invent numbers; if you estimate, label it an estimate. If something cannot be
judged from what is given, say what you would need. Prefer the simplest fix that addresses
the risk. Assume the owner is technical and will implement your suggestions.

## Leakage and timing checklist (answer each in section 3)

1. SEC facts used from the day after the filing *date* (EDGAR gives no acceptance time in
   company facts). Is next-day enough? Any case where a 10-K/A or late XBRL revision leaks?
2. Restatements: the original value stands until the restatement's filing date.
3. FRED/ALFRED vintages: latest vintage published strictly before the date, from the next day.
4. Yahoo prices are **adjusted for later dividends and splits**. Returns are unaffected, but
   level-based features (distance from 52-week high, range position, moving-average gaps) and
   the raw-price reconstruction for valuation use adjusted series. Any leak or bias?
5. Universe: today's S&P 500, each stock from its "date added". Remaining bias?
6. Decision timing: features through the close of t; tradable return from the close of t+1.
   Path events measured from the close of t over t+1..t+h. Consistent?
7. Purging: training rows only where the tradable label ended before the test block; the
   inner validation split (last fifth of training dates) is also purged.
8. Cross-sectional ranking within each date: any leak from ranking against stocks that
   later entered the index, or from the requested stocks being added to the cutoff ranking?
9. Cost estimate (Abdi-Ranaldo) uses the next day's mid; the window stops a day before t.
10. The models used in the batch are fitted on all rows whose labels ended by the cutoff;
    the evidence attached to them comes from the walk-forward study. Is that evidence
    representative of the deployed model?
11. News: collected prospectively, not used in training; sentiment from FinBERT on headlines.
12. Analyst data: excluded from stock training because Yahoo's lists are today's versions.
13. Thresholds: defined in code before the study, versioned (now `thr-016cb9da59`), not tuned.
14. Verdict scores come from the daily decision fitted on rows whose labels ended by the cutoff; stocks
    outside the index are ranked against members without changing members' ranks. Any leak?
15. Fund verdicts use the ETF engine's combined forecast; its weights come from walk-forward ICs. Any
    look-ahead in choosing the weights?
16. The evidence packet cuts every series at the cutoff before computing; news needs a known publication
    time by the cutoff's end of day (New York); SEC facts count from the day after filing; FRED uses
    vintages published before the cutoff or the series' lag. Any remaining leak (e.g. a profile or analyst
    value that is today's, or first_seen after the cutoff)?
17. Agent runs analyze a past day with tools that withhold present-day profile values; is anything in
    their tool data from after the analysis day?

---

# The system (full reference)

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
| 13 | Risk model | `daily` bundle | panel | P(large loss while held) | classifier | flat across stocks today (about 7%) |
| 14 | Position engine and policies | `stocks/positions.py` | prices, dividends | trades for fixed 5/20/60 and barrier 20/60 policies | rule-based; T+1, next-close fills, no same-day exits | best development policy +210% vs SPY +358%; exploratory +5% vs SPY +43% |
| 15 | Challengers: PatchTST, Chronos-2, TimesFM 2.5, Kronos-mini | `stocks/challengers.py` | price sequences, 150 stocks | return scores | compact PatchTST trained locally; others pretrained | no skill except Kronos-mini in development (0.041), likely contaminated; 0.014 after |
| 16 | News sentiment (FinBERT) | `stocks/news.py` | stored headlines and summaries | sentiment score, relevance (about / mentions / other), events | pretrained FinBERT, local GPU or CPU | collected prospectively; not used in training |
| 17 | Moat and quality checklists | `research/moat.py`, `quality.py` | Yahoo annual statements, profile, prices, dividends, analysts | pass / check / flag per test, verdict counts | fixed rules with sector bars | untested for returns |
| 18 | Moat sources (AI reading of the 10-K) | `research/moat_ai.py` | the latest 10-K text | switching costs, network effects, cost advantages, intangibles, efficient scale, with quotes | one paid LLM call per filing, cached | qualitative |
| 19 | Market brief | `optimizer/market.py` | macro data and headlines | summary and trends | one LLM call per day, cached | qualitative |
| 20 | TradingAgents multi-agent graph | `tradingagents/graph` | tool data (prices, indicators, news, statements, macro), memory of past lessons | rating Buy..Sell with thesis | LLM agents (13 roles), one debate round each | shown as the agents' view; not used for decisions; no evaluation yet |
| 21 | Verdict and entry plan | `stocks/verdict.py`, `signals.py` | items 10-13 (stocks), item 3 (funds), item 7 (dips, breakdown) | BUY / WATCH / HOLD / SELL / AVOID; limit entries with stops and targets | fixed rules on model outputs | inherits items 10-13: experimental |

---

# Appendix: source of the critical functions

These are copied from the code at build time. Review them for leakage, off-by-one errors,
and mismatches with the reference above.

## Targets and labels (tradingagents/stocks/targets.py)

```python
@dataclass(frozen=True)
class Thresholds:
    """Every event threshold, as multiples of sigma_h. Frozen before a test period is scored:
    a study records ``version()``, and a forecast run says which version it used."""
    range_window: int = 20
    trend_k: float = 0.5
    trend_efficiency: float = 0.30
    sideways_k: float = 0.35
    sideways_range_k: float = 1.25
    loss_k: float = 1.5
    drawdown_k: float = 1.5
    contract: str = "v2"      # v2: position risks from the next-close entry; arithmetic barriers via expm1

    def version(self) -> str:
        return "thr-" + hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
```

```python
def labels(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, cash: pd.Series, spy: pd.Series,
           dates: pd.DatetimeIndex, horizon: int, th: Thresholds = Thresholds()) -> pd.DataFrame:
    """Every target for each (date, ticker) in ``dates`` x ``close.columns``; NaN where the future
    has not happened yet. ``label`` (the tradable return above cash) is what return models fit."""
    sessions = close.index
    rows = sessions.get_indexer(dates)
    if (rows < 0).any():
        raise ValueError("decision dates must be sessions of the price frame")
    tickers = list(close.columns)
    c, hi, lo = (f.to_numpy(dtype=float) for f in (close, high.reindex_like(close), low.reindex_like(close)))
    logc = np.log(c)
    n = len(sessions)
    sigma_h = ewma_vol(close).to_numpy() [rows] * np.sqrt(horizon)

    start = c[rows]
    fwd_c = _forward(c, rows, horizon)
    fwd_h = _forward(hi, rows, horizon)
    fwd_l = _forward(lo, rows, horizon)
    complete = (rows + horizon < n)[:, None]
    ended = np.where(complete, 1.0, np.nan)

    r = (np.log(fwd_c[:, -1, :]) - np.log(start)) * ended
    daily = np.diff(np.concatenate([np.log(start)[:, None, :], np.log(fwd_c)], axis=1), axis=1)
    efficiency = np.abs(r) / np.nansum(np.abs(daily), axis=1)
    top, bottom = np.nanmax(fwd_h, axis=1) * ended, np.nanmin(fwd_l, axis=1) * ended
    drawdown = bottom / start - 1
    path_range = (top - bottom) / start

    w = th.range_window
    upper = high.rolling(w, min_periods=w).max().to_numpy()[rows]
    lower = low.rolling(w, min_periods=w).min().to_numpy()[rows]
    above = fwd_h > upper[:, None, :]
    below = fwd_l < lower[:, None, :]
    up_break = above.any(axis=1)
    down_break = below.any(axis=1)
    first_up = np.where(up_break, above.argmax(axis=1), horizon)
    first_down = np.where(down_break, below.argmax(axis=1), horizon)
    both_same_bar = up_break & down_break & (first_up == first_down)
    end_close = fwd_c[:, -1, :]

    # The session (1 to h) on which the drawdown first went past its threshold: an alarm's lead time.
    # A fall of k sigma in log terms is an arithmetic return of expm1(-k sigma).
    dd_line = np.expm1(-th.drawdown_k * sigma_h)
    breached = (fwd_l / start[:, None, :] - 1) <= dd_line[:, None, :]
    dd_day = np.where(breached.any(axis=1), breached.argmax(axis=1) + 1.0, np.nan)

    big = th.trend_k * sigma_h
    up = (r >= big) & (efficiency >= th.trend_efficiency)
    down = (r <= -big) & (efficiency >= th.trend_efficiency)
    sideways = (np.abs(r) <= th.sideways_k * sigma_h) & (path_range <= th.sideways_range_k * sigma_h)
    state = np.full(r.shape, 3.0)                         # unclear
    state[sideways] = 2.0
    state[down] = 1.0
    state[up] = 0.0

    # Tradable: enter at the next close, hold h sessions; above cash and above SPY.
    entry = np.minimum(rows + 1, n - 1)
    exit_ = np.minimum(rows + 1 + horizon, n - 1)
    tradable_ok = (rows + 1 + horizon < n)[:, None]
    stock_ret = (logc[exit_] - logc[entry]) * np.where(tradable_ok, 1.0, np.nan)
    lc = np.log(cash.reindex(sessions).ffill().to_numpy())
    ls = np.log(spy.reindex(sessions).ffill().to_numpy())
    cash_ret = lc[exit_] - lc[entry]          # missing before the cash fund existed: no label, not zero
    spy_ret = ls[exit_] - ls[entry]

    # Position risks, on the actual holding: entry at close t+1, adverse excursion over the lows of
    # t+2..t+1+h relative to the entry close, never above zero; terminal on the exit close.
    entry_px = c[entry]
    later_lows = _forward(lo, entry, horizon)              # lows of entry+1 .. entry+h
    mae = np.minimum(np.nanmin(later_lows, axis=1) / entry_px - 1, 0.0) * np.where(tradable_ok, 1.0, np.nan)
    pos_ret = np.expm1(stock_ret)

    def frame(values, mask_to=None):
        v = np.asarray(values, dtype=float)
        if mask_to is not None:
            v = np.where(np.isnan(mask_to), np.nan, v)
        return pd.DataFrame(v, index=dates, columns=tickers).stack(future_stack=True)

    out = pd.DataFrame({
        "label": frame(stock_ret - cash_ret[:, None]),
        "label_rel": frame(stock_ret - spy_ret[:, None]),
        "ret_raw": frame(r),
        "drawdown": frame(drawdown),
        "sigma_h": frame(sigma_h),
        "upper": frame(upper), "lower": frame(lower),
        "up_break": frame(up_break, r), "down_break": frame(down_break, r),
        "failed_up": frame(up_break & (end_close <= upper), r),
        "failed_down": frame(down_break & (end_close >= lower), r),
        "both_same_bar": frame(both_same_bar, r),
        "large_loss": frame(r <= -th.loss_k * sigma_h, r),
        "deep_drawdown": frame(drawdown <= dd_line, r),
        "mae_position": frame(mae),
        "pos_large_loss": frame(pos_ret <= np.expm1(-th.loss_k * sigma_h), stock_ret),
        "pos_deep_drawdown": frame(mae <= dd_line, mae),
        "dd_breach_day": frame(dd_day, r),
        "trend_state": frame(state, r),
    })
    # Where sigma_h is unknown (too little history), no threshold applies: no event label.
    unknown = out["sigma_h"].isna()
    out.loc[unknown, ["large_loss", "deep_drawdown", "trend_state", "pos_large_loss", "pos_deep_drawdown"]] = np.nan
    label_end = pd.Series(sessions[np.minimum(rows + 1 + horizon, n - 1)], index=dates)
    label_end[~tradable_ok[:, 0]] = pd.NaT
    out["label_end"] = label_end.reindex(out.index.get_level_values(0)).to_numpy()
    out.index.names = ["date", "ticker"]
    return out
```

```python
def current_trend(close: pd.DataFrame) -> pd.DataFrame:
    """The trend already in place at each close (detection, not forecast): +1 when the price is
    above a rising 50-day average that is above the 200-day, -1 for the mirror image, else 0."""
    ma50, ma200 = close.rolling(50, min_periods=40).mean(), close.rolling(200, min_periods=150).mean()
    rising = ma50.diff(10) > 0
    up = (close > ma50) & rising & (ma50 > ma200)
    down = (close < ma50) & ~rising & (ma50 < ma200)
    return up.astype(float) - down.astype(float)
```

## Point-in-time fundamentals (tradingagents/stocks/fundamentals.py)

```python
def _features(known: dict, splits: list | None) -> dict:
    """The features from what is on file: ``known[(concept, kind)] = {period end: value}``, each
    period holding its latest filing so far."""
    get = lambda c, k: known.get((c, k), {})  # noqa: E731
    out = {k: np.nan for k in FEATURES + RAW}
    rev_y = get("revenue", "annual")
    end_y, revenue = _latest(rev_y)
    if end_y:
        out["revenue_y"] = revenue
        out["rev_growth_y"] = _ratio(revenue, _year_before(rev_y, end_y)) - 1
        gross = get("gross_profit", "annual").get(end_y)
        if gross is None and get("cost_of_revenue", "annual").get(end_y) is not None:
            gross = revenue - get("cost_of_revenue", "annual")[end_y]   # revenue less cost of revenue
        out["gross_margin"] = _ratio(gross, revenue)
        out["op_margin"] = _ratio(get("op_income", "annual").get(end_y), revenue)
        out["net_margin"] = _ratio(get("net_income", "annual").get(end_y), revenue)
        ocf, capex = get("ocf", "annual").get(end_y), get("capex", "annual").get(end_y)
        if ocf is not None and capex is not None:     # a missing capex is unknown, not zero
            fcf = ocf - capex
            out["fcf_y"], out["fcf_margin"] = fcf, _ratio(fcf, revenue)
        out["net_income_y"] = get("net_income", "annual").get(end_y, np.nan)
    end_q, rq = _latest(get("revenue", "quarter"))
    if end_q:
        out["rev_growth_q"] = _ratio(rq, _year_before(get("revenue", "quarter"), end_q)) - 1
    end_a, a = _latest(get("assets", "instant"))
    equity = get("equity", "instant")
    if end_a:
        liabilities = get("liabilities", "instant").get(end_a)
        if liabilities is None and equity.get(end_a) is not None:
            liabilities = a - equity[end_a]                      # assets less equity
        out["liabilities_to_assets"] = _ratio(liabilities, a)
        if end_y:
            out["roa"] = _ratio(out["net_income_y"], a)
    out["equity_i"] = _latest(equity)[1]
    shares = get("shares", "instant")
    end_s, sh = _latest(shares)
    if end_s:
        out["shares_i"] = sh
        out["shares_end_ord"] = float(_day(end_s).toordinal())   # the date the count is as of
        before = _year_before(shares, end_s, slack=60)
        if before is not None:
            prior_end = (_day(end_s) - timedelta(days=365)).isoformat()
            out["dilution_1y"] = _ratio(sh, before * _split_factor(splits, prior_end, end_s)) - 1
    return {k: (np.nan if v is None else float(v)) for k, v in out.items()}
```

```python
def history(facts: dict, dates: pd.DatetimeIndex, splits: list | None = None) -> pd.DataFrame:
    """Features at each date, from the filings made strictly before it.

    The known set changes only on filing days, so it is updated one filing day at a
    time and the features recomputed then; dates in between carry them forward.
    """
    events = _events(facts)
    if not events:
        return pd.DataFrame(index=dates, columns=FEATURES + RAW, dtype=float)
    known, states, days = {}, [], []
    i = 0
    while i < len(events):
        day = events[i][0]
        while i < len(events) and events[i][0] == day:
            _, concept, kind, end, value = events[i]
            known.setdefault((concept, kind), {})[end] = value
            i += 1
        states.append(_features(known, splits))
        days.append(day)
    # Usable from the day after filing: a decision on the filing date itself does not see it.
    usable = pd.to_datetime(days) + pd.Timedelta(days=1)
    frame = pd.DataFrame(states, index=usable)
    out = frame.reindex(frame.index.union(dates)).ffill().reindex(dates)
    seen = pd.Series(usable, index=usable)
    seen = seen.reindex(seen.index.union(dates)).ffill().reindex(dates)
    out["filing_age_days"] = (dates - pd.DatetimeIndex(seen)).days + 1
    return out
```

```python
def valuation(fund: pd.DataFrame, raw_price: pd.Series, splits: list | None = None) -> pd.DataFrame:
    """Yields on the market value at each date: the price as it then traded times the share count
    reported, carried to that date's split basis (a split between the count's date and the
    valuation date multiplies the count), so price and shares always match."""
    shares = fund["shares_i"].copy()
    for i, (d, ordv) in enumerate(zip(fund.index, fund.get("shares_end_ord", pd.Series(np.nan, index=fund.index)),
                                      strict=True)):
        if np.isfinite(ordv):
            shares.iloc[i] *= _split_factor(splits, date.fromordinal(int(ordv)).isoformat(), d.date().isoformat())
    cap = raw_price.reindex(fund.index) * shares
    cap = cap.where(cap > 0)
    return pd.DataFrame({"earnings_yield": fund["net_income_y"] / cap, "sales_yield": fund["revenue_y"] / cap,
                         "fcf_yield": fund["fcf_y"] / cap, "book_yield": fund["equity_i"] / cap}, index=fund.index)
```

## Economic vintages (tradingagents/stocks/macro.py)

```python
def as_known(rows: list[dict], dates: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Series]:
    """For each date, the series as published strictly before it: each observation's latest vintage
    whose publication date is earlier than the date."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {d: pd.Series(dtype=float) for d in dates}
    frame["date"], frame["from"] = pd.to_datetime(frame["date"]), pd.to_datetime(frame["from"])
    frame = frame.sort_values("from")
    out = {}
    for d in dates:
        seen = frame[frame["from"] < d]
        out[d] = seen.groupby("date")["value"].last().sort_index()
    return out
```

## Cost estimate and panel (tradingagents/stocks/panel.py)

```python
def spread_cost(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Round-trip cost estimate at each close: the Abdi-Ranaldo spread over the previous 21 sessions
    (each term uses the next day's mid, so the window stops a day before the date), plus slippage."""
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2
    term = 4 * (c - eta) * (c - eta.shift(-1))
    s2 = term.shift(1).rolling(21, min_periods=15).mean().clip(lower=0)
    return (np.sqrt(s2) + 2 * SLIPPAGE).clip(lower=2 * SLIPPAGE, upper=0.05)
```

```python
def build(data: StockData, horizon: int, dates: pd.DatetimeIndex | None = None, th: Thresholds = Thresholds(),
          groups=DEFAULT_GROUPS, start: str = "2006-01-01", macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """The panel for one horizon. ``macro``: precomputed economic features (shared across horizons)."""
    tickers = sorted(data.stocks)          # a fixed order: results never depend on input order
    sessions = data.close.index
    if dates is None:
        dates = decision_dates(sessions, start=start, every=every_for(horizon)).union(pd.DatetimeIndex([sessions[-1]]))
    # Point-in-time membership is applied before any cross-sectional ranking: ranks compare a stock
    # only with that date's members, and requested non-members never move a member's rank.
    member = member_mask(data, dates, tickers)
    frames = []
    if "technical" in groups:
        for name, frame in _technical(data, tickers, dates).items():
            if name not in GROUPS["technical"] and name != "vol_3m":
                continue
            sampled = frame.reindex(dates)
            frames.append((sampled if name in UNRANKED else rank_against(sampled, member))
                          .stack(future_stack=True).rename(name))
    if "fundamental" in groups:
        for name, frame in _fundamental(data, tickers, dates).items():
            frames.append(rank_against(frame, member).stack(future_stack=True).rename(name))
    panel = pd.concat(frames, axis=1) if frames else pd.DataFrame(
        index=pd.MultiIndex.from_product([dates, tickers]))
    panel.index.names = ["date", "ticker"]
    idx = panel.index
    level = lambda frame: frame.reindex(dates).reindex(idx.get_level_values(0)).set_index(idx)  # noqa: E731
    if "market" in groups:
        market = {m: data.close[m].dropna() for m in ("SPY", "^VIX", "^VIX3M", "^IRX", "^TNX") if m in data.close}
        panel = panel.join(level(market_features(market).reindex(sessions).ffill()))
    if "macro" in groups and macro is not None:
        panel = panel.join(level(macro.reindex(dates)))

    close, high, low = data.close[tickers], data.high[tickers], data.low[tickers]
    raw_vol = np.log(close).diff().rolling(63, min_periods=40).std() * np.sqrt(252)
    panel["raw_vol_3m"] = raw_vol.reindex(dates).stack(future_stack=True).reindex(idx).to_numpy()
    panel["ewma_vol_daily"] = ewma_vol(close).reindex(dates).stack(future_stack=True).reindex(idx).to_numpy()
    panel["cost_rt"] = spread_cost(close, high, low).reindex(dates).stack(future_stack=True).reindex(idx).to_numpy()
    cash = data.close[CASH]
    cash_log = np.log(cash).diff()
    panel["cash_h"] = (cash_log.rolling(21, min_periods=10).mean() * horizon).reindex(dates).fillna(0.0) \
        .reindex(idx.get_level_values(0)).to_numpy()
    targets = labels(close, high, low, cash, data.close["SPY"], dates, horizon, th)
    panel = panel.join(targets)
    # ``in_index``: a member of the reference universe on that date. Only these rows train or score
    # models. Requested stocks outside it are forecast, never trained on, so asking about a stock
    # cannot change the models or anyone else's forecast.
    panel["in_index"] = member.stack(future_stack=True).reindex(idx).fillna(False).to_numpy(dtype=bool)
    panel["requested"] = idx.get_level_values(1).isin(data.requested)
    panel["removed"] = idx.get_level_values(1).isin([t for t, i in data.info.items() if i.get("removed")])
    # A row needs a price on its date and a year of history.
    have = close.reindex(dates).stack(future_stack=True).reindex(idx).notna().to_numpy()
    year = raw_vol.reindex(dates).stack(future_stack=True).reindex(idx).notna().to_numpy() & \
        close.shift(252).reindex(dates).stack(future_stack=True).reindex(idx).notna().to_numpy()
    return panel[have & year]
```

## Walk-forward split (tradingagents/research/walkforward.py)

```python
def walk_forward(panel: pd.DataFrame, factories, features: list[str], horizon: int, first_test: str,
                 block: str = "12MS", universe: str = "", min_train_dates: int = 36, cost_bps: float = 10.0,
                 progress=None, should_stop=None) -> WalkForwardResult:
    """Refit every block from ``first_test`` on, and score each model's test forecasts.

    ``factories`` make fresh mean models (wrapped in ``Probabilistic`` here).
    ``should_stop()`` is checked between blocks, so a long run can be cancelled.
    """
    started = time.time()
    panel = panel.sort_index(level="date")
    dates = panel.index.get_level_values("date")
    has_label = panel["label"].notna().to_numpy()
    test_dates = np.sort(dates[(dates >= pd.Timestamp(first_test)) & has_label].unique())
    if not len(test_dates):
        raise ValueError("no test dates with outcomes after " + first_test)
    edges = sorted({pd.Timestamp(test_dates[0]), *pd.date_range(pd.Timestamp(test_dates[0]),
                                                                 pd.Timestamp(test_dates[-1]), freq=block)})
    edges.append(pd.Timestamp(test_dates[-1]) + pd.Timedelta(days=1))
    chunks, log = [], []
    for n, (start, end) in enumerate(zip(edges, edges[1:], strict=False), 1):
        if should_stop and should_stop():
            raise InterruptedError("walk-forward cancelled")
        test = panel[(dates >= start) & (dates < end) & has_label]
        train = panel[(pd.to_datetime(panel["label_end"]) < start) & has_label]
        if test.empty or train.index.get_level_values("date").nunique() < min_train_dates:
            continue
        if progress:
            progress(n, len(edges) - 1, start)
        out = test[["label", "cash_h"]].copy()
        for make in factories:
            model = Probabilistic(make)
            t0 = time.time()
            model.fit(train, features, horizon)
            got = model.predict(test)
            for c in COLUMNS:
                out[f"{model.name}:{c}"] = got[c].to_numpy()
            log.append({"block": start.strftime("%Y-%m-%d"), "model": model.name, "train_rows": len(train),
                        "train_dates": int(train.index.get_level_values("date").nunique()),
                        "train_last_label_end": str(pd.to_datetime(train["label_end"]).max().date()),
                        "seconds": round(time.time() - t0, 3), "fit": model.describe()})
        chunks.append(out)
    pred = pd.concat(chunks)
    models = sorted({c.split(":")[0] for c in pred.columns if ":" in c})
    scores = {m: score(pred, m, cost_bps) for m in models}
    adjusted = holm({m: s.p_value for m, s in scores.items()})
    for m, s in scores.items():
        s.p_value = adjusted[m]
    by_period = {}
    pdates = pred.index.get_level_values("date")
    for a, b, name in (("1990", "2012-12-31", "to 2012"), ("2013", "2019-12-31", "2013-19"), ("2020", "2100", "2020 on")):
        part = pred[(pdates >= pd.Timestamp(a)) & (pdates <= pd.Timestamp(b))]
        if part.index.get_level_values("date").nunique() >= 12:
            by_period[name] = {m: float(_ic_by_date(part[["label"]].assign(mean=part[f"{m}:mean"]), "mean").mean())
                               for m in models}
    return WalkForwardResult(horizon, universe, str(pd.Timestamp(test_dates[0]).date()),
                             str(pd.Timestamp(test_dates[-1]).date()), pred, scores, by_period, log,
                             time.time() - started)
```

## Recommendation sizing (tradingagents/stocks/ideas.py)

```python
def make(con, store, cache_dir, run_id: str | None = None, horizon: int = 20, as_of: date | None = None) -> dict:
    """Build and save the recommendation for a run; see the module docstring."""
    from tradingagents.ledger.db import now_utc, transaction

    from .pipeline import load_run

    rules = store.get("global", "stock_rules")
    as_of = as_of or date.today()
    run = load_run(con, run_id)
    if run is None:
        raise ValueError("no stock run yet: run a batch first")
    sleeve = rules["sleeve"]
    state = store.ledger.state(sleeve, as_of)
    rows = [f for f in run["forecasts"] if f["horizon"] == horizon]
    tickers = [f["ticker"] for f in rows]
    # Prices as of the decision date, never later ones; a price over a week old is flagged below.
    known = _prices(cache_dir, [*tickers, *state.positions]).ffill().loc[:pd.Timestamp(as_of)]
    last = known.iloc[-1] if len(known) else pd.Series(dtype=float)
    price_day = known.index[-1].date() if len(known) else None
    held_value = sum(q * float(last.get(t, 0) or 0) for t, q in state.positions.items())
    sleeve_value = held_value + state.cash.total
    available = state.cash.available
    lines, buys = [], []
    for f in rows:
        fc = f.get("forecast") or {}
        t = f["ticker"]
        held = state.positions.get(t, 0.0)
        line = {"ticker": t, "name": f.get("name"), "held": held, "price": float(last.get(t, np.nan)),
                "net": fc.get("net"), "expected": fc.get("mean"), "cost_rt": fc.get("cost_rt"),
                "p_large_loss": fc.get("p_large_loss"), "evidence": (f.get("evidence_status") or {}).get("return"),
                "reasons": list(f.get("reasons") or [])}
        flagged = f["status"] != "ok" or any(r.startswith(("stale", "insufficient", "no trading")) for r in line["reasons"])
        if price_day is None or (as_of - price_day).days > 7:
            flagged = True
            line["reasons"].append(f"price as of {price_day}: too old for {as_of}")
        risk = fc.get("p_large_loss")
        risk_ok = isinstance(risk, (int, float)) and np.isfinite(risk) and 0 <= risk <= 1
        if f["status"] != "ok" or fc.get("net") is None:
            line["action"] = "abstain"
        elif not risk_ok and fc["net"] > 0:
            # A missing or invalid risk estimate blocks approval; it is never read as zero risk.
            line["action"] = "watch (risk estimate missing or invalid: approval blocked)"
        elif fc["net"] <= 0:
            line["action"] = "consider reducing (no short orders)" if held > 0 else "avoid (no short orders)"
        elif line["evidence"] == "backed" and risk <= rules["max_large_loss_prob"] and not flagged:
            line["action"] = "buy candidate"
            buys.append(line)
        else:
            why = "return evidence not backed by the final test" if line["evidence"] != "backed" else \
                "large-loss probability above the cap" if risk > rules["max_large_loss_prob"] \
                else "data-quality flag"
            line["action"] = f"watch ({why})"
        lines.append(line)
    # The name limit counts what Trading already holds, not only new buys.
    held_names = {t for t, q in state.positions.items() if q > 1e-9}
    new_slots = max(0, rules["max_names"] - len(held_names))
    ranked = sorted(buys, key=lambda x: -x["net"])
    buys = [b for b in ranked if b["ticker"] in held_names] + [b for b in ranked if b["ticker"] not in held_names][:new_slots]
    for b in ranked:
        if b not in buys:
            b["action"] = f"watch (name limit: {rules['max_names']} names including holdings)"
    sizes = {}
    if buys:
        per = min(rules["max_position_share"] * sleeve_value, available / len(buys))
        for line in buys:
            have = line["held"] * (line["price"] if np.isfinite(line["price"]) else 0)
            dollars = max(0.0, per - have)
            if dollars < rules["min_order"] or not np.isfinite(line["price"]):
                line["order"] = None
                line["action"] += f" (no order: under ${rules['min_order']:.0f} or already at target)"
                continue
            line["order"] = {"side": "buy", "dollars": round(dollars, 2), "shares": round(dollars / line["price"], 6)}
            sizes[line["ticker"]] = dollars
    from tradingagents.ledger.sessions import next_session

    execute = next_session(date.fromisoformat(run["info_cutoff"]))
    payload = {"run_id": run["id"], "info_cutoff": run["info_cutoff"], "earliest_execution": str(execute),
               "hold_until": str(next_session(execute, horizon)), "sleeve": sleeve, "sleeve_value": round(sleeve_value, 2),
               "settled_cash": round(available, 2), "rules": rules, "lines": lines,
               "exposure": exposure(store, cache_dir, list(sizes), sizes, as_of),
               "note": ("Research, not orders: nothing is sent to the broker. Buys need settled cash; hold through the "
                        "horizon (selling the day you buy would be a day trade). Negative forecasts never become short orders.")}
    idea_id = uuid.uuid4().hex
    with transaction(con):
        con.execute("INSERT INTO stock_ideas VALUES (?, ?, ?, ?, ?, ?)",
                    (idea_id, run["id"], now_utc(), sleeve, horizon, json.dumps(payload, default=str)))
    return {"id": idea_id, **payload}
```

## Position engine (tradingagents/stocks/positions.py)

```python
@dataclass(frozen=True)
class Policy:
    """An exit policy. ``kind``: 'fixed' (exit at the deadline only) or 'barrier' (close-confirmed
    profit and risk levels, with the deadline). Levels are multiples of sigma_H, the daily EWMA
    volatility at the decision times sqrt(H), turned into arithmetic levels with expm1."""
    name: str = "barrier_20"
    kind: str = "barrier"
    horizon: int = 20               # H: the maximum holding period, in sessions
    profit_k: float = 1.0
    risk_k: float = 1.0

    def version(self) -> str:
        return "pol-" + hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def levels(self, sigma_daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(profit level, risk level) as arithmetic returns from the entry close."""
        s = np.asarray(sigma_daily, dtype=float) * np.sqrt(self.horizon)
        if self.kind == "fixed":
            return np.full(s.shape, np.inf), np.full(s.shape, -np.inf)
        return np.expm1(self.profit_k * s), np.expm1(-self.risk_k * s)
```

```python
def trades(px: pd.DataFrame, low_adj: pd.DataFrame, close_adj: pd.DataFrame, div: pd.DataFrame,
           decisions: pd.DataFrame, policy: Policy, cost_rt: np.ndarray | None = None) -> pd.DataFrame:
    """The outcome of entering each decision row under ``policy`` (vectorized).

    ``decisions``: rows with columns ``date`` (decision close t), ``ticker`` and ``sigma`` (daily
    EWMA volatility at t). Returns entry and exit dates, the exit reason, holding sessions, gross
    and net arithmetic return (price change plus dividends, less ``cost_rt``), the adverse
    excursion during the position, and whether the outcome is complete.
    """
    sessions = px.index
    cols = px.columns.get_indexer(decisions["ticker"])
    t_idx = sessions.get_indexer(pd.to_datetime(decisions["date"]))
    if (cols < 0).any() or (t_idx < 0).any():
        raise ValueError("every decision needs its ticker and date in the price frame")
    H = policy.horizon
    P, L, C, D = (f.to_numpy(dtype=float) for f in (px, low_adj, close_adj, div))
    entry = t_idx + 1
    n = len(sessions)
    entry_px = np.where(entry < n, P[np.minimum(entry, n - 1), cols], np.nan)
    # Path from the session after entry: closes (for decisions), lows (excursion), dividends.
    fwd_px = _forward(P, entry, cols, H)                       # sessions entry+1 .. entry+H
    fwd_div = _forward(D, entry, cols, H)
    # Lows on the executable basis: the adjusted low times that day's executable/adjusted ratio.
    ratio = P / C
    fwd_low = _forward(L * ratio, entry, cols, H)
    profit, risk = policy.levels(decisions["sigma"].to_numpy(dtype=float))
    ret = fwd_px / entry_px[:, None] - 1
    hit_t = ret >= profit[:, None]
    hit_s = ret <= risk[:, None]
    # A close-confirmed trigger at session k (1-based after entry) exits at k+1; the deadline exit is
    # at H whatever happens. A trigger at H-1 or H therefore exits at H.
    first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1) + 1, H + 1)
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1) + 1, H + 1)
    trigger = np.minimum(first_t, first_s)
    exit_k = np.minimum(trigger + 1, H)                         # sessions after entry
    reason = np.where(trigger + 1 <= H, np.where(first_s <= first_t, "stop", "target"), "deadline")
    rows = np.arange(len(decisions))
    exit_px = fwd_px[rows, exit_k - 1]
    held = np.arange(1, H + 1)[None, :] <= exit_k[:, None]
    divs = np.nansum(np.where(held, fwd_div, 0.0), axis=1)
    lows = np.where(held, fwd_low, np.nan)
    with np.errstate(all="ignore"):
        mae = np.minimum(np.nanmin(lows, axis=1) / entry_px - 1, 0.0)
        gross = (exit_px + divs) / entry_px - 1
    complete = np.isfinite(entry_px) & np.isfinite(exit_px) & (entry + exit_k < n)
    # A path with a missing price before the exit (a delisting, a halt) has no known outcome.
    gap = np.isnan(np.where(held, fwd_px, 0.0)).any(axis=1)
    complete &= ~gap
    cost = np.zeros(len(decisions)) if cost_rt is None else np.asarray(cost_rt, dtype=float)
    entry_dates = np.where(entry < n, sessions[np.minimum(entry, n - 1)], pd.NaT)
    exit_dates = np.where(entry + exit_k < n, sessions[np.minimum(entry + exit_k, n - 1)], pd.NaT)
    out = pd.DataFrame({
        "date": pd.to_datetime(decisions["date"]).to_numpy(), "ticker": decisions["ticker"].to_numpy(),
        "entry_date": entry_dates, "exit_date": exit_dates,
        "reason": np.where(complete, reason, "incomplete"), "held_sessions": np.where(complete, exit_k, np.nan),
        "entry_px": entry_px, "exit_px": np.where(complete, exit_px, np.nan),
        "dividends": np.where(complete, divs, np.nan),
        "gross": np.where(complete, gross, np.nan), "net": np.where(complete, gross - cost, np.nan),
        "mae": np.where(complete, mae, np.nan), "profit_level": profit, "risk_level": risk,
        "cost_rt": cost, "complete": complete})
    out["policy"] = policy.name
    return out
```

## Daily decision gates and exits (tradingagents/stocks/daily.py)

```python
def evidence(con, policy_name: str, features: list[str], thresholds: Thresholds, r: dict) -> dict:
    """Whether the policy's study passed on development, and whether the prospective record confirms it."""
    found = None
    for row in con.execute("SELECT id, config_json, scores_json FROM research_trials WHERE name = 'decision_study:h20'"
                           " ORDER BY id DESC"):
        cfg = json.loads(row["config_json"] or "{}")
        if cfg.get("smoke") is False and cfg.get("thresholds_version") == thresholds.version() \
                and cfg.get("features") == features and policy_name in (cfg.get("policies") or {}):
            found = row
            break
    if found is None:
        return {"trial": None, "development": "untested", "prospective": "none", "validated": False,
                "note": "no full decision study for this policy, thresholds and features"}
    s = json.loads(found["scores_json"])
    sim = (s.get("simulation") or {}).get(policy_name) or {}
    dev = sim.get("development:equal") or {}
    q = ((s.get("quality") or {}).get(policy_name) or {}).get("development") or {}
    checks = {
        "beats_spy_after_costs": bool((dev.get("vs_spy_ci") or [None])[0] is not None and dev["vs_spy_ci"][0] > 0),
        "beats_cash": bool(dev.get("total", -1) > dev.get("cash_total", 0)),
        "drawdown_ok": bool(dev.get("max_drawdown", -1) >= r["max_drawdown_limit"]),
        "calibration_ok": bool(q and (q.get("quality_logit_sig") or {}).get("log_loss", 9) <= q.get("log_loss_base", 0)),
    }
    passed = all(checks.values())
    shadow = con.execute("SELECT state_json FROM shadow_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    pros = json.loads(shadow["state_json"]).get("record", {}) if shadow else {}
    confirmed = passed and pros.get("round_trips", 0) >= r["prospective_min_trades"] and pros.get("excess_vs_spy", -1) > 0
    return {"trial": found["id"], "development": "passed" if passed else "failed", "checks": checks,
            "prospective": pros, "validated": bool(confirmed),
            "chosen_rule": sim.get("chosen")}
```

```python
def review_lot(lot: dict, cutoff: pd.Timestamp, sessions: pd.DatetimeIndex, px: pd.Series, sigma_at_entry: float,
               policy, sleeve_value: float, r: dict, today: pd.DataFrame) -> dict:
    """HOLD, REDUCE, EXIT or REVIEW for one lot, close-confirmed at the cutoff, executed the next close."""
    from tradingagents.ledger.sessions import next_session

    entry = pd.Timestamp(lot["entry_date"])
    t = lot["ticker"]
    price = float(px.get(t, np.nan))
    nxt = pd.Timestamp(next_session(cutoff.date()))
    entry_i = sessions.searchsorted(entry)
    deadline = sessions[min(entry_i + policy.horizon, len(sessions) - 1)] if entry_i + policy.horizon < len(sessions) \
        else pd.Timestamp(next_session(entry.date(), policy.horizon))
    profit, risk = policy.levels(np.array([sigma_at_entry if np.isfinite(sigma_at_entry) else 0.02]))
    out = {"ticker": t, "shares": lot["shares"], "entry_date": str(entry.date()), "entry_px": lot["entry_px"],
           "deadline": str(deadline.date()), "price": price, "profit_level": float(profit[0]), "risk_level": float(risk[0]),
           "age_sessions": int(((sessions > entry) & (sessions <= cutoff)).sum()), "reasons": []}
    if not np.isfinite(price) or not lot.get("entry_px"):
        return {**out, "action": "REVIEW", "reasons": ["no current price or no entry price: a person decides"]}
    ret = price / lot["entry_px"] - 1
    value = lot["shares"] * price
    out.update({"unrealized": ret, "value": value, "share_of_sleeve": value / sleeve_value if sleeve_value else None})
    if entry >= cutoff:
        return {**out, "action": "HOLD", "reasons": ["bought this session: no same-day exit"]}
    if nxt >= deadline:
        return {**out, "action": "EXIT", "execute": str(nxt.date()), "reasons": ["holding deadline reached (no extension)"]}
    if ret <= risk[0]:
        return {**out, "action": "EXIT", "execute": str(nxt.date()),
                "reasons": [f"close-confirmed risk level: {ret:+.1%} at or below {risk[0]:+.1%}; the next close may gap further"]}
    if ret >= profit[0]:
        return {**out, "action": "EXIT", "execute": str(nxt.date()),
                "reasons": [f"close-confirmed profit level: {ret:+.1%} at or above {profit[0]:+.1%}"]}
    cap = r["max_position_share"] * sleeve_value
    if value > cap * 1.05:
        fraction = 1 - cap / value
        return {**out, "action": "REDUCE", "fraction": fraction, "execute": str(nxt.date()),
                "reasons": [f"{value / sleeve_value:.0%} of the sleeve, above the {r['max_position_share']:.0%} cap"]}
    if t in today.index:
        out["continuation_info"] = {"expected_net_new_entry": float(today.loc[t, "expected_net"]),
                                    "note": "the ML continuation exit is information only until validated"}
    return {**out, "action": "HOLD", "reasons": ["within its levels and deadline"]}
```

```python
def entries(today: pd.DataFrame, held: set, exiting: set, r: dict, sleeve_value: float, spendable: float,
            ira: dict, prices: pd.Series, validated: bool, cutoff, execute) -> list[dict]:
    """ENTER (validated) or PAPER ENTER (experimental) and WAIT, with sizes and blocked constraints."""
    out = []
    # The model's own order: its score, then P(profit), then the ticker (the same order as the saved ranks),
    # so the input order never decides which tied stock comes first. Not the expected net: that comes from
    # a binned map that need not rise with the score, and would put lower-ranked stocks first. Ties of the
    # expected net are still marked, since tied stocks share one estimate.
    cand = today[today["in_index"]].reset_index().rename(columns={"index": "ticker"})
    cand = cand.sort_values(["score_pct", "p_win", "ticker"], ascending=[False, False, True])
    ties = cand.groupby("expected_net")["ticker"].transform("count")
    cand = cand.assign(tied_with=ties - 1).set_index("ticker")
    active = (held - exiting)
    slots = max(0, r["max_names"] - len(active))
    budget = spendable
    for t, row in cand.head(25).iterrows():
        # Model gates decide whether the signal is good enough at all (they also decide PAPER ENTER, which
        # the shadow account sizes on its own cash); account gates decide whether this account can act.
        model_blocks, account_blocks = [], []
        risk = row["p_risk"]
        if not (isinstance(risk, float) and np.isfinite(risk)):
            model_blocks.append("risk estimate missing or invalid")
        elif risk > r["max_large_loss_prob"]:
            model_blocks.append(f"P(large loss during the position) {risk:.0%} above {r['max_large_loss_prob']:.0%}")
        if not np.isfinite(row["expected_net"]) or row["expected_net"] < r["min_expected_net"]:
            model_blocks.append(f"expected net P&L {row['expected_net']:+.2%} below {r['min_expected_net']:+.2%}")
        if not np.isfinite(row["p_win"]) or row["p_win"] < r["min_p_win"]:
            model_blocks.append(f"P(profitable after costs) {row['p_win']:.0%} below {r['min_p_win']:.0%}")
        price = float(prices.get(t, np.nan))
        if not np.isfinite(price):
            model_blocks.append("no current price")
        if t in held:
            account_blocks.append("already held")
        dollars = min(r["max_position_share"] * sleeve_value, budget / max(slots, 1)) if slots else 0.0
        if not slots:
            account_blocks.append(f"name limit: {r['max_names']} names including holdings")
        if dollars < r["min_order"]:
            account_blocks.append(f"settled cash: ${budget:,.2f} available after reservations")
        ira_total = ira["total"] or 1.0
        if (ira["by_ticker"].get(t, 0.0) + dollars) / ira_total > r["combined_name_cap"]:
            account_blocks.append(f"IRA concentration: would exceed {r['combined_name_cap']:.0%} of the whole IRA in {t}")
        line = {"ticker": t, "expected_net": row["expected_net"], "net_sd": row["net_sd"], "net_q10": row["net_q10"],
                "expected_excess_vs_spy": row["expected_excess"], "p_win": row["p_win"], "p_risk": risk,
                "cost_rt": row["cost_rt"], "score_pct": row["score_pct"], "sigma": row["sigma"], "price": price,
                "tied_with": int(row["tied_with"]), "blocked": model_blocks + account_blocks, "execute": str(execute)}
        if not model_blocks:
            line["paper_action"] = "PAPER ENTER"           # the shadow account sizes it on its own cash
        if model_blocks or account_blocks:
            line["action"] = "WAIT"
        else:
            line.update({"dollars": round(dollars, 2), "shares": round(dollars / price, 6),
                         "expected_benefit_usd": round(dollars * row["expected_net"], 2),
                         "action": "ENTER" if validated else "WAIT"})
            if not validated:
                line["blocked"] = ["evidence: not validated (experimental mode shows PAPER ENTER)"]
            budget -= dollars
            slots -= 1
        out.append(line)
    return out
```

## Capital settings (tradingagents/ledger/capital.py)

```python
def reallocation(store, as_of, prices: dict[str, float] | None = None) -> dict:
    """A reviewable proposal to bring each sleeve's net funding to its capital target.

    Only settled, unreserved cash can move, and only what the source sleeve really has: the
    proposal never creates money. It changes nothing until a transfer is recorded.
    """
    cap = load(store)
    out = {"as_of": str(as_of), "targets": asdict(cap), "sleeves": {}, "transfers": [], "notes": []}
    for sleeve, target in (("trading", cap.trading_capital_usd), ("investing", cap.investing_capital_usd)):
        state = store.ledger.state(sleeve, as_of, prices)
        out["sleeves"][sleeve] = {"target": target, "funded": round(funded(store, sleeve, as_of), 2),
                                  "equity": round(state.equity, 2) if prices is not None else None,
                                  "settled_cash": round(state.cash.available, 2),
                                  "unsettled": round(state.cash.unsettled, 2), "reserved": round(state.cash.reserved, 2)}
    gap = out["sleeves"]["trading"]["target"] - out["sleeves"]["trading"]["funded"]
    if abs(gap) >= 0.01:
        source, dest, amount = ("investing", "trading", gap) if gap > 0 else ("trading", "investing", -gap)
        movable = out["sleeves"][source]["settled_cash"]
        move = round(min(amount, movable), 2)
        if move >= 0.01:
            out["transfers"].append({"from": source, "to": dest, "cash": move})
        if move < round(amount, 2):
            out["notes"].append(f"{source} has ${movable:,.2f} of settled cash; ${amount - move:,.2f} more would need "
                                "sales there first (proposed separately), or a smaller target")
    total_target = cap.trading_capital_usd + cap.investing_capital_usd
    total_funded = out["sleeves"]["trading"]["funded"] + out["sleeves"]["investing"]["funded"]
    if total_target > total_funded + 0.01:
        out["notes"].append(f"targets add up to ${total_target:,.2f} but ${total_funded:,.2f} has been put in: the "
                            "difference needs a deposit, recorded when it arrives; no transfer can create it")
    out["notes"].append("Nothing moves until a transfer is recorded. Targets are net funding, not value: profits and "
                        "losses stay with the sleeve that earned them.")
    return out
```

```python
def daily_plan(store, as_of, reserved_today: float = 0.0) -> dict:
    """The planned Investing purchase for the next market session, bounded by settled cash."""
    from tradingagents.ledger.sessions import is_session, next_session, session_on_or_after

    cap = load(store)
    day = session_on_or_after(as_of)
    state = store.ledger.state("investing", as_of)
    available = max(0.0, state.cash.available - reserved_today)
    buy = round(min(cap.daily_investment_usd, available), 2)
    notes = []
    if not is_session(as_of):
        notes.append(f"{as_of} is not a market session; the plan is for {day}")
    if buy < cap.daily_investment_usd:
        notes.append(f"settled cash ${available:,.2f} covers ${buy:,.2f} of the ${cap.daily_investment_usd:,.2f} "
                     "planned; the rest is skipped this session (it does not carry over)")
    notes.append("A purchase plan, not a deposit: record deposits when they arrive.")
    return {"session": str(day), "next_session": str(next_session(day)), "planned": cap.daily_investment_usd,
            "settled_available": round(available, 2), "buy": buy, "shortfall": round(cap.daily_investment_usd - buy, 2),
            "notes": notes}
```

## The verdict (tradingagents/stocks/verdict.py)

```python
def verdict(held: bool, decision: dict | None, signal: dict | None, screen: dict | None, rating: dict | None,
            today: date | None = None, score: dict | None = None, horizon: int = 20) -> dict:
    """``{"verdict", "summary", "checks", "context", "held", "model"}``. ``checks`` decide the verdict;
    ``context`` is shown beside it and changes nothing. ``ok`` None means no data."""
    today = today or date.today()
    screen, rating = screen or {}, _fresh(rating, today)
    action = (decision or {}).get("action")
    s = score or {}
    # The position among today's stocks (ties broken), not the tied score, decides the fifths.
    pct = s.get("rank_pct", s.get("score_pct"))
    net, p_win, p_risk = s.get("expected_net"), s.get("p_win"), s.get("p_risk")

    checks = []
    if s:
        fund = s.get("kind") == "fund"
        peers = "funds" if fund else "S&P 500 stocks" + ("" if s.get("member", True) else ", where it would stand")
        checks = [
            _check("Model rank", pct >= TOP, f"#{s['rank']} of {s['of']} {peers} (ahead of {pct:.0%}; "
                                             f"BUY needs the top fifth, AVOID is the bottom fifth)"),
            _check("Expected return", net > 0,
                   f"{net:+.2%} above cash over {horizon} sessions (BUY needs above zero)" if fund else
                   f"{net:+.2%} over {horizon} sessions after costs of {s.get('cost_rt') or 0:.2%} (BUY needs above zero)"),
            _check("Chance of profit", p_win >= MIN_P_WIN, f"{p_win:.0%} (BUY needs {MIN_P_WIN:.0%}+)"),
        ]
        if p_risk is not None:
            checks.append(_check("Chance of a large loss", p_risk <= MAX_P_RISK,
                                 f"{p_risk:.0%} while held (BUY needs {MAX_P_RISK:.0%} or less)"))
    if action:
        checks.append(_check("Daily decision", action not in ("EXIT", "REDUCE"),
                             f"{action}{' (' + decision['paper_action'] + ')' if decision.get('paper_action') else ''}"))

    warnings = {w["kind"]: w["text"] for w in (signal or {}).get("warnings", [])} \
        if signal and signal.get("status") == "ok" else {}
    bad = [v for k, v in warnings.items() if k in BAD_SIGNALS]
    q, c, u = screen.get("quality_score"), screen.get("analyst_mean"), screen.get("target_upside")
    context = [
        _check("Breakdown or crash warning", None if not signal or signal.get("status") != "ok" else not bad,
               "no signals saved" if not signal or signal.get("status") != "ok" else ("; ".join(bad) or "none"), False),
        _check("Quality score", None if q is None else q >= MIN_QUALITY_BUY,
               "not screened" if q is None else f"{q:.0f} of 100", False),
        _check("Wall Street", None if c is None else c <= MAX_CONSENSUS_BUY,
               "no analysts" if c is None else f"consensus {c:.2f} (1 strong buy to 5 strong sell)"
               + (f", target {u:+.0%}" if u is not None else ""), False),
        _check("Agents' rating", None if not rating else None,
               "no recent agent analysis" if not rating else f"{rating['rating']} on {rating['date']} "
                                                             f"({rating['age_days']} days ago); not used", False),
    ]

    if held:
        if action in ("EXIT", "REDUCE"):
            v, why = "SELL", f"the daily decision says {action}"
        elif s and pct <= BOTTOM and net < 0:
            v, why = "SELL", f"the model ranks it #{s['rank']} of {s['of']} and expects {net:+.2%} after costs"
        else:
            v = "HOLD"
            if not s:
                why = "held; no model view (the model has not scored this ticker), so nothing says to sell"
            else:
                peers = "funds" if s.get("kind") == "fund" else "stocks"
                where = f"ranks #{s['rank']} of {s['of']} {peers}"
                why = (f"held; the model {where}, in the top fifth: a candidate to add to" if pct >= TOP else
                       f"held; the model {where}, in the bottom fifth, but still expects {net:+.2%}: "
                       f"weakest of what you hold, though no sell signal" if pct <= BOTTOM else
                       f"held; the model {where}, in the middle: no reason to sell or add")
    elif action == "ENTER":
        v, why = "BUY", "the daily decision says ENTER"
    elif not s:
        v, why = "WATCH", "no model view: the model has not scored this ticker"
    elif pct <= BOTTOM or net < 0:
        v = "AVOID"
        why = (f"the model ranks it in the bottom fifth (#{s['rank']} of {s['of']})" if pct <= BOTTOM
               else f"the model expects {net:+.2%} after costs over {horizon} sessions")
    elif all(x["ok"] for x in checks):
        v, why = "BUY", f"the model ranks it #{s['rank']} of {s['of']} and every test passes"
    else:
        missing = [x["name"].lower() for x in checks if x["ok"] is False]
        v, why = "WATCH", "the model's view is not strong enough: " + ", ".join(missing)
    return {"verdict": v, "summary": why, "checks": checks, "context": context, "held": held,
            "model": {"horizon": horizon, **s} if s else None,
            "rules": {"top": TOP, "bottom": BOTTOM, "min_p_win": MIN_P_WIN, "max_p_risk": MAX_P_RISK,
                      "rating_days": RATING_DAYS}}
```

```python
def ai_decision(rating: dict, held: bool) -> dict:
    """The agents' rating as an action: Buy / Overweight -> BUY, Hold -> HOLD, Underweight / Sell -> SELL
    when held (AVOID when not), with the rating's age. Shown as the agents' view; it decides nothing."""
    r = rating.get("rating")
    action = AI_DECISIONS.get(r, "REVIEW")
    if action == "SELL" and not held:
        action = "AVOID"
    if action == "HOLD" and not held:
        action = "WATCH"
    try:
        age = (date.today() - date.fromisoformat(str(rating.get("date"))[:10])).days
    except ValueError:
        age = None
    return {"action": action, "rating": r, "date": rating.get("date"), "age_days": age,
            "stale": age is None or age > RATING_DAYS}
```

## Signals and the entry plan (tradingagents/stocks/signals.py)

```python
def peers(forecasts: list[dict]) -> dict:
    """Today's spread of each probability across stocks: sorted values, and whether the model tells stocks apart."""
    out = {}
    for k, col in [*((k, f"p_{k}") for k in EVENTS), *((n, k) for k, n, _ in TRENDS)]:
        v = np.sort([f[col] for f in forecasts if f.get(col) is not None])
        enough = len(v) >= MIN_PEERS
        spread = float(np.percentile(v, 90) - np.percentile(v, 10)) if enough else None
        out[k] = {"values": v if enough else None, "spread": spread,
                  "flat": bool(enough and spread < FLAT_SPREAD), "n": int(len(v))}
    return out
```

```python
def describe(forecast: dict, base_rates: dict, peer: dict | None = None) -> dict:
    """Trend, lifts, peer ranks and warnings from one saved forecast's numbers (pure arithmetic)."""
    f, b, peer = forecast or {}, base_rates or {}, peer or {}
    mix = b.get("trend_mix") or []
    shares = [m / sum(mix) for m in mix] if mix and sum(mix) else []
    trend_lifts = {name: _lift(f.get(k), shares[i] if i < len(shares) else None) for k, name, i in TRENDS}
    # Only a class the model tells stocks apart on, high for this stock among today's, counts as its trend.
    usable = [n for _, n, _ in TRENDS if not (peer.get(n) or {}).get("flat")]
    best = max(((trend_lifts[n], n) for n in usable if trend_lifts[n] is not None
                and (_rank(f.get(f"trend_{n}"), peer.get(n)) or 1.0) >= TOP_SHARE), default=(None, None))
    if not usable:
        trend = "no stock-specific view"
    else:
        trend = best[1] if best[0] is not None and best[0] >= LIFT_TREND else "no clear trend"
    trend_p = f.get(f"trend_{best[1]}") if best[1] and trend == best[1] else None
    lifts = {k: _lift(f.get(f"p_{k}"), b.get(k)) for k in EVENTS}
    ranks = {k: _rank(f.get(f"p_{k}"), peer.get(k)) for k in EVENTS}
    flat = [k for k in EVENTS if (peer.get(k) or {}).get("flat")]

    def top(k):
        return ranks[k] is None or ranks[k] >= TOP_SHARE

    warnings = []
    if (lifts["up_break"] or 0) >= LIFT_BREAK and top("up_break") and "up_break" not in flat:
        warnings.append({"kind": "breakout", "level": "info",
                         "text": f"breakout likely: {f['p_up_break']:.0%} vs {b['up_break']:.0%} usual"})
    if (lifts["down_break"] or 0) >= LIFT_BREAK and top("down_break") and "down_break" not in flat:
        warnings.append({"kind": "breakdown", "level": "warn",
                         "text": f"breakdown risk: {f['p_down_break']:.0%} vs {b['down_break']:.0%} usual"})
    if (lifts["large_loss"] or 0) >= LIFT_CRASH and f.get("p_large_loss", 0) >= MIN_CRASH and top("large_loss") \
            and "large_loss" not in flat:
        warnings.append({"kind": "crash", "level": "high",
                         "text": f"crash warning: {f['p_large_loss']:.0%} chance of a large loss vs {b['large_loss']:.0%} usual"})
    if (lifts["deep_drawdown"] or 0) >= LIFT_CRASH and f.get("p_deep_drawdown", 0) >= MIN_DEEP and top("deep_drawdown") \
            and "deep_drawdown" not in flat:
        warnings.append({"kind": "drawdown", "level": "high",
                         "text": f"deep drawdown warning: {f['p_deep_drawdown']:.0%} vs {b['deep_drawdown']:.0%} usual"})
    return {"trend": trend, "trend_p": trend_p, "trend_lifts": trend_lifts, "current_trend": f.get("current_trend"),
            "lifts": lifts, "ranks": ranks, "flat": flat, "warnings": warnings}
```

```python
def entry_plan(forecast: dict, close: float, horizon: int = 20) -> dict | None:
    """Where to buy, and the exits that go with it, from the stock models' own numbers (pure arithmetic).

    Limit prices, each with the model's chance it is reached within ``horizon`` sessions:
    - the 20-day low (reached when the stock breaks down: P(breakdown)),
    - the median dip (the model's median drawdown from today's close: about an even chance),
    - a deep dip (its 5% drawdown quantile: about a 1-in-20 chance).
    Exits for an entry E follow the barrier policy the decision study uses: stop-loss at
    E * expm1(-sigma_h) below, take-profit at E * expm1(+sigma_h) above (sigma_h = the stock's
    volatility over the horizon), with the deadline of ``horizon`` sessions.
    """
    f = forecast or {}
    if not close or not (close > 0) or f.get("sigma_h") is None:
        return None
    import math

    sigma = float(f["sigma_h"])
    levels = []
    if f.get("lower") and f["lower"] < close:
        levels.append({"name": "At the 20-day low", "price": float(f["lower"]), "chance": f.get("p_down_break")})
    if f.get("dd_q50") is not None and f["dd_q50"] < 0:
        levels.append({"name": "Median dip", "price": close * (1 + float(f["dd_q50"])), "chance": 0.5})
    if f.get("dd_q05") is not None and f["dd_q05"] < 0:
        levels.append({"name": "Deep dip", "price": close * (1 + float(f["dd_q05"])), "chance": 0.05})
    levels.sort(key=lambda x: -x["price"])

    def exits(entry):
        return {"stop": entry * (1 + math.expm1(-sigma)), "profit": entry * (1 + math.expm1(sigma))}

    rows = [{"name": "Buy at the next close", "price": close, "chance": 1.0, **exits(close)},
            *({**lv, **exits(lv["price"])} for lv in levels)]
    for r in rows:
        r["vs_close"] = r["price"] / close - 1
    return {"close": close, "horizon": horizon, "sigma_h": sigma, "rows": rows}
```

## The evidence packet (tradingagents/agents/packet.py)

```python
def assemble(inputs: dict, limits: dict | None = None, include_model_forecasts: bool = False,
             forecasts: list[dict] | None = None, portfolio: dict | None = None) -> dict:
    """The packet, from already-loaded inputs. Deterministic: the same inputs give the same packet."""
    lim = {**LIMITS, **(limits or {})}
    ticker, cutoff = inputs["ticker"].upper(), inputs["cutoff"]
    cut = pd.Timestamp(cutoff)
    missing: list[dict] = []
    quality: list[dict] = []
    profile = inputs.get("profile") or {}

    kind_raw = profile.get("quoteType")
    kind = {"EQUITY": "stock", "ETF": "etf", "MUTUALFUND": "fund"}.get(kind_raw, "unknown")
    instrument = {"symbol": ticker, "type": kind,
                  "type_source": "Yahoo quoteType in the cached profile" if kind_raw else None,
                  "name": profile.get("longName") or profile.get("shortName")}
    if kind == "unknown":
        missing.append({"field": "instrument.type", "reason": "no cached profile with a quote type"})

    # Prices: cut at the cutoff before computing anything.
    bars = inputs.get("bars")
    bars = bars[bars.index <= cut].dropna(how="all") if bars is not None and len(bars) else pd.DataFrame()
    packet = {"version": PACKET_VERSION, "instrument": instrument,
              "cutoff": {"date": cutoff, "exchange_timezone": "America/New_York",
                         "latest_session": _d(bars.index[-1]) if len(bars) else None,
                         "rule": "only data observed and available on or before the cutoff's end of day"},
              "portfolio": portfolio or {"holding": "unknown", "position_size": "unknown", "cash": "unknown",
                                         "note": "portfolio context not supplied"}}
    if not len(bars) or "Close" not in bars:
        missing.append({"field": "prices", "reason": "no stored daily bars on or before the cutoff"})
        packet["snapshot"] = None
    else:
        packet.update(_prices(bars, inputs.get("ohlcv"), profile, cut, lim, missing, quality))
    if kind == "etf":
        packet["fund"] = _fund(profile, inputs.get("profile_fetched"), ticker, bars, missing)
    elif kind == "stock":
        packet["fundamentals"] = _fundamentals(inputs.get("facts"), cutoff, missing)
    packet["news"] = _news(inputs.get("news") or [], ticker, instrument.get("name"), cutoff, lim, missing)
    packet["macro"] = _macro(inputs.get("macro") or {}, cutoff, missing)
    packet["macro_availability"] = {"vintage": "the latest vintage published before the cutoff (ALFRED); "
                                               "revisions after it are not used",
                                    "lag_Nbd": "an unrevised daily series, usable N business days after its "
                                               "observation date"}
    if include_model_forecasts:
        packet["model_forecasts"] = [{"id": f"MF{i + 1}", **f} for i, f in enumerate(forecasts or [])] or None
        if not forecasts:
            missing.append({"field": "model_forecasts", "reason": "no saved model forecast for this instrument"})
    packet["checks"] = quality
    packet["missing"] = missing
    packet["trimmed"] = []
    return _fit(packet, lim)
```

```python
def _prices(bars, ohlcv, profile, cut, lim, missing, quality) -> dict:
    close, high, low = bars["Close"].astype(float), bars.get("High"), bars.get("Low")
    vol = bars.get("Volume")
    n = int(close.notna().sum())
    # Checks actually run on the cut series.
    order_ok = bool(bars.index.is_monotonic_increasing and not bars.index.duplicated().any())
    gaps = int(close.isna().sum())
    ohlc_bad = None
    if high is not None and low is not None:
        ohlc_bad = int(((close > high * 1.0001) | (close < low * 0.9999) | (low > high)).sum())
    quality.append({"id": "Q1", "check": "date order", "result": "ascending, no duplicates" if order_ok else "FAILED"})
    quality.append({"id": "Q2", "check": "missing closes", "result": f"{gaps} of {len(bars)} rows"})
    quality.append({"id": "Q3", "check": "OHLC consistency (low <= close <= high)",
                    "result": "not run: no highs/lows" if ohlc_bad is None else f"{ohlc_bad} inconsistent rows"})
    quality.append({"id": "Q4", "check": "price basis", "result": "closes, highs and lows split- and dividend-"
                    "adjusted (total-return basis) in the local store; volume as reported"})

    def at(series, back):
        s = series.dropna()
        return (None, None) if len(s) <= back else (_d(s.index[-1 - back]), _r(s.iloc[-1 - back]))

    ema10, sma50, sma200 = _ema(close, 10), close.rolling(50).mean(), close.rolling(200).mean()
    rsi = _rsi(close)
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    mid, sd = close.rolling(20).mean(), close.rolling(20).std()
    atr = None
    if high is not None and low is not None:
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    last = close.index[-1]
    rets = close.pct_change()
    hi250 = close.rolling(250, min_periods=1).max()
    indicators = {
        "close": _r(close.iloc[-1]), "ema_10": _r(ema10.iloc[-1]),
        "sma_50": _r(sma50.iloc[-1]) if n >= 50 else None, "sma_200": _r(sma200.iloc[-1]) if n >= 200 else None,
        "rsi_14": _r(rsi.iloc[-1], 1) if n >= 15 else None,
        "macd_12_26": _r(macd.iloc[-1], 3), "macd_signal_9": _r(signal.iloc[-1], 3),
        "macd_hist": _r((macd - signal).iloc[-1], 3),
        "boll_mid_20": _r(mid.iloc[-1]) if n >= 20 else None,
        "boll_upper_20_2sd": _r((mid + 2 * sd).iloc[-1]) if n >= 20 else None,
        "boll_lower_20_2sd": _r((mid - 2 * sd).iloc[-1]) if n >= 20 else None,
        "atr_14": _r(atr.iloc[-1]) if atr is not None and n >= 15 else None,
        "high_20d": _r(high.iloc[-20:].max()) if high is not None and n >= 20 else None,
        "low_20d": _r(low.iloc[-20:].min()) if low is not None and n >= 20 else None,
        "high_250d_close": _r(close.iloc[-250:].max()) if n >= 250 else None,
        "drawdown_from_250d_high_pct": _r(100 * (close.iloc[-1] / hi250.iloc[-1] - 1), 1) if n >= 250 else None,
        "realized_vol_20d_annualized_pct": _r(100 * rets.iloc[-20:].std() * math.sqrt(252), 1) if n >= 21 else None,
        "avg_volume_20d": _r(vol.iloc[-20:].mean(), 0) if vol is not None and n >= 20 else None,
    }
    for k, v in indicators.items():
        if v is None:
            missing.append({"field": f"snapshot.{k}", "reason": f"needs more history (have {n} sessions) or the "
                                                                f"input series"})
    for k, w in (("sma_200", 200), ("sma_50", 50)):
        quality.append({"id": f"Q{len(quality) + 1}", "check": f"{k} coverage",
                        "result": f"{min(n, w)} of {w} sessions" + ("" if n >= w else ": not computed")})
    opens = None
    if ohlcv is not None and len(ohlcv):
        o = ohlcv[ohlcv.index <= cut]
        if len(o) and o.index[-1] == last:
            opens = _r(o["open"].iloc[-1])
            other = float(o["close"].iloc[-1])
            gap = abs(other / float(close.iloc[-1]) - 1)
            quality.append({"id": f"Q{len(quality) + 1}", "check": "close cross-check (local store vs OHLCV cache)",
                            "result": f"differ by {100 * gap:.2f}%" + (" -> DISCREPANCY; the local store's value "
                                                                       "is used" if gap > lim["discrepancy_tol"] else "")})
    if opens is None:
        missing.append({"field": "snapshot.open", "reason": "no opens cached for this session"})
    # The cached profile's moving averages are today's, so they are compared only when the cutoff is today's data.
    if profile.get("fiftyDayAverage") and indicators["sma_50"] and cut == last:
        gap = abs(profile["fiftyDayAverage"] / indicators["sma_50"] - 1)
        if gap > lim["discrepancy_tol"]:
            quality.append({"id": f"Q{len(quality) + 1}", "check": "50-day average cross-check (computed vs profile)",
                            "result": f"differ by {100 * gap:.2f}% -> DISCREPANCY; the value computed from the "
                                      f"stored closes is used (precedence: local store)"})
    snapshot = {"id": "PX1", "session": _d(last), "source": "local price store (Yahoo daily bars)",
                "lookbacks": "EMA 10, SMA 50/200, RSI 14 (Wilder), MACD 12/26/9, Bollinger 20 / 2 sd, ATR 14, "
                             "20-day high/low, 250-session high, 20-day realized volatility",
                "history_sessions": n, "open": opens, **indicators}
    comps = []
    for back in lim["comparisons"]:
        d_c, v_c = at(close, back)
        if d_c is None:
            continue
        row = {"id": f"CMP{len(comps) + 1}", "sessions_back": back, "date": d_c, "close": v_c,
               "change_pct": _r(100 * (close.iloc[-1] / close.dropna().iloc[-1 - back] - 1), 2)}
        if n >= 50 + back:
            row["sma_50"] = _r(sma50.dropna().iloc[-1 - back]) if len(sma50.dropna()) > back else None
        if n >= 200 + back:
            row["sma_200"] = _r(sma200.dropna().iloc[-1 - back]) if len(sma200.dropna()) > back else None
        comps.append(row)
    recent = []
    tail = bars.iloc[-lim["recent_bars"]:] if lim["recent_bars"] else bars.iloc[:0]
    for i, (ts, r) in enumerate(tail.iterrows()):
        recent.append({"id": f"BAR{i + 1}", "date": _d(ts), "close": _r(r.get("Close")), "high": _r(r.get("High")),
                       "low": _r(r.get("Low")), "volume": _r(r.get("Volume"), 0)})
    return {"snapshot": snapshot, "comparisons": comps, "recent_bars": recent}
```

```python
def _news(rows, ticker, name, cutoff, lim, missing) -> list[dict]:
    from tradingagents.stocks.news import _jaccard, _norm, relevance

    end = _cutoff_end(cutoff)
    start = end - timedelta(days=lim["news_days"])
    unknown, future, kept = 0, 0, []
    for r in rows:
        if not r.get("pub_date"):
            unknown += 1
            continue
        pub = datetime.fromisoformat(r["pub_date"])
        pub = pub if pub.tzinfo else pub.replace(tzinfo=ZoneInfo("UTC"))
        if pub > end:
            future += 1
            continue
        if pub < start:
            continue
        kept.append((r, pub))
    # Duplicates (the same story from two feeds) collapse to the earliest copy.
    kept.sort(key=lambda x: (x[1], x[0].get("title") or ""))
    unique = []
    for r, pub in kept:
        if any(_jaccard(_norm(r["title"] or ""), _norm(u[0]["title"] or "")) >= 0.8 for u in unique):
            continue
        unique.append((r, pub))
    rank = {"about": 0, "mentions": 1, "other": 2}
    scored = sorted(unique, key=lambda x: (rank.get(relevance(x[0]["title"] or "", x[0].get("summary"), ticker, name), 3),
                                           -x[1].timestamp()))
    out = []
    for i, (r, pub) in enumerate(scored[:lim["news_items"]]):
        rel = relevance(r["title"] or "", r.get("summary"), ticker, name)
        summary = (r.get("summary") or "").strip()
        out.append({"id": f"N{i + 1}", "headline": r["title"], "publisher": r.get("publisher"),
                    "published_at": pub.astimezone(ZoneInfo("UTC")).isoformat(timespec="minutes"),
                    "first_seen_at": r.get("first_seen"), "url": r.get("link"), "relevance": rel,
                    "text_available": "feed summary only (no article body)" if summary else "headline only",
                    "summary": summary or None})
    if unknown:
        missing.append({"field": "news", "reason": f"{unknown} stored headline(s) with no publication time were "
                                                   f"left out (cannot pass a point-in-time check)"})
    if future:
        missing.append({"field": "news", "reason": f"{future} headline(s) published after the cutoff were left out"})
    if not out:
        missing.append({"field": "news", "reason": f"no stored headline for {ticker} in the {lim['news_days']} days "
                                                   f"to the cutoff"})
    return out
```

```python
def _known_at(series: str, rows: list[dict], cutoff: str) -> pd.Series | None:
    """The series as known on the cutoff: vintages published before it, or (unrevised series) observations
    whose availability lag has passed."""
    if not rows:
        return None
    if MACRO[series][4] and any(r.get("from") for r in rows):
        best = {}
        for r in rows:
            if r.get("from") and r["from"] < cutoff and r["date"] <= cutoff:
                if r["date"] not in best or r["from"] >= best[r["date"]][0]:
                    best[r["date"]] = (r["from"], r["value"])
        s = pd.Series({pd.Timestamp(d): v for d, (_f, v) in best.items()}).sort_index()
        return s if len(s) else None
    lag = LAG_BDAYS.get(series, 1)
    s = pd.Series({pd.Timestamp(r["date"]): r["value"] for r in rows}).sort_index()
    usable = [d for d in s.index if (d + pd.offsets.BDay(lag)) <= pd.Timestamp(cutoff)]
    return s.loc[usable] if usable else None
```

```python
def _macro(macro, cutoff, missing) -> list[dict]:
    out = []
    for series, (title, units, change, freq, revised) in MACRO.items():
        s = _known_at(series, macro.get(series), cutoff)
        if s is None or not len(s):
            missing.append({"field": f"macro.{series}", "reason": "not in the local FRED cache on or before the cutoff"})
            continue
        last_d, last_v = s.index[-1], float(s.iloc[-1])
        item = {"id": f"M{len(out) + 1}", "series": series, "title": title, "units": units, "frequency": freq,
                "observation_date": _d(last_d), "value": _r(last_v, 3),
                "availability": "vintage" if revised else f"lag_{LAG_BDAYS.get(series, 1)}bd",
                "comparisons": []}
        months = (1, 12) if freq != "daily" else None
        targets = [("1 month earlier", last_d - pd.DateOffset(months=1)), ("12 months earlier", last_d - pd.DateOffset(years=1))]
        for label, target in targets:
            prior = s[s.index <= target]
            if not len(prior):
                continue
            pd_, pv = prior.index[-1], float(prior.iloc[-1])
            if months and (target - pd_).days > 40:   # a monthly series must have that month
                continue
            comp = {"label": label, "observation_date": _d(pd_), "value": _r(pv, 3)}
            if change == "pp":
                comp["change"] = _r(last_v - pv, 3)
                comp["change_unit"] = "percentage points"
            elif change == "pct":
                comp["change"] = _r(100 * (last_v / pv - 1), 2)
                comp["change_unit"] = "percent change in the level"
            else:
                comp["change"] = _r(last_v - pv, 3)
                comp["change_unit"] = "index points"
            comp["period"] = f"{_d(pd_)} to {_d(last_d)}"
            item["comparisons"].append(comp)
        if series == "CPIAUCSL":
            yoy = next((c for c in item["comparisons"] if c["label"] == "12 months earlier"), None)
            item["inflation_12m_pct"] = yoy["change"] if yoy else None
            item["inflation_note"] = ("12-month CPI inflation is the change from the observation 12 months earlier; "
                                      "other spans are not year-over-year rates")
        out.append(item)
    return out
```

```python
def _fit(packet: dict, lim: dict) -> dict:
    """Drop whole low-priority items until the packet fits ``max_chars``; required parts stay intact."""
    def size():
        return len(render(packet))

    if size() <= lim["max_chars"]:
        return packet
    # Lowest value first: old daily bars (the comparisons carry the trend), the 250-session comparison,
    # quarterly statement items (annual figures and balances stay), then news beyond the five most relevant.
    bars = packet.get("recent_bars") or []
    while size() > lim["max_chars"] and len(bars) > 3:
        bars.pop(0)
        _note(packet, "recent_bars", f"kept the last {len(bars)} bars")
    comps = packet.get("comparisons") or []
    if size() > lim["max_chars"] and comps and comps[-1].get("sessions_back", 0) >= 250:
        comps.pop()
        _note(packet, "comparisons_250", "dropped the 250-session comparison")
    f = packet.get("fundamentals") or {}
    if size() > lim["max_chars"] and f.get("items"):
        f["items"] = [i for i in f["items"] if i["period"] != "quarter"]
        _note(packet, "quarterly", "dropped quarterly statement items (annual and balances kept)")
    news = packet.get("news") or []
    while size() > lim["max_chars"] and len(news) > 5:
        news.pop()
        _note(packet, "news", f"kept the {len(news)} most relevant headlines")
    if size() > lim["max_chars"]:
        _note(packet, "over_limit", f"required parts alone exceed {lim['max_chars']} characters; nothing cut mid-field")
    return packet
```

## Per-role agent models (tradingagents/llm_clients/roles.py)

```python
def resolve(config: dict) -> dict[str, tuple[str, str]]:
    """(provider, model) for every role."""
    default = config["llm_provider"].lower()
    overrides = validate(config.get("agent_models"))
    out = {}
    for role, (_, _, tier) in ROLES.items():
        spec = overrides.get(role)
        out[role] = (spec["provider"], spec["model"]) if spec else \
            (default, config["deep_think_llm" if tier == "deep" else "quick_think_llm"])
    return out
```

```python
def suggest(default_provider: str, providers: list[str]) -> dict[str, dict[str, str]]:
    """A split of the debates across the providers given (those with keys), the default first.

    Each debate puts its opposing sides on different providers (bull vs bear; aggressive vs
    conservative) so one model does not argue with itself; a third provider, when there is one,
    takes the neutral seat and the research manager, so the judge of the bull/bear debate is
    neither side. The analysts, trader and portfolio manager stay on the default. One provider:
    nothing to split.
    """
    order = [p for p in dict.fromkeys([default_provider.lower(), *providers])
             if p in providers and first_model(p, "quick")]
    if len(order) < 2:
        return {}
    a, b = order[0], order[1]
    c = order[2] if len(order) > 2 else None
    seats = {"bull": (a, "quick"), "bear": (b, "quick"), "aggressive": (a, "quick"), "conservative": (b, "quick")}
    if c:
        seats |= {"neutral": (c, "quick"), "research_manager": (c, "deep")}
    return {role: {"provider": p, "model": first_model(p, tier) or first_model(p, "quick")}
            for role, (p, tier) in seats.items()}
```

## AI gateway: budget, prices, cache (tradingagents/ledger/ai_gateway.py)

```python
    def run(self, *, task: str, provider: str, model: str, schema_version: str, info_cutoff: str, inputs: dict,
            work: Callable[[UsageCallback], Any], est_usd: float, est_requests: int,
            allow_paid: bool = False, also_priced: list[tuple[str, str]] = ()) -> tuple[Any, str]:
        """``(result, "cache" | "paid")``. ``work`` receives the callback to hand to the LLM client;
        ``also_priced`` lists any other models the work calls, which need a price too."""
        key = cache_key(task, provider, model, schema_version, info_cutoff, inputs)
        hit = self.cached(key)
        if hit is not None:
            return hit, "cache"
        if os.environ.get("TRADINGAGENTS_NO_PAID_AI") == "1":
            raise AIBlocked("paid AI is switched off (TRADINGAGENTS_NO_PAID_AI=1)")
        if not allow_paid:
            raise AIBlocked(f"{task} is not cached and paid AI was not requested")
        for p, m in [(provider, model), *also_priced]:
            self.price(p, m)
        self._claim(key, task, est_usd, est_requests)
        callback = UsageCallback(self, key, task, provider, model, models=[(provider, model), *also_priced])
        try:
            result = work(callback)
        except BaseException as exc:
            self._finish(key, "failed", str(exc)[:500])
            raise
        with transaction(self.con):
            self.con.execute("INSERT OR REPLACE INTO ai_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (key, task, provider, model, schema_version, info_cutoff, stable_hash(inputs),
                              json.dumps(result, default=str), now_utc()))
            self._finish(key, "done", None)
        return result, "paid"
```

```python
    def _which(self, *names) -> tuple[str, str]:
        """The (provider, model) a request belongs to, from the model names it reported; a versioned
        name ("deepseek-v4-flash-2026-07-01") matches its base ID. Unknown: the job's own model."""
        for name in filter(None, names):
            for p, m in self.models:
                if name == m:
                    return p, m
            for p, m in sorted(self.models, key=lambda pm: -len(pm[1])):
                if name.startswith(m) or name.endswith("/" + m):
                    return p, m
        return self.provider, self.model
```
