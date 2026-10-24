# Forecasting engine: what was built, what it found (2026-10-24)

Everything here runs locally on Yahoo Finance data (with the dated analyst and earnings
histories Yahoo provides). No paid AI is called in training, forecasting, backtesting or tests.

## What a forecast is

One record per asset, horizon and information cutoff (`research/forecast.py`, stored in
`forecast_runs` / `forecasts` in `portfolio.sqlite3`):

- **mean** and **median** log return **above cash** over the horizon (5, 20 or 60 trading
  sessions; never annualized), **5th and 95th percentiles**, and the **probability of a gain**
  (total return above zero);
- **sigma** (the predictive scale) and **estimation_sd** (how much the mean moves across
  bootstrap refits: model uncertainty, kept apart from the outcome's own spread);
- the model, its version, the feature schema, the training window, the walk-forward
  trial that backs it, data-quality flags, the CPU or GPU backend, time and peak memory.

Intervals come from the model's own errors on the newest fifth of its training data, which
it had not seen. Quantile models (XGBoost quantiles, Chronos-2) use their own quantiles.

## Data and leakage discipline

- Features at date *t* use closes up to *t* and analyst or earnings records stamped
  **before** *t*. The label is the next *h* sessions' log return above cash. Tests check that
  changing prices after a date leaves that date's features and forecasts unchanged.
- Folds are chronological and shared by all securities. A model trains only on rows whose
  labels **ended** before the test block (purge), with the label horizon as embargo. A test
  checks this for every fold.
- Uncertainty is computed over dates (a block bootstrap with 12-date blocks), not over the
  thousands of correlated rows.
- Funds enter once they have a year of prices; nothing before inception is invented.
- Every run is in the trial registry (`research_trials`); p-values are Holm-adjusted for
  the models tried.

## Return forecasts, ETF universe (39 ETFs plus holdings), tests 2012 to 2026

Trials 4 to 6 (rescores of 1 to 3) and the challenger trials. IC: rank correlation between
forecast and outcome across ETFs on each date (0 is chance), with 95% bootstrap interval.
"Within-date calibration": 1 means the *relative* sizes are right.

| Horizon | Model | IC (95% interval) | Within-date calibration | Market timing | 5-95% coverage |
|---|---|---|---|---|---|
| 20 | current rule (vol × Sharpe) | 0.067 (0.023, 0.110) | **−0.70** | −0.18 | 89% |
| 20 | momentum | 0.080 (0.021, 0.142) | 0.61 | −0.10 | 89% |
| 20 | Elastic Net | 0.111 (0.007, 0.219) | 1.01 | −0.08 | 89% |
| 20 | XGBoost | 0.113 (0.043, 0.192) | −0.28 | −0.07 | 88% |
| 20 | XGBoost quantiles | 0.107 (0.042, 0.177) | 0.90 | 0.02 | 88% |
| 20 | LSTM | 0.114 (0.042, 0.182) | 1.11 | 0.00 | 89% |
| 20 | Chronos-2 (zero-shot)\* | 0.101 (0.047, 0.158) | −0.06 | −0.25 | 87% |
| 60 | current rule | 0.118 (0.045, 0.193) | −0.89 | −0.26 | 90% |
| 60 | XGBoost | 0.212 (0.111, 0.326) | 0.92 | −0.08 | 88% |
| 60 | XGBoost quantiles | 0.178 (0.072, 0.290) | 1.01 | 0.06 | 88% |
| 60 | LSTM | 0.156 (0.060, 0.271) | 0.92 | 0.01 | 89% |
| 60 | Chronos-2 (zero-shot)\* | 0.207 (0.127, 0.306) | 0.33 | −0.06 | 90% |
| 5 | XGBoost quantiles | 0.092 (0.065, 0.124) | 0.75 | 0.04 | 87% |

\* Chronos-2's training data may include these prices (no cutoff is published). After its
October 2025 release its IC was 0.076 (20 sessions, 12 dates) and 0.186 (60 sessions, 10
dates): too few dates to tell skill from contamination.

**Readings.**
1. Every learned model ranks ETFs better than chance, and better than the current rule.
   This is the one reliable skill.
2. **No model can time the market.** The correlation between the average forecast and the
   average outcome is about zero or negative everywhere.
3. **Probabilities of a gain are worse than the base rate.** Every model's Brier score is
   above the null's.
4. The current rule's within-date calibration is negative: higher-volatility ETFs did
   relatively **worse**. That is the opposite of what the rule assumes.

## Do the forecasts make the portfolio better? (Mode A, after costs)

The optimizer was given each model's out-of-sample forecasts at each monthly rebalance,
2012 to 2026, ETF universe, worst-case stress limit, 10 bp per dollar traded
(`experiments/forecast_portfolio_study.py`). "rel:" uses forecasts only *between* assets
(a common premium plus each asset's difference from the day's average). Growth of $1:

| Model | 10% limit | 30% limit | No limit | Turnover (no limit) |
|---|---|---|---|---|
| **current rule** | **$1.92** (4.5%/yr, max drop 8.2%) | **$3.24** (8.3%) | **$4.32** (10.5%, 33.2%) | 2.2×/yr |
| momentum | $1.38 | $2.32 | $3.53 (8.9%, 22.3%) | 6.8× |
| XGBoost | $1.31 | $1.83 | $2.22 | 7.1× |
| XGBoost ranking | $1.57 | $2.68 | $4.03 | 12.6× |
| rel: XGBoost quantiles | $1.78 (4.0%, 7.8%) | $2.70 | $4.27 (10.4%, 28.5%) | 12.0× |
| rel: Elastic Net | $1.38 | $1.84 | $2.11 | 5.9× |
| *VTI held* | | | *$7.02 (14.2%/yr, max drop 35.0%)* | |
| *SPY held* | | | *$7.40 (14.6%/yr, 33.7%)* | |
| *SGOV (T-bills before 2020)* | | | *$1.26 (1.6%/yr)* | |

**No forecaster beat the current rule after costs.** The best (relative XGBoost quantiles)
roughly tied it, with a smaller worst drop at no limit, but five times the trading. Ranking
skill measured on single dates did not survive the optimizer, the stress limit and costs.
**Every optimizer setting, the current one included, trailed simply holding VTI or SPY over
this period.**

## Risk forecasts

Volatility over the horizon, scored by QLIKE (lower is better) on close-to-close realized
variance (there is no intraday data, so HAR-RV was not attempted). Coverage is the share
inside ±1.645 sigma (target 90%). See `experiments/output/risk_forecasts.json` for the final numbers.

| Horizon | 10-year sample (current) | trailing 3 months | EWMA | GARCH(1,1)-t |
|---|---|---|---|---|
| 5 | 0.685 (cov 92%) | 0.511 (87%) | 0.441 (87%) | **0.403** (88%) |
| 20 | 0.698 (93%) | 0.516 (88%) | 0.391 (88%) | **0.328** (92%) |
| 60 | 0.765 (94%) | 0.771 (88%) | 0.724 (88%) | **0.427** (93%) |

GARCH is best at every horizon, and EWMA beats the optimizer's 10-year figure. GARCH fits that
did not converge or forecast implausibly (about 2% of rows) count as no forecast and are left
out of its score. Median realized/forecast variance: 10-year 0.51-0.56 (too wide in a typical
month), EWMA 0.93-1.02, GARCH 0.71-0.84. That sits with coverage near 90% because
realized variance is skewed: most months are calm, and a few are not.

Covariance for the optimizer: the **10-year sample covariance did best**. Its long-only
minimum-variance portfolio had the lowest realized volatility, and its equal-weight
variance forecast was nearly unbiased (realized/forecast 1.04). Ledoit–Wolf, a 3-factor
model and EWMA, each on one year of data, underestimated next month's variance by 30-40%.
The optimizer keeps its covariance.

## Hardware (measured)

RTX 3050 Ti Laptop GPU (4 GB), i5-11400H, 16 GB RAM, Windows 11, driver 610.62, PyTorch
2.11 (CUDA 12.8), XGBoost 3.4.1 (CUDA build), scikit-learn 1.9.1, arch 8.0.

| Workload | CPU | GPU | Notes |
|---|---|---|---|
| XGBoost, 200,000 rows × 40, 300 rounds | 2.4 s | 1.6 s | GPU used automatically only at 200,000+ rows |
| Chronos-2, 64 series × 1,024 context, 20 steps | 4.8 s | 1.0 s | peak 631 MB; CPU and GPU medians differ by at most 6×10⁻⁸ |
| LSTM + Chronos-2 walk-forward, one horizon | n/a | 80-92 s | peak VRAM 1.4 GB |

One GPU job at a time (a lock file); a GPU failure reruns on the CPU, never on a paid service.

## What is switched on

Nothing changed in your plan: it still uses the current rule. Forecasts can be run and
inspected under **Forecasts** in the web app, and a sleeve can be set to use them
(`return_model`). That is not recommended on this evidence. A forecast run with no backing
trial moves nothing, because its weight is scaled to zero.

## Limits

- ETFs only for the portfolio test. The S&P 500 study (with analyst and earnings features)
  uses today's members and is survivorship-biased.
- Yahoo's analyst and earnings histories are its lists **as of today**. Corrections or
  dropped coverage cannot be detected.
- One path of history (2012-2026, mostly a US bull market). Results could differ in a
  different regime.
- Mid-period Chronos-2 contamination cannot be ruled out.
