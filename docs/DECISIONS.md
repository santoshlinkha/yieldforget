# Daily stock decisions: settings, execution contract, models, evidence and records

Status as of 2026-10-24, branch `portfolio-app`. Recommends and records; **never submits orders**.

## 1. The three capital amounts (Settings > Capital, `book capital`, `/api/capital`)

| Setting | Default | Meaning |
|---|---|---|
| `trading_capital_usd` | 1,000.00 | Capital assigned to the Trading sleeve: a target for its **net funding**, not a daily budget |
| `investing_capital_usd` | 4,800.00 | The same target for the Investing sleeve |
| `daily_investment_usd` | 27.61 | The planned Investing purchase per market session, bounded by settled cash |

- One canonical interface: `tradingagents/ledger/capital.py` (`load`, `update`, `snapshot`, `history`).
  Every write is validated on the backend (finite, 0 to $10M; daily up to $100k; rounded to cents) and
  kept in `settings_history`. The old keys `trading.initial_allocation` and
  `investing.daily_contribution` are aliases; values saved under them seeded the new settings.
- The ledger stays the truth. Changing a target never moves money: `reallocation` compares each
  target with what the sleeve was **funded** with (contributions and transfers in, net of
  withdrawals and transfers out), not its equity, so profits and losses stay with the sleeve that
  made them and nothing refills a losing Trading sleeve. Only settled, unreserved cash can be
  proposed for transfer; a shortfall needs a deposit (recorded when it arrives).
- The daily plan (`daily_plan`): the next session (holidays skipped), the planned amount, the
  settled cash after reservations, the buy, the shortfall (skipped that session, not carried
  over), and a reminder that a plan is not a deposit.
- `snapshot()` (values, settings version, hash) is stored in every official recommendation, every
  stock decision, the shadow account's first state and every decision study, so a later change never
  alters a past record. Dollar changes resize plans without retraining any model (model caches are
  keyed by data, features, thresholds, policy and code, never by dollars).

## 2. Execution and label contract

- Decisions after the close of session t, from data known by then.
- Earliest entry: the close of **t+1**. Fixed exit after h sessions: the close of **t+1+h**.
- Position adverse excursion: lows of t+2 .. t+1+h relative to the entry close, including zero.
- Descriptive events measured from the close of t (breakouts, trend states) stay separate from
  position risks (`pos_large_loss`, `pos_deep_drawdown`), which risk gates use.
- Arithmetic barriers from log thresholds: `expm1(-k * sigma)`.
- **Close-confirmed levels**: a close through a profit or risk level schedules the exit at the next
  close; no intraday fill is invented; gaps can lose more than the stated level.
- Holding deadlines are exit dates; the decision is taken the session before. Never extended.
- No lot is sold on its entry session. Sale proceeds settle T+1. Buys use settled cash only.
- Accounting uses executable prices (closes as traded, split-adjusted) and dividends as cash on the
  ex-date; research labels use the total-return series; tests reconcile the two.
- One engine (`tradingagents/stocks/positions.py`) produces trade labels, the historical simulation
  and the prospective shadow account.

## 3. Point-in-time foundations (verified and fixed)

- Historical S&P 500 membership (fja05680/sp500, 1996 on, including 479 members removed since 2005,
  of which 168 still have Yahoo prices). Survivorship is **partial**: removed stocks without prices
  are missing and their delisting outcomes unknown. Removed tickers get no SEC facts (ticker reuse).
- Cross-sectional ranks use only that date's members; requested non-members are placed against the
  member distribution and never trained on, so asking about a stock changes no one else's features
  or models. Tickers are sorted, so input order cannot matter.
- SEC facts by filing date (from the next day), restatements at their own filing date; missing capex
  is unknown (no FCF); share counts carried to the valuation price's split basis.
- FRED: revised series through ALFRED vintages (next day); daily series lagged by business days per
  series (DFF 1, T10Y2Y 2, BAA10Y 2) to their publication.
- Cash-fund returns missing before the fund existed leave no label (never zero).
- Recommendation inputs priced as of the decision date; a price over a week old blocks approval;
  a missing or invalid risk probability blocks approval; name and concentration limits count holdings.
- Evidence is bound to the exact thresholds version and feature list: a changed design starts untested.
- The price cache was checked: an incremental update that finds Yahoo re-adjusted the past (a new
  dividend or split) re-downloads that ticker's whole history, so adjustment bases are never mixed.

## 4. Decisions

**ENTER / PAPER ENTER / WAIT** for candidates; **HOLD / REDUCE / EXIT / REVIEW** for lots.

- Inputs per candidate (models and arithmetic only): within-date score percentile from XGBoost
  learning-to-rank; expected net arithmetic P&L under the complete exit policy and its spread and
  10th percentile (a shrunken map learned on cross-fitted predictions only); expected excess over SPY;
  P(profitable after costs) (trade-quality logistic, sigmoid-recalibrated on purged chronological
  holdouts); P(large loss during the position); estimated round-trip cost.
- Gates: expected net >= 0.5%, P(profit) >= 50%, P(large loss) <= 15% (missing blocks), current data,
  settled cash after reservations, at most 5 names including holdings, at most 25% of the sleeve per
  name including holdings, at most 10% of the whole IRA in one stock (Investing and Trading together).
- Modes: **validated** requires the decision study to pass on development (beats cash-flow-matched SPY
  after costs with a positive interval, beats cash, drawdown >= -35%, calibration no worse than the base
  rate) **and** the prospective shadow record to confirm (>= 20 round trips, ahead of SPY). Otherwise
  **experimental**: the validated action is WAIT and the buy shows as PAPER ENTER. WAIT is acceptable.
- Lots: deterministic, close-confirmed: deadline, risk level or profit level -> EXIT at the next close;
  above the per-name cap -> REDUCE to it; no price -> REVIEW; bought this session -> HOLD. Earnings
  within about a week are flagged. The ML continuation exit is information only until validated.
- Sizing: equal dollars across open name slots within the caps (volatility-aware sizing is studied).

## 5. Models and their status

| Model | Kind | Where | Status |
|---|---|---|---|
| Cash, momentum (12-1), within-date Elastic Net | baselines | `adapters.py` | in studies |
| XGBoost learning-to-rank (groups = dates) | primary ranker | `adapters.Ranker` | deployed (experimental) |
| Trade quality: logistic / XGBoost, none / sigmoid | meta-label | `adapters.TradeQuality` | deployed: logistic + sigmoid |
| Position risk (XGBoost) | probability | `models.XGBEvent` | deployed |
| ML continuation exit (XGBoost regression on position states) | exit | `study.continuation_states` | study only |
| Compact PatchTST (~30k weights) | challenger | `challengers.PatchTST` | study |
| Chronos-2 (amazon/chronos-2 @ 29ec376, Apache-2.0) | pretrained | `challengers.Chronos2Score` | study, contamination unknown |
| TimesFM 2.5 200M (google/timesfm-2.5-200m-pytorch @ 1d95242, Apache-2.0) | pretrained | `challengers.TimesFM25Score` | study, contamination unknown |
| TimesFM 3.0 (google/timesfm-3.0-pytorch @ 43046b8) | pretrained | not wired by default | **non-commercial licence**: owner decision |
| Kronos-mini (NeoQuasar/Kronos-mini @ f4e6869, tokenizer -2k @ 26966d0, code commit 67b630e, MIT) | pretrained OHLCV | `challengers.KronosScore` | study, contamination high |

No reinforcement learning, and no ensemble required by default. Pretrained scores are retrospective
exploration; only prospectively recorded forecasts can confirm one.

## 6. Three performance records

1. **Historical simulation** (`experiments/decision_study.py`): capital from the Trading setting,
   shared yearly folds, cross-fitted second stage, rules chosen on development (to June 2024) only,
   July 2024 on reported as exploratory (it was inspected before), capital reused only after T+1
   settlement, dividends, costs, daily marks.
2. **Prospective Day-1 shadow account** (`shadow.py`, table `shadow_snapshots`): the frozen PAPER
   ENTER decisions followed with the same engine, capital frozen at inception, against SPY bought with
   the same capital on the same day.
3. **Actual performance**: the ledger's fills.

## 7. How to run

```
.venv\Scripts\python.exe -m cli.main book capital --trading 1000 --investing 4800 --daily 27.61
.venv\Scripts\python.exe -m cli.main stocks today --train        # first run of the week fits the models
.venv\Scripts\python.exe -m cli.main stocks today                # later runs reuse them
.venv\Scripts\python.exe -m cli.main stocks decisions [--decision ID]
.venv\Scripts\python.exe -m cli.main stocks shadow
PYTHONPATH=. .venv\Scripts\python.exe experiments\decision_study.py
PYTHONPATH=. .venv\Scripts\python.exe experiments\challenger_stock_study.py
```
API: `GET /api/capital`, `POST /api/capital {"values": {...}}`, `POST /api/capital/transfer
{"from","to","cash"}`, `GET /api/decisions`, `POST /api/decisions/run {"train": false}`.
Web: Settings > Capital; the Decisions tab (opening it only reads saved records).

## 8. Results of the decision study (2026-10-24, trial `decision_study:h20`)

Universe: point-in-time S&P 500 since 2005 (982 tickers, survivorship **partial**), decisions every 5
sessions, horizon/deadline 20 sessions, Trading capital $1,000 (settings snapshot recorded). Development
2013 (simulation) / 2012 (models) to June 2024; **exploratory** July 2024 to October 2026 (inspected in an
earlier study, so not an untouched test). Runtime 23 minutes on the laptop (GPU for boosted trees).

**Ranking** (IC against the tradable label; 95% block-bootstrap interval, blocks of 4 decision dates =
the 20-session overlap at a 5-session cadence):

| Model | Development | Exploratory | Ranked dates (dev) |
|---|---|---|---|
| XGBoost learning-to-rank | **0.020 (0.005, 0.036)** | 0.037 (0.007, 0.076) | 628 of 628 |
| Within-date Elastic Net | -0.017 (-0.056, 0.025) | none (constant) | 357 of 628 |
| Momentum 12-1 | 0.005 (-0.021, 0.033) | 0.032 (-0.011, 0.081) | 628 |

The ranker has small but consistent ranking skill (IC with the policy's net P&L 0.016 / 0.012).

**Trade quality** (log loss against the base rate, lower is better; base rate of a profitable trade
about 53-54%): logistic 0.89 (badly overconfident), logistic + sigmoid 0.73, boosted 0.70, boosted +
sigmoid 0.72, base rate **0.69**. **No trade-quality model beats the base rate** in development; in the
exploratory period boosted trees roughly tie it (0.691 vs 0.692). Sigmoid recalibration helps logistic a
lot and boosted trees not at all.

**Expected net P&L map**: the top tenth by mapped expected net (+1.1% expected) realized **-0.5%** per
trade in development against +0.55% for all candidates: the rank-to-return map is not reliable at the
top, despite the positive IC.

**Capital simulation** ($1,000, at most 5 names, 25% each including holdings, settled cash, T+1,
dividends, costs; vs SPY with the same capital, daily excess interval by 20-session blocks):

| Policy (rule chosen on development) | Development total / yearly / max drawdown | Exploratory total | SPY dev / exploratory | Cash dev / exploratory |
|---|---|---|---|---|
| Fixed 20 sessions (net >= 0, P >= 0.55) | +53% / 3.8% / -33% | +8.7% | +358% / +43% | +14% / +9.9% |
| Barrier 1 sigma / 1 sigma + 20-session deadline (net >= 0.5%, P >= 0.5) | +210% / 10.4% / -22% | **+5.0%** | +358% / +43% | +14% / +9.9% |
| Barrier, volatility-aware sizing | +155% / 8.5% / -16% | -1.3% | | |
| Momentum top-5, fixed 20 | +51% / 3.6% / -43% | +79% | | |

- Daily excess over SPY: development interval (-0.04%, +0.01%) for the barrier policy, below zero for the
  fixed policy; exploratory intervals include zero but centre below it.
- The barrier policy's development figure comes from the best of eight rule settings (the next best was
  +64%): selection on development inflates it. Its exploratory result (+5.0%) is below cash (+9.9%).
- Exposure was about 42-62%: candidates often failed the gates, so the sleeve sat partly in cash.
- Exits: barrier policy 243 deadline / 123 target / 81 stop exits in development.

**ML continuation exit vs fixed exit on identical entries** (61,782 entries, top fifth by score; XGBoost on
position states, chronologically cross-fitted): the model almost never predicted a negative continuation
benefit, so it almost never exited early (0.03% of trades at tau = 0) and changed P&L by about zero. No
evidence it helps.

**Verdict.** The complete policy does **not** pass: it does not beat cash-flow-matched SPY after costs,
its trade-quality probabilities are worse than the base rate, and its exploratory result trails cash.
The daily decision therefore runs in **experimental** mode: validated actions are WAIT; buys appear only
as PAPER ENTER and feed the prospective shadow account, which is the record that could change this.

## 9. Challenger study (trial `challenger_stock_study:h20`)

150 of today's S&P 500 members with 10+ years of prices (seed 11; survivorship-**biased**), decision
dates every 20 sessions from 2014, two-year folds from 2016, horizon 20, identical rows for every model.
Pretrained models are zero-shot and retrospective (exploration only). Laptop RTX 3050 Ti (4 GB).

| Model | Development IC (95%) | Exploratory IC (95%) | Ranked dates (dev) | Total runtime | Peak GPU memory |
|---|---|---|---|---|---|
| XGBoost learning-to-rank | 0.012 (-0.027, 0.047) | 0.014 (-0.045, 0.070) | 81 / 106 | 4 s | small |
| Within-date Elastic Net | 0.000 (-0.048, 0.051) | 0.007 | 50 / 106 | 51 s | CPU |
| Momentum 12-1 | -0.014 (-0.050, 0.023) | 0.025 (-0.037, 0.099) | 106 | 0 s | none |
| PatchTST (compact) | -0.002 (-0.030, 0.025) | -0.016 | 106 | 11 s | 0.5 GB |
| Chronos-2 | 0.009 (-0.015, 0.033) | -0.018 | 106 | 136 s | 1.4 GB |
| TimesFM 2.5 | 0.012 (-0.013, 0.036) | -0.018 | 106 | 170 s | 2.1 GB |
| Kronos-mini | **0.041 (0.016, 0.067)** | 0.014 (-0.038, 0.056) | 106 | 32 min | 1.8 GB |

Only Kronos-mini's development interval is above zero, and it is the model most likely to have seen
these prices in pretraining (financial K-lines); it did not hold up in the exploratory period. Treat it as
an exploration lead: the only test that can confirm it is a prospective one (forecasts recorded before
the outcomes). Chronos-2, TimesFM 2.5 and the compact PatchTST show no ranking skill here. TimesFM 3.0 was
not run (non-commercial licence); its adapter path is the same as 2.5's once the owner decides.

## 10. Audit fixes (2026-10-25)

- **Validated means validated for this exact model.** The evidence check binds a study to its model code,
  policy version, features and thresholds, and the prospective record to the shadow version and policy.
  Studies recorded before the binding count as untested, so the model is not validated until a new full
  study runs. Until then BUY shows as WAIT and SELL as HOLD, with the model's view labelled "not validated".
- **No look-ahead in a past decision.** A decision for an earlier day sees only prices, trades and outcomes
  known at its close, and never reuses a model fitted later.
- **Nothing is overwritten.** A new agent run or pasted answer archives the earlier run for that day;
  each pasted answer keeps the exact evidence packet it answered.
- **Honest backtests.** The ETF backtest trades the session after it plans; the legacy stock scorer
  compounds correctly and annualizes on real spacing. Old figures stay as recorded.

## 11. Entry plans (2026-10-25)

- **The AI writes the plan, the app checks it, you execute it.** An entry plan is only actionable after
  the local check passes: traded-price basis verified, settled cash, caps, timing, no day trading. A blocked
  plan is kept with its reasons.
- **The model's verdict still decides BUY or SELL**; the entry plan says how an entry could be made.
- **No extra AI calls.** The plan rides in the trader's and manager's existing answers, for about 1.5k more
  input tokens per run.
