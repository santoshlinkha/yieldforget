# Trade outcome models and the decision layer

What was audited, what was built, how it is validated, and what it still cannot claim.
Measured results are in `docs/POLICY_STUDY.md`, written by the study itself.

## 1. Audit (before the change)

How one stock (GEN, 2026-10-26) reaches a recommendation today:

1. `stocks/data.py:load` reads the local Yahoo store: adjusted Close, High, Low, Volume. **There are no
   opening prices**, so no fill can be priced at an open.
2. `stocks/panel.py:build` makes 54 features at the decision close (prices, market, fundamentals as
   filed, macro as published) on point-in-time S&P 500 members (survivorship **partial**).
3. `stocks/daily.py:fit_bundle` fits weekly: the XGBoost ranker (label: 20-session return over cash), a
   binned score-to-net map, a logistic P(profit) and an XGBoost P(large loss).
4. `stocks/daily.py:entries` applies fixed gates (expected net >= 0.5%, P(profit) >= 50%, P(large loss) <= 15%).
5. `stocks/positions.py:trades` simulates the barrier exit: entry at the t+1 close, close-confirmed
   target/stop at +-1 sigma_H, sale at the next close, 20-session deadline, dividends, costs.
6. `stocks/signals.py` turns separate event models into warnings; `stocks/verdict.py` labels BUY / WATCH /
   AVOID / HOLD / SELL; the pages show them.

Reusable and sound: execution after the cutoff (`positions.trades`: entry t+1), close-confirmed exits
that take gap losses, dividends, a spread-based cost (`panel.py` Abdi-Ranaldo + 2 bp), label-end
purging in cross-fitting (`study.crossfit_primary`), point-in-time fundamentals and macro vintages,
cash-flow-matched SPY benchmarks, the immutable decision records and the shadow account.

Problems found:

| # | Finding | Where | Change |
|---|---|---|---|
| A1 | "Breakdown" is an **intraday** event (the label uses daily lows: `targets.py` `below = fwd_l < lower`), but docs and the screener said "close below the range" | `signals.py` docstring, `index.html` screener notes and column titles | wording fixed everywhere to "trades below the 20-day low (intraday)"; it is a barrier touch, not a loss, crash or fill |
| A2 | The ticker page's entry plan called touch probabilities "fills" (20-day low: P(touch); median and deep dip: 50% / 5% by construction of a drawdown quantile) | `signals.entry_plan`, `ticker.html` | relabelled "trades there"; "a touch is not a fill" |
| A3 | "Crash warning: P(a large loss while held)" but the label is terminal (at the 20-session end) | `index.html` | corrected |
| A4 | P(profit) is displayed as a probability although no trade-quality model beat the base rate (log loss 0.73 vs 0.69, `docs/DECISIONS.md` section 8) | `verdict.py`, pages | the new layer reports calibrated outcome probabilities with their skill; the old P(profit) is left as the benchmark's input |
| A5 | No limit entry, no no-fill outcome, no cash while waiting | `positions.py` | `stocks/policies.py` |
| A6 | Delisted paths are dropped as "incomplete" (a survivorship bias in labels) | `positions.trades` | new labels keep them as `delisted`, sold at the last price |
| A7 | No untouched test period: July 2024 on was inspected in the earlier study | `study.py` | a holdout from 2025-01-01 scored once by this study; disclosed as not unseen by the author |
| A8 | Expected net map unreliable at the top (top tenth realized -0.5%) | `adapters.score_to_net` | the new models predict the excess directly and are judged on rank correlation and coverage |
| A9 | Fixed probability thresholds as buy rules | `daily.entries`, `verdict.py` | the new layer compares actions by expected excess with one margin chosen on validation |

Account constraints honoured (from `ledger`, `daily.rules`): cash account, T+1 settlement, no same-day
exit (no day trading), long only, at most 5 names, 25% per name, 10% of the IRA per name, the trading
budget (SGOV interest + match) as the only money for stocks.

## 2. What was built

- `stocks/policies.py`: the fixed policy set (next close; a limit 0.5 x 5-session volatility below the
  close for 5 sessions, filled only when the low goes 0.1% through it, priced at the limit; hold from now)
  and triple-barrier labels: target / stop / expiry / nofill / delisted / incomplete, net and excess over
  cash, holding sessions, adverse excursion, the label end for purging. Exits from the actual fill.
  Close-confirmed barriers mean one close per bar, so there is no intrabar ordering to assume.
- `stocks/outcome_models.py`: LightGBM and a linear baseline behind one interface: outcome probabilities,
  expected excess and its 10th percentile; per-class isotonic (sigmoid below 5,000 rows) calibration on a
  separate later slice.
- `stocks/decision_layer.py`: buy now / limit / cash and hold / sell by expected excess over cash after
  costs, a no-trade margin, a risk budget on the 10th percentile, and abstention with reasons. Holding is
  judged from now on, never from the purchase price.
- `stocks/policy_study.py` (+ `experiments/policy_study.py`, `stocks/report_policy.py`): the purged
  walk-forward, the holdout, benchmarks, block-bootstrap intervals, the promotion rule, the live bundle.
- `stocks/daily.py`: the saved daily decision carries the policy view (options, action, model version,
  status), next to and separate from the rules' actions.
- Pages: a "Trade outcome model" section on the ticker page and a card on the Trade view, showing only
  saved values, "–" when unsupported, and "insufficient evidence" with its reason.
- Agents: `decision_layer.agent_evidence` gives the trader, risk debaters and portfolio manager the
  model's estimates as labelled evidence **only when validated** (empty otherwise); no instruction is
  added and the debate is unchanged. Agent ratings stay context, never probabilities.

## 3. Reproduce

    .venv/Scripts/python.exe -m experiments.policy_study --device cuda
    .venv/Scripts/python.exe -m pytest tests/test_policies.py tests/test_policy_models.py

## 4. Not done, and why

- Meta-labeling: a take/skip model on the ranker's out-of-fold candidates was studied before
  (`TradeQuality`): no version beat the base rate. Agent decisions have no history; they are recorded
  prospectively (memory log, saved runs), and a meta-model can be studied once enough have matured.
- Intraday stop orders, fills at the open, queue position: no intraday or opening data.
- Conformal intervals: the 10th percentile is checked by its empirical coverage instead.
- Foundation-model challengers (TabPFN, Chronos, Kronos): not run; Chronos-2 was benchmarked earlier
  for ETFs with no edge after costs (`docs/FORECASTING.md`), and the baseline must first pass.

## 5. Measured result (trial 33, 2026-10-27; full table in `docs/POLICY_STUDY.md`)

**Not validated.** On the 2025-2026 holdout the LightGBM outcome model's P(target first) and P(stop first)
were *worse* than the training base rates (Brier skill -0.015 and -0.047; stops were more frequent than
forecast, ~30% observed against ~22% forecast), and the decision layer averaged -0.94% over cash per
decision (95% interval -2.58% to +0.64%), behind the existing rule (+1.15%), SPY (+1.01%) and 12-1
momentum (+2.81%), all of whose intervals also include zero. On development (2014-2024) it had a small
positive edge (+0.87% per decision, interval above zero) that did not exceed SPY (+0.99%) and did not
carry into the holdout. The linear baseline chose cash on every holdout date.

Consequence: the live layer abstains ("insufficient evidence") on every stock, shows its estimates for
reference only, and gives the agents nothing. The existing model stays the benchmark; nothing here is a
recommendation.

Evidence needed before trusting any of it: a policy whose holdout probabilities beat the base rates
and whose decisions beat cash *and* SPY with an interval above zero, then a prospective record (the
saved daily decisions, scored as they mature) of at least a year that agrees.
