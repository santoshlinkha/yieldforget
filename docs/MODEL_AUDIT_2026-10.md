# Model audit, 2026-10-27

I audited the daily stock decision (model A), the policy layer (C) and the agent routing, treating an outside
critique as a set of claims to check rather than as a spec. The audit used the code, the saved decision bundle
`ca2fb82cf224bc88c18b` (data to 2026-10-26, XGBoost 3.4.1), the `model_predictions` table and new experiments.

**Status labels** used below:
- **confirmed**: shown in code and data.
- **hypothesis**: plausible, not shown.
- **unavailable**: the evidence does not exist here.
- **validated**: measured on chronological data.

**Evidence status of every number:** every historical period, including 2025 onward, has been looked at in
earlier studies. Every measurement is therefore **development evidence**. The untouched test is the forward
record: `model_predictions`, scored as it matures, plus the new `agent_runs` table. Nothing here shows a useful
trading strategy.

## 1. The reported symptoms, reproduced

| Claim | Finding | Status | Evidence |
|---|---|---|---|
| GEN and DVA both score 0.0131 | Bitwise-equal float32 scores (0.01311726775…). The two share every leaf in all 10 trees, with different feature vectors. Of 499 members today only 151 scores are distinct, and 413 stocks are in a tie. **Cause:** early stopping found no gain on the purged inner validation, so the forced minimum of 10 trees was kept (depth 3, eta 0.02). The ties are legitimate given that model, not a join or rounding bug. | confirmed | `research/models.py:209`, `research/models.py:57`, `research/models.py:170` |
| Liabilities/assets contributes about +0.014 | That figure was mine, and it was GEN's contribution *minus the median stock's*. The raw TreeSHAP contribution is **+0.0080**. Baseline −0.00006 plus all contributions reproduces the score to 4e-9. GEN's positive contributions total +0.0154 and its negatives −0.0023. Across today's members, liabilities/assets carries 35% of all absolute attribution. | confirmed (my earlier number was misleading) | `pred_contribs` on the bundle |
| Many stocks share +0.54% expected net | The score-to-P&L map has 10 bins (`score_to_net`), so every stock in a bin shares one number. The top bin (0.537%) is below the 0.7–0.8 bin (0.654%): the map does not rise with the score. | confirmed | `stocks/adapters.py:151` |
| About −0.44% vs SPY for the same stocks | It is the same binned map of (trade net − SPY's total return over the identical entry/exit dates). Every bin is negative, from −0.32% to −0.77%. A positive absolute number and a negative active number coexist, with no arithmetic contradiction. | confirmed | `stocks/daily.py:250` |
| The 10th percentile is breached 12–15% of the time | This is the **policy layer's** 10th percentile of terminal excess over cash (a no-fill counts 0). It is not the ticker page's dip levels; my earlier doc linked it to the "patient dip" by mistake. Breach rates as fitted: 14.3% (next close) and 12.4% (limit dip), similar on 2025+. The existing gate accepts 5–15%. | confirmed | `stocks/policy_study.py:275`, `:378` |
| P(large loss) is a constant 7.1% | The risk model kept 10 trees with no useful split and returns **one value, 0.0713877, for all 499 members**: its training base rate. The verdict and the gate treated it as a stock-specific number, which let every stock pass a 15% cap. | confirmed | `stocks/models.py:32`, `research/models.py:209` |
| P(profit) is an unskilled tie-breaker | Out-of-sample Brier skill vs the training base rate: −0.051 (development) and −0.025 (2025+), AUC 0.53 / 0.47. Today's spread is 53.4–54.3%. It decided GEN over DVA. | confirmed | `docs/QUANT_UPGRADES.md`, `stocks/daily.py` (old sort) |
| Last-close buy limit vs next-close labels | The labels and the simulator buy at the next session's close, unconditionally (`entry = t_idx + 1`). The page told you to place a **limit at the last close**: a different, untested policy that skips rising stocks and fills intraday at other prices. Stops and targets were also quoted from the last close, not from the fill. | confirmed | `stocks/positions.py:137`, old `stocks/trade.py` |

**Found beyond the critique:** the live percentile was computed over **667 rows**, including about 168 former
members and extra companies. The score-to-P&L map was learned on member-only percentiles. Re-scoring
2026-10-26, **346 of 499 members** move to a different expected-net band once ranked among members only.
(confirmed, `stocks/daily.py:290`)

## 2. Information timing and leakage

| Item | Finding | Status |
|---|---|---|
| Folds | Yearly, split by date. Training rows need `label_end` before the fold starts, so purging uses the actual label interval, not a row count. Second-stage models also need the trade's exit before the fold. | confirmed correct (`stocks/study.py:64`, `:91`) |
| Inner early-stopping split | The last 20% of training dates, with earlier rows purged by label end. | confirmed correct (`research/models.py:57`) |
| Policy-layer calibration slice | The last N years before the fold; only labels that ended before the fold; fit rows end before the slice. | confirmed correct (`stocks/policy_study.py:112`) |
| Live fit | Ranker rows have labels ended by the cutoff; second-stage rows have trades exited by it. The bundle refuses to score its own training dates. | confirmed correct (`stocks/daily.py`, `training_store.guard`) |
| Feedback recalibration | An outcome is recorded only once `horizon` sessions have closed after the entry session. | confirmed correct (`stocks/feedback.py` `resolve`) |
| Index membership | Point in time, including removed stocks (fja05680). Delisting returns are unknown, so those labels are missing. **0.86%** of trades that could have finished by the cutoff have no outcome and are excluded. | limitation, disclosed (`survivorship: partial`) |
| Sector labels | `rel_sector_*` uses **today's** sector fund mapping for all history (e.g. the 2018 GICS change to Communication Services). | limitation (not point in time) (`stocks/panel.py:141`) |
| Fundamentals | Used from the day after filing, latest instant. Restatements in SEC company facts may replace original values. | hypothesis (not measured) |
| Removed stocks' fundamentals | Not attached (ticker reuse), so they are filled with 0 = the median rank in `Design`. | confirmed, limitation |
| Earnings dates (8-K Item 2.02) | Item 2.02 reports results after the fact; it is not a schedule. The quant branch's `days_to_earn_est` is an estimate from past filings and was not adopted. No schedule history is available. | unavailable |
| Holdout | 2025+ has been inspected repeatedly. It is development evidence; the forward record is the test. | confirmed |

## 3. Execution, objective and benchmark

**One execution specification**, `positions.execution_spec`, is now what the Trade page shows. It matches the
simulator and the labels:
- **Signal:** data through the 16:00 New York close.
- **Order:** buy the whole size at the next session's official close, with no limit price.
- **Fill:** always filled when a price exists. Partial fills, halts and queue position are not modelled; a missing price means no outcome.
- **Prices:** as-traded closes plus dividends.
- **Costs:** Abdi-Ranaldo spread plus 2 bp slippage per side.
- **Exit:** close-confirmed stop or target at ±1 horizon-volatility from the fill, sold at the next close; otherwise sold at the 20th close.
- **Account:** no day trades.

Daily bars cannot show the order in which intraday levels were touched. The policy avoids that by confirming on
closes, and a gap is sold at the next close, not at the level.

**Return definitions:**

| Measure | Definition | Where |
|---|---|---|
| Absolute net | (exit + dividends) ÷ entry − 1 − round-trip cost | `positions.trades` |
| Excess over cash | net − the cash fund's return over the same days | policy layer |
| Active vs SPY | net − SPY's total return over the same entry/exit dates | `map_excess` |

There is no beta-adjusted residual in production. It was tested as a ranker target (below).

**Thresholds:** `min_expected_net` = +0.5% and `min_p_win` = 50% were not chosen on development data in any
recorded study. They are listed as unselected settings. The validation gate (beat SPY after costs in the study
*and* in 20+ forward paper trades) already uses the benchmark objective.

## 4. Changes made (branch `portfolio-app`)

| Change | Where |
|---|---|
| Live percentiles among index members only; others are placed among members without moving them (`member_pct`) | `stocks/adapters.py:179`, `stocks/daily.py:290` |
| Tied scores share a rank (competition ranking) and a percentile, so the fifths never split identical scores. The order inside a tie is alphabetical (`TIE_ORDER`) and P(profit) no longer breaks ties. Each line carries `score_tied_with` as well as `tied_with` (same expected-net bin). | `stocks/daily.py:49`, `:377` |
| A risk model that gives every member the same probability (spread < 0.5 pt) is **uninformative**. The gate is not assessed and blocks with that reason. Setting `risk_gate_when_uninformative` = `skip` passes it, saying so on each line. The verdict's risk check fails with "not assessed". Scores carry `p_risk_informative`. | `stocks/daily.py:58`, `:391`, `stocks/verdict.py` |
| The Trade page shows the tested entry ("at the close of 2026-10-27, whatever it is"), not a limit. The last close is a dated reference; stop and target are percentages from the fill, with dollars "if filled at the last close". The patient dip is labelled an untested policy whose chance is of a touch, not a fill. When nothing passes, the top of the ranking is listed with its blocking reasons. | `stocks/trade.py`, `webapp/static/index.html`, `stocks/positions.py:61` |
| `agent_runs` (migration 8, append-only) records every routed agent run, failed and blocked ones included. Fields: requested and actual provider/model, each attempt with its reason, route version (hash of the chain), per-stage models, analysts, evidence hash, cache/paid, rating, final action, error, code version. | `ledger/schema.py:449`, `webapp/server.py:794` |

The agents' debate is unchanged and is not steered toward the model. They still receive model evidence only
when it is validated (none is).

**Effect today:** with the risk model constant, **no stock passes the gates**. The Trade page lists the top of
the ranking with "risk gate not assessed". This is the intended honest result, not a bug. The `quant-upgrades`
branch's fixed-8% fall label does tell stocks apart (development AUC 0.63), but it has not been validated and is
not merged.

## 5. Experiments (branch `quant-upgrades`, development evidence)

**Ranker targets** (`experiments/ranker_targets.py`, identical folds):
- **Residual targets:** both variants, (r − r_f) − β̂(r_SPY − r_f) and the same divided by ex-ante σ_h, raise IC against the residual (0.044 and 0.056 vs 0.030) and cut the top fifth's beta (0.82 and 0.62 vs 1.05). They *lower* IC against the barrier trade's net P&L (0.014 and 0.011 vs 0.022). Not promoted.
- **Simple baselines:** momentum and low volatility are no better.
- **Early stopping:** it often keeps the 10-tree minimum.

**One-sided split-conformal 10th percentile** (`experiments/q10_conformal.py`), on calibration slices that hold only matured outcomes:
- **Overall:** breaches fall from 14.3% to 10.5% (development) and from 13.7% to 10.1% (2025+).
- **By year:** coverage is not conditional, ranging from 1% to 22%.
- **Cost:** the bound widens by about 1.3 pt and pinball loss is mixed.
- **Status:** kept behind `CONFORMAL_Q10` (off).

## 6. Checks run

| Check | Result |
|---|---|
| `tests/test_model_audit.py` (member-only percentiles with ties, alphabetical tie order and `score_tied_with`, uninformative risk blocked or skipped by setting, verdict "not assessed", execution spec and Trade entry text, `agent_runs` recorded on failure and append-only) | 5 passed |
| `tests/test_quant_upgrades.py` (conformal finite-sample rank and simulated coverage, plus earlier tests) | passed |
| Full suite | 1,651 passed. 4 failed, the same 4 that fail on the untouched branch (ChatGPT sign-in, CLI update) |
| SHAP additivity on the live bundle | max error 3.7e-9 |
| Unit tests | They check mechanics only and do not validate investment performance. |

**Skipped or unavailable:**
- Intraday fill simulation: no intraday data.
- Earnings schedule history.
- Point-in-time sector history.
- A capital-constrained forward replay against static alternatives: the shadow account exists but has fewer than 20 trades.
- Fundamentals restatement audit.
- Leaf and imputation diagnostics beyond the two stocks examined.

## 7. Reproduce

```
# the bundle diagnostics (ties, leaves, SHAP, maps, risk)
.venv/Scripts/python.exe -I <scratch>/audit_bundle.py
# experiments (in the quant-upgrades worktree, on a cache copy)
PYTHONPATH=. TRADINGAGENTS_CACHE_DIR=<cache copy> python experiments/ranker_targets.py
PYTHONPATH=. TRADINGAGENTS_CACHE_DIR=<cache copy> python experiments/q10_conformal.py
.venv/Scripts/python.exe -m pytest tests/test_model_audit.py
```

## 8. What remains uncertain

- Whether any of the model's ranking survives costs going forward. Development IC is about 0.02 against net P&L; 2025+ is about 0.
- Whether the high-leverage tilt (strong in 2025+) is a priced factor or noise.
- Delisting outcomes (0.86% of trades missing).
- Restated fundamentals.
- Today's sector labels applied to the past.

**Next step:** let the forward record accumulate under the fixed code. That is the only untouched evidence left.
