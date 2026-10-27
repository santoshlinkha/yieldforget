# Quant upgrades: what was tried, what it measured, what was adopted

Every change was compared with the current model on identical rows, using the purged yearly
walk-forward. The development folds run from 2012 to 2024. The holdout runs from 2025-01-01 on and is
scored with the same frozen recipe. Ranges are 95% block-bootstrap intervals over decision dates.
Nothing was adopted on holdout numbers alone: a change had to help in development and not hurt on
the holdout.

Scripts (both read a copy of the price cache through `TRADINGAGENTS_CACHE_DIR`):

- `experiments/quant_upgrades.py` → `experiments/output/quant_upgrades.json`: models A and B, the features, and the risk and quality models.
- `experiments/policy_variants.py` → `experiments/output/policy_variants.json`: model C.

Panel: 404,839 stock-dates (every 5 sessions, index members only, partial survivorship).

## Adopted

| Change | Evidence | Where |
|---|---|---|
| **Risk label: a fixed 8% fall** (any close in the 20 sessions after the entry ≥ 8% below it) instead of a loss of 1.5× the stock's own volatility | The old label gives every stock about the same chance (AUC 0.49 in development and on the holdout; predictions 6.9–8.1% on the holdout). The new one ranks stocks: AUC 0.63 in development and 0.59 on the holdout. In 2025 the riskiest tenth fell 8% at a 26% rate vs 8% for the safest tenth; in development, 33% vs 9%. | `targets.pos_dd8`, `daily.RISK_LABEL` (bundle `decision-v2`) |
| Its gate at **30%** (new rule key `max_fall8_prob`; the old 15% key is not reused) | The base rate is 18% (development) to 24% (2025), so 15% would block about half the stocks. The riskiest fifth did not earn less after costs (+0.15% vs the rest), so the gate only removes extremes; it is a drawdown limit, not a return filter. | `daily.DEFAULT_RULES`, `verdict.MAX_P_RISK` |
| **Live recalibration waits for about two years** (20,000 matured predictions over 500 decision days, was 300 over 10) | A calibration fitted on a few weeks follows one market spell. | `feedback.MIN_ROWS, MIN_DATES` |
| **Sector causes on 8 drivers**: geopolitical threats (Caldara-Iacoviello GPR daily threats index, lagged one day) and a gold/defense/oil basket vs SPY, added to the six | Average R² over yearly windows: 0.266 with 8 vs 0.217 with 6; two noise drivers would add about 0.01. This is explanation, not forecasting. | `rotation.DRIVERS`, `rotation.driver_moves` |

Caveat on the new risk number: its level runs low in a volatile year. In 2025 the Brier skill against
the training base rate is −0.031, because the base rate jumped from 18% to 24%. Read it as an
ordering ("riskier than most today") more than as an exact chance.

## Tried, not adopted

### Model A: the stock ranker (development IC on net trade P&L; holdout IC; holdout top fifth minus all)

| Variant | Dev IC net | Holdout IC net | Holdout top fifth − all |
|---|---|---|---|
| **Current: 54 features, return-over-cash label** | **0.022** [0.008, 0.036] | 0.001 [−0.032, 0.033] | +0.05% |
| + residual momentum (126-session regression on SPY and the sector fund) | 0.018 | 0.001 | +0.05% |
| + earnings timing (SEC 8-K Item 2.02) | 0.018 | 0.002 | +0.07% |
| + geopolitical/policy regime and sensitivities | 0.015 | −0.005 | +0.05% |
| + all of the above | 0.015 | 0.011 | −0.31% |
| Label: barrier net ÷ volatility, graded into fifths | 0.025 [0.004, 0.047] | **−0.036** [−0.071, 0.004] | **−0.74%** [−1.19%, −0.37%] |

None of the feature groups beats the current set in development. The barrier-aligned label is
marginally better in development but its 2025 top fifth did significantly *worse* than the average
stock, so it is rejected. The earnings dates cover only 51% of rows, because the SEC "recent filings"
list stops after about 1,000 filings per company. Going back further needs the paged archive.

### Model A: trade quality, P(profit after costs)

| Variant | Dev Brier skill | Holdout Brier skill | Holdout AUC |
|---|---|---|---|
| Current: logistic + sigmoid | −0.051 | −0.025 | 0.47 |
| LightGBM, monotone in the ranker percentile, + volatility, VIX term, sector, earnings, residual momentum, GPR; isotonic on the last 2 training years | −0.029 | −0.034 | 0.45 |

Both are worse than always saying the base rate. Nothing in the features predicts *whether* a single
trade makes money better than the average. P(profit) stays as it is and stays a weak number.

### Model C: the trade policy (target / stop barriers, next-close vs limit-dip entry)

Strategy = per-date mean excess return over SPY of the trades chosen. The margin is picked on
development only.

| Variant | Development | Holdout (2025–26, 45 dates) |
|---|---|---|
| **Current: symmetric barriers 1.0 / 1.0** | +0.87% [+0.13%, +1.67%] | −0.94% [−2.58%, +0.64%] |
| Asymmetric: target 1.75 σ, stop 1.25 σ | +1.01% [+0.18%, +1.95%] | −0.25% [−1.47%, +1.09%] |
| Two-step limit model (P(fill), then outcome if filled) | identical to the one-step model | identical |

- **Asymmetric barriers** are a little better in both periods, but the intervals overlap heavily and
  neither period beats SPY on the holdout. It is promising, not proven. The code can run it
  (`PO.POLICY_SET[...].profit_k/risk_k`); the live policy is unchanged.
- **The two-step limit model** changes only the class probabilities, which were no better on the holdout
  (no-fill skill −0.040 vs −0.017). The trades are chosen on the expected return, which is the same model
  either way, so the strategy is identical. Kept as `kind="lgbm_seq"` for later.

### Model B: downside quantiles

The 10th-percentile excess-return prediction is undershot 12–15% of the time instead of 10%
(development and holdout, both policies). It understates the downside, so it should not set dip
limits until it is recalibrated (e.g. conformal on a later slice).

### Model D: sectors (rank IC of the next 20 sessions' excess return, holdout 2025–26)

| Variant | Development | Holdout |
|---|---|---|
| **Current: LightGBM LambdaRank** | +0.002 | +0.083 [−0.038, +0.183] |
| Pooled structural ridge (macro-implied push from rolling driver betas, RRG ratio/momentum, 10-session residual reversal) | −0.021 | −0.046 |
| 63-session momentum | – | −0.096 |

The structural ridge loses in both periods; LambdaRank stays (still not validated).

## Not done, and why

- **Open prices**: the price store keeps closes, highs and lows only, so adding opens means downloading every stock's
  history again.
- **Model E (EWMA smoothing, turnover buffer)**: the app trades no ETFs now, so there is nothing to smooth.
- **Venn-Abers calibration**: the quality model has no ranking power to calibrate (AUC < 0.5 on the
  holdout), so the calibration method is not what is holding it back.
- **Historical SIC sectors, VVIX, SKEW, T10YIE, DFII10, HY OAS**: not added this round. The regime block
  that was added (GPR threats and acts, EPU, the geo basket, the VIX term structure) did not help the
  ranker or the risk model, which makes more regime series unlikely to.
- **Unifying A and C**: not done, because C did not validate.
