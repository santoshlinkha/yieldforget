# Backtest: worst-case vs crash-scenario risk model, and growth targets (2026-10-23)

To reproduce: `PYTHONPATH=. .venv/Scripts/python.exe experiments/risk_model_study.py`, at code
commit `portfolio-app` after e301f50.

## Setup

- **Walk-forward, June 2008 to October 2026**, rebalanced every 21 trading days (221
  rebalances). Each rebalance sees only closes up to that day: return and crash
  estimates, covariance and crash scenarios are all rebuilt from that history.
  Trades happen at that close and earn returns from the next day on.
- **Universe:** the 39 long-lived ETFs plus your holdings. No S&P 500 single stocks,
  because today's member list would add survivorship bias. A fund joins once it has
  3 years of prices. BIL is cash.
- **Costs:** 0 and 10 bp per dollar traded. No AI ratings, since today's models know
  how those years ended.
- **Promise check:** each month's mix is held unchanged for the next 12 months, and
  its worst drop is compared with the loss limit it was chosen under (274 checks
  per strategy).

## Results (no costs; 10 bp lowers growth by 0.1 to 0.3 points a year)

| Model | Limit | Target | Growth/yr | Worst drop | Volatility | Sharpe | Worst year | Avg. invested | 12-month holds past limit |
|---|---|---|---|---|---|---|---|---|---|
| worst case | 2.5% | – | 1.7% | 1.4% | 0.8% | 0.49 | −0.8% | 23% | 0 / 274 |
| worst case | 10% | – | 3.2% | 7.6% | 3.0% | 0.52 | −4.7% | 61% | 0 / 274 |
| worst case | 20% | – | 5.0% | 12.0% | 5.6% | 0.55 | −7.4% | 88% | 0 / 274 |
| worst case | 30% | – | 6.7% | 16.3% | 7.5% | 0.58 | −10.2% | 99% | 0 / 274 |
| scenarios | 2.5% | – | 2.5% | **3.2%** | 1.5% | 0.67 | −1.9% | 46% | **37 / 274** |
| scenarios | 10% | – | 5.3% | **15.1%** | 5.3% | 0.61 | −8.9% | 93% | **11 / 274** |
| scenarios | 20% | – | 6.6% | **24.3%** | 8.3% | 0.53 | −16.9% | 100% | **4 / 274** |
| scenarios | 30% | – | 7.7% | 23.8% | 9.2% | 0.57 | −15.0% | 100% | 0 / 274 |
| worst case | ≤30% | 6% | 5.3% | 13.1% | 5.4% | 0.60 | −7.5% | 78% | 0 / 274 |
| worst case | ≤30% | 8% | 5.9% | 16.3% | 6.7% | 0.57 | −10.2% | 89% | 0 / 274 |
| worst case | ≤30% | 10% | 6.7% | 16.3% | 7.3% | 0.60 | −10.2% | 96% | 0 / 274 |
| scenarios | ≤30% | 6% | 6.1% | 16.3% | 5.9% | 0.66 | −8.3% | 84% | 17 / 274 |
| scenarios | ≤30% | 8% | 6.8% | 23.8% | 7.4% | 0.60 | −14.8% | 93% | 9 / 274 |
| scenarios | ≤30% | 10% | 7.2% | 23.8% | 8.6% | 0.57 | −15.0% | 99% | 2 / 274 |
| *VTI only* | | | 10.0% | 51.5% | 16.4% | 0.48 | −31.1% | | |
| *60/40 VTI/BND* | | | 7.9% | 33.6% | 9.9% | 0.55 | −17.3% | | |
| *Your mix now* (cash until each fund existed) | | | 5.3% | 11.8% | 5.8% | 0.57 | −5.4% | | |
| *Cash (BIL)* | | | 1.1% | 0.8% | 0.4% | 0.00 | −0.3% | | |

The scenario model's breaches came in 2008 (only the dot-com crash was known
then, and most ETFs had no crash history), in 2019 (mixes held into the COVID
crash), and in 2016 and 2020–22 at the 2.5% level (bonds and stocks falling
together).

## What this says

1. **The worst-case model keeps its promise.** No strategy's drop went past its
   limit, over the whole run or over any 12-month hold. It is conservative: at a
   30% limit the realized worst drop was 16%.
2. **The crash-scenario model does not keep its promise.** It invests more at the
   same level, but realized drops went past the limit by up to 1.5 times. A crash
   unlike any earlier one, which is the one that matters, is exactly what it
   cannot see. **It is not a safe replacement for the risk dial and stays
   optional, off by default.** Its Sharpe is better only at the lowest level.
3. **At equal realized risk, the scenario model gained nothing.** Worst case at 30%
   (16.3% worst drop) grew 6.7% a year; scenarios at 10% (15.1% worst drop) grew
   5.3% a year.
4. **The growth target works mechanically, but the growth estimates are too high.**
   The plan expected 0.3 to 1.3 points a year more than it realized, and targets of
   6, 8 and 10% realized 5.3, 5.9 and 6.7%. This is the expected-return defect in the
   audit (return scales with volatility), and phase 4 replaces that model.
5. **The optimizer is mostly a risk dial, not a source of extra return.** Its Sharpe
   (0.52 to 0.60) is close to a 60/40 mix's (0.55). It controls drawdown well, but it
   has not shown that it beats simple allocations on risk-adjusted terms.

## Limits

ETFs only: the single stocks the live plan can buy are not tested. Today's ETF list
was chosen knowing which funds lasted. No AI ratings. Rebalancing happens at the
close with a flat cost; there is no daily-buy path and no T+1 delay in the
backtest. One path of history; there is no bootstrap uncertainty yet (phase 5
adds block bootstrap).
