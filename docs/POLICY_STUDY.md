# Policy study (policy-study-v1)

Trial 33; runtime 359.9 s; decision dates every 10 sessions; survivorship: partial. Holdout from 2025-01-01.

**Status: not validated.** P(target first) has no Brier skill over the base rate on the holdout (-0.015); P(stop first) has no Brier skill over the base rate on the holdout (-0.047); mean excess over cash per decision -0.94%, 95% interval [-2.58%, +0.64%] does not exclude zero; below the existing rule (-0.94% vs +1.15%)

## Outcome probabilities (Brier skill = 1 - Brier / Brier of the training base rates; > 0 beats them)

| Model : policy : period | Rows | Log loss (base) | P(target) skill | P(stop) skill | P(expiry) skill | P(no fill) skill | Mean-excess IC | q10 coverage |
|---|---|---|---|---|---|---|---|---|
| lgbm:next_close:development | 116,701 | 1.0057 (1.0187) | +0.0059 | -0.0090 | +0.0539 | – | +0.0410 | 14.3% |
| lgbm:next_close:holdout | 21,434 | 1.0973 (1.0684) | -0.0148 | -0.0466 | +0.0473 | – | -0.0230 | 13.7% |
| lgbm:limit_dip:development | 116,701 | 1.2882 (1.2740) | +0.0016 | -0.0043 | -0.0003 | -0.0077 | +0.0184 | 12.4% |
| lgbm:limit_dip:holdout | 21,782 | 1.3131 (1.2845) | -0.0311 | -0.0654 | +0.0051 | -0.0167 | -0.0058 | 12.7% |
| linear:next_close:development | 116,701 | 1.1937 (1.0187) | -0.0207 | -0.1089 | +0.0080 | – | +0.0033 | 12.8% |
| linear:next_close:holdout | 21,434 | 1.0708 (1.0684) | -0.0113 | +0.0026 | +0.0762 | – | +0.0146 | 17.1% |
| linear:limit_dip:development | 116,701 | 1.5462 (1.2740) | +0.0079 | -0.1767 | -0.0358 | -0.0626 | +0.0002 | 11.9% |
| linear:limit_dip:holdout | 21,782 | 1.3199 (1.2845) | -0.0395 | -0.0023 | +0.0118 | +0.0048 | -0.0046 | 13.9% |

Skill by fold (LightGBM, next close; all classes together):

2014: +0.0273 (n=8,881), 2015: -0.0209 (n=9,160), 2016: +0.0297 (n=9,594), 2017: -0.0192 (n=10,004), 2018: +0.0258 (n=10,664), 2019: +0.0218 (n=10,544), 2020: +0.0705 (n=10,941), 2021: +0.0201 (n=11,217), 2022: +0.0386 (n=11,495), 2023: +0.0214 (n=11,781), 2024: +0.0122 (n=12,420)

By VIX regime (development): high_vix +0.0309 (n=58,814), low_vix +0.0115 (n=57,887)

Reliability, P(stop first), LightGBM next close, holdout (forecast vs observed by decile):

0.20->0.29 (n=2180); 0.21->0.33 (n=2196); 0.22->0.33 (n=2424); 0.22->0.34 (n=2212); 0.23->0.26 (n=2197); 0.23->0.32 (n=2263); 0.24->0.28 (n=2274); 0.25->0.28 (n=2174); 0.27->0.32 (n=2173); 0.50->0.26 (n=2147)

## Strategies: mean excess over cash per decision date (5 slots, empty slot = cash)

Windows overlap (20-25 sessions, every 10), so the interval is a block bootstrap. Not annualized.

| Strategy : period | Dates | Mean excess | 95% interval | Share > 0 | 5th pct | Worst |
|---|---|---|---|---|---|---|
| lgbm_layer:development | 277 | +0.87% | [+0.13%, +1.67%] | 34% | -6.20% | -16.10% |
| lgbm_layer:holdout | 45 | -0.94% | [-2.58%, +0.64%] | 24% | -7.26% | -14.96% |
| linear_layer:development | 277 | +0.14% | [-0.11%, +0.40%] | 7% | -0.03% | -8.59% |
| linear_layer:holdout | 45 | +0.00% | [+0.00%, +0.00%] | 0% | +0.00% | +0.00% |
| existing_rule:development | 277 | +0.01% | [-0.55%, +0.57%] | 29% | -8.31% | -13.27% |
| momentum_top5:development | 277 | +0.14% | [-0.83%, +1.06%] | 55% | -12.18% | -21.47% |
| all_members_equal:development | 277 | +0.31% | [-0.22%, +0.86%] | 58% | -6.04% | -9.80% |
| spy:development | 277 | +0.99% | [+0.34%, +1.60%] | 68% | -6.39% | -26.71% |
| existing_rule:holdout | 45 | +1.15% | [-0.86%, +3.38%] | 53% | -8.63% | -12.90% |
| momentum_top5:holdout | 45 | +2.81% | [-1.00%, +7.14%] | 60% | -16.96% | -21.75% |
| all_members_equal:holdout | 45 | +0.07% | [-0.98%, +0.92%] | 60% | -5.32% | -8.69% |
| spy:holdout | 43 | +1.01% | [-0.33%, +2.42%] | 60% | -4.17% | -8.57% |

Margins tried on development (selection count = 4 per model): lgbm: chose 0.005 from {'0.0': 0.008178112616859507, '0.005': 0.008715978632584396, '0.01': 0.0077140479125501005, '0.02': 0.004916661843138538}; linear: chose 0.020 from {'0.0': 0.000426624439198857, '0.005': 0.00038895343176368073, '0.01': 0.0008737449782879249, '0.02': 0.001377824392950206}

The existing rule here is the deployed ranker and its expected-net gate (>= 0.5%), without its P(profit) and P(large loss) gates. SPY: bought at the next close, held 20 sessions, over cash.

