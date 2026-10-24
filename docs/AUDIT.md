# Phase 1 audit: architecture, data and gaps (2026-10-23)

This is what exists before the portfolio-application rebuild, what is kept, and
what each later phase replaces. It covers the "Phase 1: architecture and data
audit and preservation" step of the specification.

## Machine (detected 2026-10-23)

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26300 |
| CPU | Intel Core i5-11400H, 6 cores / 12 threads |
| RAM | 15.8 GB |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, 4096 MiB VRAM, compute capability 8.6, driver 610.62; Intel UHD (integrated) |
| Python | 3.13.5 (Anaconda build), venv managed by uv (no pip in the venv) |
| Numeric stack | numpy 2.5.3, pandas 3.0.6. Not installed: torch, cupy, xgboost, lightgbm, scikit-learn, arch, hmmlearn, numba, jax, llama-cpp, onnxruntime |

4 GB of VRAM rules out most local language models beyond small quantized ones,
and limits GPU batches to a few hundred MB of working memory. It is enough for
batched Monte Carlo, scenario repricing and GPU gradient boosting.

## Code that exists

| Component | Where | Kept? |
|---|---|---|
| Multi-agent LLM analysis (upstream TradingAgents v0.5.2) | `tradingagents/agents`, `graph`, `dataflows` | Kept, but called less: one analyst, a shortlist, and caching (phase 2) |
| Point-in-time SEC EDGAR statements | `dataflows/vendors/sec_edgar.py` | Kept; becomes a source for point-in-time fundamental features |
| Price history (Yahoo adjusted closes, cached daily) | `optimizer/data.py` | Kept for ETF and stock research; not enough for shorts or derivatives (see limits) |
| Optimizer: mean-variance under a stress-loss cap | `optimizer/plan.py`, `solve.py` | Kept as the CPU reference baseline. Solver replaced by ADMM on 2026-10-23 and checked against the old solver |
| Crash-scenario risk model and growth target | `optimizer/scenarios.py`, `plan.py` | Added 2026-10-23. Optional; the default stays worst-case so that old risk-level percentages keep their meaning |
| Walk-forward backtest | `optimizer/backtest.py` | Kept. A one-day look-ahead was fixed 2026-10-23 (trades now happen at the close and earn returns from the next day) |
| Paper tracker ("followed plan" vs "kept as is") | `optimizer/tracker.py`, `paper_tracker.json` | Replaced by the Day 1 record (phase 6); the JSON is migrated, not deleted |
| Web app (stdlib HTTP server and one HTML page) | `webapp/server.py`, `static/index.html` | Kept as the framework; tabs are restructured (phase 7) |
| CLI | `cli/main.py`: `suggest`, `track`, `web`, `backtest` | Kept |

## Data stores that exist (all preserved)

A snapshot was taken before any migration, at
`~/.tradingagents/backups/2026-10-23_pre-migration/`, with SHA-256 hashes in
`MANIFEST.sha256`.

| File | Contents | Problem |
|---|---|---|
| `my_portfolio.json` | Share quantities, no cash field in use | No sleeves, settled versus unsettled cash, cost basis or history |
| `my_buys.json` | Recurring buys (about $27.61 a day) | Fine; becomes the Investing contribution rule |
| `my_risk.json` | `{"level": 5}` (changed from 2 at 18:52 on 2026-10-23, apparently by the web slider) | Global, not per sleeve |
| `my_universe.json` | Scan groups | Fine |
| `paper_tracker.json` | Day 1 = 2026-10-22. Daily plan and hold values and the orders filled | Stores orders but **not target weights or the recommendation that produced them**. Day 1 replay can only be labelled *reconstructed* |
| `~/.tradingagents/logs/optimizer/<date>.md` | Rendered plan | **Overwritten by every preview**, so it is not an immutable record |
| `~/.tradingagents/memory/trading_memory.md` | LLM rating log | Empty so far |
| `~/.tradingagents/cache/optimizer/closes_<date>.pkl` | Daily price cache, about 98 MB | Rebuildable |

No database exists. Phase 2 adds SQLite.

## Defects found (each fixed in the phase noted)

1. `api()` in the page swallows every HTTP 409, so a failed start looks like success. (Phase 7)
2. `refresh()` has no request sequencing, so a slow older response can overwrite a newer one after a slider change. (Phase 7)
3. Every plan preview rewrites the day's Markdown "saved" recommendation. Preview, saved, accepted and executed are not separated. (Phase 2)
4. Page load starts a tracker update that writes the ledger. It is guarded by an in-process job lock, but two server processes could double-process a day. (Phase 2: an atomic claim in the database)
5. Expected return = volatility × a shrunk Sharpe, with a prior Sharpe of 0.3 for every asset. **This gives volatile assets a higher expected return because they are volatile**, which the spec forbids. (Phase 4)
6. AI ratings become fixed Sharpe notches (±0.15 per notch), and correlated agents are counted as independent evidence. (Phase 4: Black-Litterman views with stated confidence)
7. One settings file serves every context, so settings are not per sleeve. (Phase 2)
8. "Earns X% a year" wording on estimates; short records are annualized; raw account-value ratios are shown as returns despite contributions. (Phase 6/7: time-weighted returns, no annualizing under a year)
9. Settled and unsettled cash are combined in some displays. (Phase 2)

## Instruments and data: honest limits

- Prices are Yahoo **adjusted daily closes** only. These are enough for ETF and stock
  research and long-only backtests. They are **not** enough for historically
  executable short selling (no borrow availability, fees or recalls), options
  (no historical quotes, implied volatility or skew) or futures (no individual
  contracts, rolls or margin). Those stay research-only, and any synthetic
  inputs are labelled until a data source is chosen.
- The S&P 500 list is today's membership, so it has survivorship bias.
  Backtests use the ETF list and your holdings only.
- No intraday data. The no-day-trading rule is enforced on session dates from
  an exchange calendar (phase 4).

## Robinhood Roth IRA: what is believed now (to be verified in phase 2)

To be verified against Robinhood's official documentation and kept as a dated
capability matrix. Until then, everything except long stocks and ETFs is treated
as **unverified, research-only**.

## Plan

Phases 2 to 8 follow the specification's order. Each phase ends with tests and a
short report. Nothing places live orders, and no brokerage permission is changed.
