# Events, policy and geopolitics: implementation log

Version 0.1 · 2026-10-29 · branch `portfolio-app`

The brief: add upcoming company and economic reports, policy and regulation, and geopolitical risk to AlphaDesk's
data and models, measure whether they help out of sample, and keep everything point in time. This log records what
exists, what was measured, and what is next. Nothing here claims a performance gain: none has been measured yet.

## Baseline

- **Original baseline, kept:** every model cache, the stacking study's out-of-fold files, the study results and a
  consistent snapshot of the app database, copied to `backups/retrain_2026-10-29/` before anything was refitted.
- **Retrained baseline:** `python -m experiments.retrain_all` refits every model from scratch on the current code
  (ensemble out-of-fold scores, the model zoo, the daily decision, the signals batch, the sector model and the fund
  forecasts). Log: `backups/retrain_2026-10-29.log`. It adds no event data, so it is the "corrected existing model"
  (ablation A) once the bug fixes below are made, and the plain baseline before them.
- There is no `AGENTS.md` in the repository; the code, `docs/REVIEW_SESSION_HANDOFF.md`,
  `docs/IMPROVEMENT_BACKLOG_2026-10-28.md` and `docs/AlphaDesk_Model_Design.pdf` are the references.

## Documented weaknesses, checked in code

| Weakness | In code | Status |
|---|---|---|
| P(profit) calibrated on XGB percentiles, applied after ensemble scoring | `stocks/daily.py` `apply_ensemble` recomputes `p_win` with the XGB-fitted trade-quality model (`quality_matched: False`) | confirmed; not fixed yet |
| Fallback and ensemble percentiles mixed without a shared calibration | `daily.entries` sorts stocks scored by either together | confirmed; not fixed yet |
| 2025-on treated as a clean holdout | `experiments/stacking_study.py` docstring already says both periods are development evidence | confirmed as documented; kept that way |
| Confidence intervals treating overlapping dates or correlated stocks as independent | `research/walkforward.py` Holm p-values from an IID t; the stock studies use date-block bootstraps | confirmed for the research forecasts; not fixed yet |
| Universe membership not fully point in time | `stocks/panel.py` / `data.py`: point-in-time intervals where available, today's members before that | confirmed ("partial"); reported, not fixable without more history |
| Temporal leakage in stacking and score-to-return maps | stacking weights and maps are fitted walk-forward on out-of-fold scores (`stacking_study.py`) | not found; to be re-audited with the calibrators |

## Built in v0.1

`tradingagents/events/` (new):

- `store.py`: append-only `events.sqlite3` in the data cache. `events` keeps every version of a scheduled release
  (a reschedule is a new row with `revision_of`); `observations` keeps every vintage of an index value (a revision is
  a new row); `fetches` records each source's outcome. Triggers refuse UPDATE and DELETE. Fields follow the brief's
  schema (event ids, source URL, evidence excerpt, published / available / first-seen / ingested / scheduled /
  effective times, status, entities, value, vintage, hash, extraction version, reliability, missing reason).
- `sources.py`: adapters, each reporting `ok`, `unavailable` or `error` without stopping the others:

| Source | What | Coverage stored | Notes |
|---|---|---|---|
| BLS `bls.ics` | CPI, Employment Situation, PPI, JOLTS, ECI, Productivity release times | 2025 onward (123 releases) | needs a contact User-Agent (reuses `SEC_EDGAR_USER_AGENT`) |
| BEA schedule page | GDP, Personal Income and Outlays (PCE), trade | upcoming only (9) | history needs another source (ALFRED release dates) |
| FOMC calendar | statement times (14:00 New York, last meeting day) | 2021-2027 (57 meetings) | |
| GPR daily (Caldara-Iacoviello) | GPRD, acts, threats | 1985 onward (45,822 values) | updated weekly, revisable: stored as vintages |
| EPU monthly (Baker-Bloom-Davis) | news-based US index | 1985 onward (501 months) | free with attribution |

- `features.py`: market-wide features for a cutoff and the 20-session holding window from the next-close entry:
  sessions to and since each release kind, releases inside the window, GPR and EPU level, trailing z-score, change,
  staleness and missingness. **Strict** mode uses only what the app had received by the decision time;
  **research** mode assumes schedules known 30 days ahead and index values available 7 days (GPR) or 38 days from
  the month start (EPU) after their period, and marks every row `exploratory` (later revisions can leak).
- The morning run syncs the sources first (Settings → Every morning → "Release calendars and policy indices").
- Tests: `tests/test_events.py` (append-only, reschedules, strict vs research, revised vintages, missing series,
  a refusing or failing source).

## Not built yet (in order)

1. Fix the two confirmed calibration issues (P(profit) on the ensemble's own out-of-fold percentiles; separate
   calibration by scoring family) as new procedure versions.
2. History for the calendars before 2025: ALFRED release dates (needs a FRED key) or BLS archives.
3. Join the event features into the panel as unranked market-wide columns (a new feature schema), then ablations
   A-E on the same folds and dates.
4. Earnings dates per company (an optional Alpha Vantage adapter; the 8-K archive gives past results dates).
5. Federal Register, TPU, OFAC, BIS, USITC and GDELT adapters; company exposure extraction from filings.
6. A market-wide SPY-vs-cash model and warning; UI for upcoming reports and event evidence.

## Commands

    python -m tradingagents.events sync                       # fetch every source once
    python -m tradingagents.events status                     # each source's last fetch and what it holds
    python -m tradingagents.events features 2026-10-28        # strict, as known now
    python -m tradingagents.events features 2025-06-02 research
    python -m experiments.retrain_all                         # back up, then retrain every model from scratch
    python -m pytest -q tests/test_events.py
