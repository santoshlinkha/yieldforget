# Improvement backlog and delivery record — 2026-10-28

Branch `claude/improvements-2026-10-28` (worktree `../TradingAgents-claude`), based on `3f72cbd`. It implements the
first batches from [REPO_REVIEW_2026-10-28.md](REPO_REVIEW_2026-10-28.md) and
[MODEL_OUTPUT_DISPLAY_REVIEW_2026-10-28.md](MODEL_OUTPUT_DISPLAY_REVIEW_2026-10-28.md). Finding numbers (#1–#16, D1–D10)
refer to those documents. Each finding was re-checked against the code before it was changed.

Everything here used synthetic fixtures in temporary folders. No personal database, credential or account record was
read or written; no order was placed; no paid model call or training run was made.

## 1. Completed fixes (code changed, regression tests added)

| Batch / commit | Finding | What changed | Evidence |
|---|---|---|---|
| 1 `8b276f5` | #1 ledger races | Balance and share checks for withdraw, fill, transfer, reserve run inside the same `BEGIN IMMEDIATE` transaction as the write (`ledger/accounting.py`). `fill(consumes=)` releases a reservation in that transaction. | `tests/test_ledger_integrity.py::test_conflicting_writers_*` (buy, withdraw, reserve, sell on separate connections): exactly one succeeds. The same tests fail 4/4 on the original `accounting.py`. |
| 1 | #2 identical fills | User actions carry an `op_id`; a retry returns the first event (`applied: false`), even after the first attempt spent the cash; an id reused for a different event is refused. Imports keep content/broker-reference dedupe. The page keeps one id per action until it succeeds; CLI commands get a fresh id each. | `test_two_identical_manual_buys_are_two_buys_but_a_retry_is_one`, `test_a_retry_is_answered_even_when…`, `test_imports_without_an_operation_id_keep…`, HTTP `test_two_identical_manual_fills_are_two_trades…` |
| 1 | #8 cash edits | `/api/portfolio` persists cash through `Store.adjust_cash`, which requires `flow` (contribution/withdrawal, excluded from performance) or `correction` (audited `adjustment`, not a flow). Holdings and cash are one transaction; a refused cash change saves nothing. Response and `/api/state` return the saved settled/unsettled/reserved cash; the page shows saved values after reload. | `test_cash_edits_are_saved_and_say_whether…`, HTTP `test_a_cash_edit_is_saved_reloaded_and_must_say_what_it_is` |
| 1 | #6 write origin | Every POST needs a loopback `Host` (port-matched), same `Origin` when sent, no `Sec-Fetch-Site: cross-site`, `Content-Type: application/json`, a JSON object under 8 MB, and the per-run token printed into the page (`X-CSRF-Token`). Numbers must be finite. | `test_writes_from_another_site_or_without_the_page_token_are_refused` across settings, ledger, portfolio, keys and decisions routes |
| 1 | (new) budget date | `income.budget_set_on` used the UTC date of the change; trades use the local date, so evening buys vanished from the trading budget. Now local. | `tests/test_capital.py::test_the_budget_is_what_was_set…` failed when run after 7 pm CDT before the fix; passes after. |
| 2 `5189b4b` | #7 Docker | `.dockerignore` is an allowlist of the package source; databases, backups, `my_*.json`, bars, results, every `.env*` stay out. | `tests/test_docker_context.py` (30 cases). No image was built. |
| 2 | #10 AI budget | Requests are claimed before dispatch counting in-flight requests; a limit of N allows N; a paid request is refused once no dollars remain; usage is recorded before a request leaves the in-flight set; free models stay limited by count only. | `tests/test_ai_budget_limits.py` (5 tests; all 5 fail on the original gateway) |
| 2 | #11 cache metadata | A bar file that cannot be replaced leaves `bars_meta` unchanged and logs a warning. | `test_a_file_that_cannot_be_replaced_leaves_its_dates…` |
| 2 | #12 half days | `sessions.early_closes/session_close`; order cutoff = close − 15 min; a bar is final 30 min after its close. 8-K and earnings knowledge times compare with that day's close; archived filings are re-flagged from `accepted_at` at read time. | `test_half_day_sessions_move_the_order_cutoff…` (the audit's Nov 27 14:00 case → Nov 30), `test_a_filing_or_report_after_a_half_day_close…` |
| 2 | D10 FRED | `stocks/macro.py` uses `get_scrubbed`. | `test_a_fred_failure_never_logs_or_raises_the_key` (fails on the original) |
| 2 | #9 stale workers | Workers get a per-run handle; after "stop waiting" their writes go to private copies. | `test_an_abandoned_run_cannot_overwrite_the_new_runs_progress_or_log` |
| 3 `8645834` | #13 downside | A buy needs a finite downside estimate in the policy layer and the daily decision; bypasses are explicit settings (`allow_missing_downside`, `require_downside_estimate=false`) and every bypassed line says so. | `test_a_buy_needs_a_finite_downside_estimate…` (None/NaN/±inf), `test_missing_downside_estimate_blocks_entry…` |
| 3 | D5 sizing | `stocks/sizing.position_size` is the single rule; the Trade view applies it to today's budget and never above the saved decision's size; each pick carries `size.basis`, budget, cap, slots. | `test_the_trade_view_keeps_the_saved_capped_size…` (audit case: $25 stays $25 with $1,000 available) |
| 3 | D4 Trade wording | Leads with "Top model candidate" (experimental), the tested next-close entry, allowed size, a dated reference close; target/stop as % from the fill; the patient limit is a collapsed untested alternative; "I bought this" asks for the actual fill. | `tests/test_model_outputs.py::test_the_trade_page_leads_with_the_tested_entry…` |
| 3 | D1 probability | P(total return > 0) is no longer called "beating cash" (ticker, Forecasts, fund verdicts, agents' summary). | `test_the_forecast_probability_is_of_a_positive_total…` (audit's 0.6 vs 0.4 case), page-text tests |
| 3 | Output contract | `tradingagents/model_outputs.py`: each family's lifecycle, consumer, evidence, source; each output's target, unit, benchmark, horizon, entry/exit, costs, calibration; `procedure_id`/`Identity` separating recipe, procedure and artifacts. Served at `/api/model-outputs`; verdict scores carry `measures`. | `tests/test_model_outputs.py` |
| 3 | D8 contrast | Text on the accent uses `--on-accent` (dark theme 7.4:1, was 2.5:1); light green 5.5:1 (was 3.8:1); light muted 5.6:1 (was 4.4:1). Computed ratios, not browser-measured. | Calculation in the commit |
| 4 `ff36718` | #3 feedback target | Outcomes resolved through `positions.trades` (the label engine) from the saved sigma: `policy_net/exit_on/reason`. Fixed-horizon result kept as a named diagnostic, never calibrated on. | `test_a_barrier_exit_is_scored_as_the_policy…` (audit case: +9.6% policy vs −10% fixed horizon) |
| 4 | #4 calibration cutoff/identity | Calibration needs a cutoff and `procedure_id`: exits on or before the cutoff, same procedure, one observation per (date, ticker). Raw, calibration version and calibrated numbers saved separately (it was refitting on its own output). Replays resolve only what their prices reach. Migration 11. | `test_outcomes_after_the_cutoff_cannot_change…`, `test_another_procedures_record_cannot_change…`, `test_running_a_day_again_does_not_count…`, `test_a_replay_resolves_only…`, `test_calibration_is_fitted_on_raw_numbers…` |
| 4 | #5 (part) cohorts | `daily.decision_procedure` hashes recipe, ranker/ensemble, gates, ordering, sizing, costs, execution; refits and dollar changes keep it. | `test_models_are_reused_when_only_the_dollar_amounts_change` asserts the same `procedure_id` |
| 4 | D6 track record | Summaries lead with average net of the round-trip cost (the verdict's cost), raw beside it, data dates and a by-date SE; labelled fixed-horizon diagnostic; "beat cost"/"right direction"; rows are **hidden**, not deleted. | `test_the_track_record_leads_with_net_returns…` |
| 5 `12abb6c` | D7, D2, D3 | `model_zoo.describe` (works on saved runs): coverage per model, lifecycle (decides / fallback / challenger / baseline), output definition, failures, ensemble present or not. Models tab leads with that table; matrix secondary with selectable columns, sort buttons with `aria-sort`, "Mean pct (N)". Ticker table says which model decides and each forecast's unit/clock. | `tests/test_model_display_semantics.py` |
| 5 | D9 | Chronos weights pinned only by `TRADINGAGENTS_CHRONOS_REVISION`; each zoo run records requested pin and resolved commit. | code; no model was loaded in tests |
| 5 | #5 (part) | When the ensemble ranks, the record and Trade page state that P(profit) is the base ranker's trade-quality model applied to ensemble percentiles, and the ensemble's coverage. | `test_the_trade_view_says_when_p_profit_is_a_transfer…` |
| 6 `aa4fd76` | Windows tests, #16 | POSIX mode-bit asserts gated; upgrade tests pin POSIX form (file:// case skipped on Windows); Azure forwards `max_tokens`; keys enter the process only after the file write succeeds. CI: Windows lane, research-extras CPU lane, page-script parse check. | `test_azure_forwards_the_output_cap…`, `test_a_key_that_cannot_be_written_is_not_used_either` |
| 7 `7656b11` | UI behaviour | Tabs load on first view (start-up was 16 loaders); newer reads abort older ones; WAI-ARIA keyboard tabs; ticker dialog focus containment, inert background, focus return; Portfolio account breakdown (stocks, SGOV, cash settled/unsettled/reserved/available; partial when a price is missing). | `test_hidden_tabs_load_on_demand…`, `test_the_portfolio_view_separates…` |

## 2. Measured improvements

None. No model was retrained or re-evaluated in this session, so there is no new performance evidence. Do not read the
fixes above as improving returns; they make records, labels and evidence trustworthy.

## 3. Pending work, in priority order

| # | Item | Source | Acceptance criteria | Blocked on |
|---|---|---|---|---|
| P1 | Evidence gate must require the live procedure to match the study's: candidate ordering (study orders by expected net, live by score percentile), chosen gates, ensemble substitution | `stocks/daily.py:111` `evidence`; `experiments/decision_study.py:120`; `stocks/shadow.py:89` | `evidence()` compares the live `procedure_id` with the study's recorded one; a mismatch makes the mode experimental with that reason | Owner: primary objective/horizon |
| P1 | Refit trade quality on the ensemble's out-of-fold percentiles, or stop applying it to them | `daily.apply_ensemble`; `experiments/stacking_study.py` | Quality model trained and calibrated on ensemble OOF; reliability table on held-out years; `quality_matched` true | Resource budget (a stacking run) |
| P2 | Content-addressed manifests for research evidence and stacking caches (code, data snapshot, features, folds, cutoff, params, versions, seed); reject incompatible reuse | `research/engine.py:69,:400`; `experiments/stacking_study.py:99`; artifact writers in forecast engine, pipeline, daily, policy bundles | A changed feature list or fold set invalidates the cache in a test | — |
| P2 | Dependence-aware IC inference (date-block bootstrap or HAC) and a registered search family; disclose exploratory searches | `research/walkforward.py:161,:257`; `research/engine.py:217` | Overlapping-label synthetic series: IID t-stat significant, block bootstrap not | — |
| P2 | Checkpoint identity includes prompt/schema/code; immutable run IDs and atomic writes for graph logs | `graph/trading_graph.py:155,:468,:504` | Resume refused across a code change; rerun keeps both logs | — |
| P2 | Destructive migration (`retire_investing`) atomic or resumable | `ledger/store.py:197` | Injected failure between deletion and restatement leaves a recoverable state | — |
| P2 | Windows credential ACL check (not mode bits) | `llm_clients/chatgpt_auth.py:75`, `webapp/keys.py` | Test reads the ACL (icacls/pywin32) and finds no broad read grant | — |
| P2 | Restore the repo-wide lint gate (CI claims clean; ~110 pre-existing violations, incl. 10 B023 closure captures needing review) | `ruff check . --statistics` | `ruff check .` exits 0; B023 cases reviewed individually | — |
| P2 | Lock deployment dependencies (`uv.lock` is git-ignored; Docker/CI resolve lower bounds) | `.gitignore`, `Dockerfile`, `pyproject.toml` | Committed lock or constraints file used by Docker and CI | Owner: compatibility policy |
| P3 | `p_excess_positive` beside `p_total_positive` in the forecast engine | `research/forecast.py:110` | Stored and shown with its own label | Research DB migration |
| P3 | Signal pipeline and sector outputs in `model_outputs` (each event definition) | `stocks/pipeline.py:70,:171` | Every displayed probability names its event | — |
| P3 | Split `webapp/server.py` (1,900+ lines) into route, service and job modules; extract shared page JS (formatting, tables, api) | `webapp/server.py`, `static/*.html` | No behaviour change; tests unchanged | After the UI batches settle |
| P3 | Deprecate `/api/ratings` (no UI caller) via a versioned notice; archive `optimizer/tracker.py`, `optimizer/auto_invest.py` (test-only); drop unused job/fields in `server.py`; split risk-JSON persistence | §5 below | Callers migrated; history readable | Owner: external compatibility |

### Model research (Phase 3) — not started; needs an agreed resource budget

Controlled comparison of linear, momentum, XGBoost, LightGBM and the ensemble under one frozen procedure (same dates,
universe, next-close execution, capital constraints, costs), against cash and cash-flow-matched SPY; chronological
splits with label-interval purging; date-block uncertainty; declared search history. Metrics: net and benchmark-excess
return, drawdown, turnover, calibration and interval coverage, selected-candidate results, missingness and coverage,
prediction spread/ties/effective trees, runtime/memory/inference cost. Hypotheses to test, not claims: explicit
missingness indicators; lot-state continuation model; conformal/rolling uncertainty corrections; optimizer convergence
and residuals against a reference QP solver; LLM roles on frozen evidence packets (grounding, numeric consistency,
schema validity, abstention, stability, latency, cost) with role ablations. Existing historical studies remain
development evidence; only forward records of the same `procedure_id` are prospective evidence.

### Web display (Phase 4) — remaining

Done: Trade lead card, Portfolio account breakdown, Models lifecycle-first, Track record net-first and Hide, keyboard
tabs, sortable model columns, dialog focus, on-demand loading and cancellation, contrast tokens.

Remaining: Trade (account readiness and evidence status beside the candidate; market research behind disclosure);
Portfolio (holdings before a compact Record trade panel); Screener (toolbar, default columns, row details); Forecasts
(Results / Configure run / Validation; constituent models per stock); Decisions (history and analysis, quant vs
agents, less overlap with Trade); Settings (Account, Data & universe, AI providers & cost; Advanced collapsed);
Ticker (summary first on mobile, chart range controls and a visible log-scale indicator, one entry-plan panel with
quant/agent provenance); global typography/spacing, sticky identity columns, explicit empty/partial/stale/loading/error
states; live job status announcements.

### Visual verification — still pending

No supported browser control was available in this session, so **no page was rendered**. Changes were checked by API
tests, page-text tests and `node --check` of every inline script. Desktop, 768, 390 and 320 px in light and dark
themes remain to be inspected, with populated, absent-model, partial-coverage, stale, error and loading states. Do not
work around the earlier "blocked by policy" rejection of a headless launch.

## 4. Owner decisions (with the defaults used)

| Decision | Default applied here | Consequence of the default |
|---|---|---|
| Hosting / users | Single user on loopback | Origin/token checks suffice; multi-user hosting would need authentication and per-user stores |
| Primary objective and horizon | Unchanged: 20-session barrier trades vs cash and SPY | Procedure ids and evidence are about this policy; a change starts a new cohort |
| Missing-risk behaviour | Missing downside **blocks** buys; uninformative risk model still **skips** (existing `risk_gate_when_uninformative`) and says so per line | Fewer candidates when quantiles are missing; flip either setting explicitly |
| Cash reconciliation | The user must choose flow vs correction for every cash edit; corrections are not flows (same as share corrections) | A correction changes measured profit; if corrections should restate opening capital instead, change `adjust_cash` to record `opening_cash` |
| Resource budget | None: no training, studies or paid calls | Phase 3 and the ensemble trade-quality refit wait |
| Preferred UI workflow | Trade-first overview kept | — |
| External compatibility | API fields kept where possible (`buy_limit`, `expected_net`/`p_win` on fund scores); `avg_return` in the track-record summary was **removed** in favour of `avg_net_return`/`avg_raw_return` | Any external reader of `summary.*.avg_return` must switch |

## 5. Component classification (no removals made)

| Component | Class | Note |
|---|---|---|
| `stocks/daily.py`, `stocks/trade.py`, `stocks/sizing.py`, `stocks/positions.py`, ensemble in `stocks/model_zoo.py` | Active decision | |
| Forecast engine, signal pipeline, policy outcome layer, sector model, LLM team, sentiment/quality/moat | Active context | Shown, decide nothing (policy layer abstains unless validated) |
| `research/risk_models.py`, challenger studies, `experiments/*` | Research-only | Keep reproducible runners |
| `trade.buy_limit` field, fund-score `expected_net`/`p_win` names, `recommendations.history/load`, `/api/ratings` | Compatibility | `/api/ratings` has no page caller |
| Stored recommendations, tracker files, immutable snapshots | Archive-only | Keep readable |
| `optimizer/tracker.py`, `optimizer/auto_invest.py`, `research.engine.expected_returns`, `stocks.models.event_model`, `decision_layer.portfolio`, `recommendations.save/official/decide` | Uncalled (production) | Test-only or library surface; archive deliberately after checking external users |

## 6. Verification record

See the handoff for the full-suite result of this session. Focused suites were run after each batch; each new
regression that guards a reproduced defect was also run against the original code and failed there (ledger races,
AI budget limits, FRED redaction), except where noted.
