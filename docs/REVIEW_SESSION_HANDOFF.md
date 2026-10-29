# Repository review handoff

## Session 3 (2026-10-28/29): merged into portfolio-app

- `portfolio-app` was fast-forwarded to `claude/improvements-2026-10-28` at the owner's request. The untracked audit docs
  that were in the main folder were moved to `..udit-docs-backup-2026-10-28` (the branch has the committed copies).
  Restarting the app applies migrations 11-12 (a backup is taken first).
- Added: agent report Copy (before Close), Dismiss for failed runs, Delete per saved run (moved to `_deleted`);
  provider cooldowns (out of credit 6 h, free daily limit to 00:00 UTC, rate limit 10 min; saving a key clears them);
  same-inputs reuse for agents, daily decisions, every-model runs and forecasts (`force` overrides);
  progress bars on Screener/Forecasts; Models tab Decision column; morning schedule (Settings > Every morning);
  refits every `schedule.retrain_every_sessions` (20) for the decision bundle, the every-model fits and the ensemble
  weights; Track record $1-per-recommendation totals and 5-day paging.
- **Ensemble trained** 2026-10-29 00:17 (`experiments/output/stacking_stock.log`, bundle in
  `~/.tradingagents/cache/optimizer/decision_models/ensemble.json`). Development 2014-24 IC 0.032 (0.011, 0.053),
  below equal weights of the same members (0.043); 2025 on IC -0.010 (-0.055, 0.038). Not validated. The app ranks by
  it now (experimental mode). Owner decision pending: keep, switch to equal weights, or keep XGBoost.
- Full suite on the branch before the merge: 1,754 passed, 5 skipped. No browser check of the pages yet.

## Implementation session 2 (2026-10-28) — read this first

- **Where:** branch `claude/improvements-2026-10-28`, worktree `C:\Users\santo\Desktop\random_projects\Robinhood\TradingAgents-claude`, based on `3f72cbd`. The main checkout (`portfolio-app`) and the other worktrees (`codex/improvements-2026-10-28`, `quant-upgrades`) were not touched. Nothing was pushed.
- **Python:** the main checkout's venv, run from the worktree so the worktree's package is imported: `..\TradingAgents\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
- **Done:** seven commits, `8b276f5` … `7656b11`. What each fixed, with the tests that guard it, is in [IMPROVEMENT_BACKLOG_2026-10-28.md](IMPROVEMENT_BACKLOG_2026-10-28.md) §1. Migration 11 adds feedback columns; existing rows keep NULLs.
- **Full suite this session:** run after commit `7656b11` on Windows: **1,746 passed, 5 skipped, 1 deselected, 100 subtests passed** (141 s). Audit baseline was 1,662 passed, 4 failed; the 4 Windows failures are fixed (one is now a documented POSIX-only skip). Repo-wide ruff still reports pre-existing violations (backlog P2); changed files add none.
- **Not done:** the prioritized pending list, model research (needs a resource budget), the remaining page redesign, and owner decisions with the defaults applied are in the backlog §3–§4.
- **Visual verification:** still not performed. No browser control was available; do not claim it, and do not work around the earlier "blocked by policy" rejection. Page changes were checked by API tests, page-text tests and `node --check`.
- **Helper scripts** used for patches and checks are in the session scratchpad only (not part of the repo); the page-script parse check is now a CI job (`page-scripts`).
- **Next session:** start with backlog P1 (evidence gate vs study procedure; ensemble trade-quality refit once a budget is agreed), then the Phase 4 page work, then the repo-wide lint batch.

## Audit session 1 (original handoff, kept as written)

Task: review the whole repository, all model families and saved outputs, redundancy/obsolete paths, and the actual web-displayed interface. User permits multiple sessions. This is an audit/recommendation task; application fixes and deletions have not been requested.

## Current state

- Source commit reviewed: `3f72cbd`. Tracked application source unchanged.
- Completed initial repository audit: 390 tracked files, 328 Python files parsed. Full suite: 1,662 passed, 4 failed, 4 skipped, 1 deselected, 100 subtests passed. Four failures concern Windows assumptions. Ruff: 111 violations. Do not rerun merely because this handoff exists.
- Completed follow-up model lifecycle/callsite inventory, current HTML/CSS/renderer review, and read-only model-output inspection. No paid inference, large training, or account writes.
- Deliverables: `REPO_REVIEW_2026-10-28.md`, `MODEL_OUTPUT_DISPLAY_REVIEW_2026-10-28.md`, `IMPROVEMENT_PROMPTS_2026-10-28.md`, this handoff. The user's requested consolidated improvement prompt is in `MASTER_IMPROVEMENT_PROMPT.md`; it proposes future implementation, while the audit itself has not changed application code.
- Generated QA artifacts under ignored `results/review-2026-10-28/`: `output_inventory.json`, `render_fixture.py`, synthetic fixture DB/files, `server.json`. Fixture server session 67704 was stopped. No screenshots were produced.
- Existing unrelated synthetic temp fixture from the earlier audit remains at `C:\Users\santo\AppData\Local\Temp\tradingagents-web-audit-qdi91ztf\portfolio.sqlite3`; cleanup was rejected by automatic review. Do not retry cleanup to bypass that rejection.

## Confirmed priorities

1. Ledger read/validate/write races and content-based identity collapsing independent manual events.
2. Feedback and Track-record fixed-horizon outcomes differ from trained barrier-policy outcomes; raw versus net return presentation differs.
3. Calibration cutoff/spec identity/deduplication and decision-procedure evidence compatibility.
4. Display labels: total-positive probability versus beat-cash; ensemble arithmetic policy P&L versus log excess; Chronos current-close versus next-close timing; Trade last-close limit wording versus canonical next-close entry.
5. Trade API recomputes full-budget sizing rather than preserving saved capped size.
6. Missing downside estimates, inactive/partial ensemble coverage, matching ensemble OOF trade-quality calibration.
7. Local write-origin controls, Docker private-data exclusions, budget boundary, cache replacement metadata, early-close sessions and raw FRED exception redaction.
8. Clearer displayed screen hierarchy, model lifecycle, performance definitions, units/ranks, contrast and table navigation.

Important corrections already incorporated: `trade.buy_limit` still exists; do not report an undefined-field crash. Model refits under a frozen evaluated retraining procedure should preserve a cohort, with each artifact recorded; material procedure changes should start a new cohort. Risk-model “skip” versus “block” is an explicit product choice, not automatically a bug. Research challengers and differently scoped backtests are not dead merely because absent from the main web workflow.

## Remaining visual task

Live visual inspection is pending. `cua.getState()` returned no browsers/apps and “Computer Use native pipe is unavailable ... os error 2.” The installed computer-use skill's documented Sky initialization/list-apps path returned the same error. Automatic approval review rejected the isolated headless Chrome launch with **“blocked by policy.”** Do not work around the rejected launch through another execution method.

When a supported browser-control service is available in a later session:

1. Read the relevant computer-use documentation and inspect current git status first.
2. Use isolated synthetic data; never run app startup against the personal folder because startup can start daily model work. The retained fixture script constructs an isolated app and blocks external Python data requests and all fixture POSTs. Review it before reusing it; it serves a synthetic model-zoo response and is not performance evidence.
3. Inspect Trade, Portfolio, Screener, Forecasts, Decisions, Models, Track record, Settings and Ticker at desktop, 768px, 390px and 320px widths, light and dark. Check real supported viewport controls before assuming resize APIs.
4. Include populated, absent-model, partial-coverage, stale, error and loading states. Verify hierarchy, clipping, table containment, mobile decision order, chart labels, keyboard tabs/sorts and report-dialog focus.
5. Capture screenshots through supported tools and label synthetic data. Update the display report from code-derived proposals to observed visual findings.

Do not claim this visual pass has already happened. Do not rerun live provider models or training as a substitute for layout review.

## Owner decisions that can improve the next changes

Hosting/user audience; primary objective and holding horizon; missing-risk behavior; cash correction versus external contribution semantics; data/AI budget and latency; desired default workflow; external compatibility/deprecation expectations. The prompt pack contains a concise requirements template and separate implementation prompts. These questions do not block documenting the confirmed defects.
