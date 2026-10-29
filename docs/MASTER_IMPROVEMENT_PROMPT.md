You are a senior Python engineer, quantitative-model reviewer, and web product designer working on this TradingAgents/Yieldforge repository. Turn the completed audit into practical, tested improvements to code, models, outputs, and the interface actually displayed in the web browser.

Read these first:
- docs/REPO_REVIEW_2026-10-28.md
- docs/MODEL_OUTPUT_DISPLAY_REVIEW_2026-10-28.md
- docs/REVIEW_SESSION_HANDOFF.md

Check current git status and repository instructions. Verify findings against current code; the audited baseline was commit 3f72cbd. Preserve unrelated changes. Work in small, reviewable phases and maintain the handoff so another session can continue without repeating completed work.

Preserve personal databases, account records, credentials, and historical model artifacts. Use isolated synthetic fixtures for changes and verification. Do not place brokerage orders, make paid model calls, or start expensive training without an agreed resource budget. Do not delete compatibility interfaces or historical records merely because current internal callers are absent.

1. FIX CORRECTNESS AND RELIABILITY

- Make ledger read, validation, reservation handling, and append atomic across connections. Conflicting purchases, sales, withdrawals, and reservations must respect available cash and shares.
- Give independent manual operations distinct identities; reuse an operation ID only for retries. Preserve broker/import deduplication.
- Resolve feedback and policy performance using the same frozen execution/exit policy that created the training labels. Keep fixed-horizon diagnostics separately named.
- Calibrate only compatible outcomes matured and observable by the decision cutoff. Prevent reruns from overweighting the same economic observation.
- Distinguish training-recipe, decision-procedure, and individual artifact/run identities. Bind studies and prospective cohorts to the evaluated procedure; record each fitted artifact. Scheduled refits under the same procedure should preserve the cohort; material procedure changes require compatible new evidence.
- Match the actual ranking source, ensemble construction, quality calibration, gates, ordering, sizing, costs, and execution between evaluation and deployment. Train downstream trade-quality models on matching out-of-fold scores when the ensemble replaces the base ranker.
- Require finite downside estimates for eligible buys. Preserve an explicit, visible distinction between assessed risk and an intentional configured risk bypass.
- Use one account-aware sizing service. The Trade API must not silently replace a saved capped allocation with the entire available budget.
- Fix cash edits that return success without persistence. Use explicit audited correction/flow semantics and return canonical saved values.
- Protect mutation routes with suitable Host/origin/session-token checks, JSON validation, and request limits. Exclude private data and additional environment files from Docker builds.
- Reserve AI request capacity atomically before dispatch; fence job updates by run ID; advance cache freshness only after successful writes; use actual exchange close times on half-days; redact credential-bearing request errors, including the stock FRED client.

2. STANDARDIZE MODEL OUTPUTS

Create shared model/output metadata covering lifecycle, consumer, target, units, benchmark, horizon in trading sessions, information cutoff, entry/exit clock, cost treatment, coverage, calibration, procedure ID, artifact ID, and evidence status.

Cover the forecast engine, stock signals, daily decision, model zoo/ensemble, policy outcome layer, sector model, risk research, pretrained/sequence challengers, LLM roles, and sentiment/quality/moat rules. Do not imply every research model drives trading decisions.

Correct these verified display mismatches:
- P(positive total return) is currently labeled as P(beating cash). Give them distinct names and calculations.
- Model tables mix log excess returns with arithmetic barrier-policy net P&L. Render the actual target/benchmark and compare only compatible quantities.
- Chronos current-close forecasts and next-close trade labels have different clocks. Align or explicitly distinguish their timing.
- Trade foregrounds a last-close limit while its evaluated policy enters at the next close. Use canonical execution text, dated reference prices, and fill-relative exits. Keep patient limits visibly experimental.
- Track-record headline averages and correctness labels use different cost treatments. Separate fixed-horizon diagnostics, policy replay, prospective capital accounts, and actual fills.
- Display missing/partial model coverage and the active fallback. Distinguish the equal-weight average percentile from the learned ensemble score.
- Pin pretrained revisions or record the actual resolved revision. Replace misleading “days,” “Worst drop ever,” and probability/significance descriptions with precise labels.

3. IMPROVE MODEL QUALITY THROUGH CONTROLLED EVALUATION

- Compare existing linear, momentum, XGBoost, LightGBM, and ensemble approaches against cash and cash-flow-matched SPY using identical dates, universes, execution, costs, and capital constraints.
- Preserve chronological splits and label-interval purging. Use date-level, dependence-aware uncertainty and disclose the experiment/search family.
- Report net performance, drawdown, turnover, calibration, interval coverage/width, missingness, prediction spread/ties, runtime, and model coverage. Evaluate probability/risk quality on the candidates actually selected.
- Investigate flat predictions and early stopping, explicit missing-feature indicators, continuation models conditioned on real lot state, and uncertainty corrections as controlled experiments.
- Expose optimizer convergence/residual/feasibility diagnostics and compare with a trusted reference solver.
- Evaluate LLM role assignments with frozen evidence packets: grounding, numerical consistency, structured validity, abstention, stability, latency, and measured cost. Ablate debate roles to measure incremental value.
- Treat inspected history as development evidence. Neither passing unit tests nor reaching a minimum forward-trade count proves investment value. Promote no architecture solely because it is newer.

4. IMPROVE THE WEB INTERFACE AS DISPLAYED

Keep the current framework unless a demonstrated requirement justifies migration. Use consistent spacing, readable typography, sufficient contrast, restrained status colors, contained wide tables, and clear loading/error/stale states.

- Trade: lead with permitted action, top candidate, reasons to wait, allowed size, account readiness, entry/exit timing, active model, evidence state, and cutoff. Use “Top model candidate” while experimental. Put detailed market research behind progressive disclosure.
- Portfolio: distinguish full account value, stock holdings, SGOV, settled cash, and reservations. Show holdings before a compact Record trade panel. Confirm saved values after reload.
- Screener: prioritize search, filters, rank, quality, analyst context, valuation, and freshness. Expand detailed calculations and methodology on demand.
- Forecasts: separate Results, Configure run, and Validation. Show precisely labeled outcomes, intervals with a zero reference, probabilities, and cutoffs. Expand constituent models per stock.
- Decisions: emphasize history and analysis; separate quant decisions from agents’ opinions and reduce duplication with Trade.
- Models: begin with model health, lifecycle, coverage, compatible evidence, and active/fallback status. Make the stock-by-model matrix secondary with selectable columns.
- Track record: lead with net outcomes, benchmark, matured sample, and uncertainty. Separate cohorts and label recoverable row hiding as Hide.
- Settings: group Account, Data & universe, and AI providers & cost. Collapse advanced role overrides/pricing.
- Ticker: show price, position, permitted action, and evidence first, including on mobile. Add chart range controls and a log-scale indicator; consolidate repeated entry-plan panels with source provenance.
- Complete keyboard navigation, labeled controls, accessible sorting, live job status, modal focus containment/return, and mobile navigation overflow. Load hidden tabs on demand and cancel obsolete requests.

Complete browser verification with synthetic populated, empty, partial-coverage, stale, loading, and error states at desktop, 768px, 390px, and 320px in light/dark themes. Earlier browser-control access was unavailable and an isolated headless launch was rejected by automatic approval review with “blocked by policy.” Use supported tools; do not bypass that rejection or claim visual verification without evidence.

5. REDUCE REDUNDANCY DELIBERATELY

- Classify components as active decision, active context, research-only, compatibility, archive-only, or uncalled before changing them.
- Review optimizer/auto_invest.py, optimizer/tracker.py, unused web jobs/fields, retired risk-JSON persistence, old recommendation publishing, and /api/ratings for deprecation. Preserve history and provide migration paths where needed.
- Share duplicated LightGBM/Ridge study/live adapters, artifact manifests, provenance utilities, scrubbed transport, and cache freshness interfaces.
- Preserve differently scoped backtests, immutable ledger snapshots, useful research challengers, and live report writers. Migrate still-reachable legacy stock-ideas callers before removal.
- Split the large server and inline frontend into route/service/shared presentation modules. Update authoritative documentation and then regenerate derived guides/prompts.
- Add Windows, research-dependency, and browser CI coverage; repair platform-specific tests, restore lint, and lock deployment dependencies.

6. VALIDATION AND DELIVERY

Use meaningful regressions for competing ledger operations, independent identical fills versus retries, exact budget limits, failed writes, half-day timing, future-outcome invariance, policy-aligned feedback, sizing consistency, and output-to-display semantics. Verify model fallback and save/reload behavior in the browser.

Run relevant checks as each phase stabilizes, then the full isolated suite and lint when justified by changes. Explain remaining failures accurately; do not weaken tests to hide defects. Never report old test results as newly run.

Deliver an ordered backlog and implementation batches with: problem, change, source references, acceptance criteria, verification evidence, and remaining limitations. Keep the session handoff current. Clearly distinguish completed fixes, measured improvements, research hypotheses, and deferred visual checks.

Ask only for missing decisions that materially affect dependent work: intended hosting/users, primary objective and horizon, missing-risk policy, cash reconciliation semantics, resource budget/latency, default UI workflow, and external compatibility requirements. Recommend defaults with consequences and continue independent work while answers are pending.
