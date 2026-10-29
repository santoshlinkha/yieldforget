# Repository review — 2026-10-28

Reviewed source commit: `3f72cbd`. A follow-up working-tree check found no changes to tracked source; the full-suite results below are retained from the completed audit rather than rerun unnecessarily.

For the later model/output inventory, read-only saved-output checks, redundancy analysis and detailed web presentation review, see [MODEL_OUTPUT_DISPLAY_REVIEW_2026-10-28.md](MODEL_OUTPUT_DISPLAY_REVIEW_2026-10-28.md). Current rendered-browser verification remains pending; see [the session handoff](REVIEW_SESSION_HANDOFF.md).

## Assessment

The highest-value improvements are accounting correctness, consistency between model outcomes and live feedback, and a verifiable link between a decision and its supporting evidence. Address these before expanding the model catalog. The web app also needs reliable edits and clearer presentation of the active model, actionable decision, and uncertainty.

The repository has useful foundations worth preserving: an append-only ledger, shared trade simulation, chronological training splits with label purging, prediction records, immutable recommendation artifacts, provider adapters, and extensive isolated tests. Those foundations do not yet cover several important boundary cases below.

This review changes documentation only. It makes no claim that a model or strategy is profitable.

## Scope and verification

- Inventoried all **390 tracked files**. Parsed all **328 Python files**, containing **56,658 physical lines**, with no syntax errors. Scanned the tracked source, tests, configuration, workflows, experiments, and documentation; performed deeper manual review of the execution paths discussed below. This was a repository-wide audit, not an assertion that every line received equal manual scrutiny.
- Reviewed both web pages and existing screenshots. Verified HTTP behavior with isolated synthetic fixtures. **No live browser layout or accessibility session was performed**; responsive and accessibility findings are based on source structure and screenshot inspection.
- Used synthetic temporary databases for behavioral reproductions. Did not inspect personal portfolio contents, print secrets, place orders, make paid model calls, or run large training studies.
- Existing audits and model documents supplied context; current code was checked independently because some documents are stale.

| Check | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q --durations=15` | **1,662 passed, 4 failed, 4 skipped, 1 deselected; 100 subtests passed**, 111 warnings; 177.41 seconds |
| `.venv/Scripts/ruff.exe check . --output-format concise` | **111 violations**; 51 marked automatically fixable |
| `uv pip check --python .venv/Scripts/python.exe` | All 133 installed packages compatible |
| AST parse of all tracked Python | Passed |
| Additional focused backend checks | 53 passed |
| Additional focused model checks | 65 passed; 89 subtests passed |

The four suite failures are Windows-specific: two auth tests require POSIX mode `0600`, and two update-command tests expect a command although implementation intentionally returns `None` on Windows. They existed in the unmodified code reviewed here. This does not establish whether Windows credential ACLs are sufficiently restrictive; test that separately. Skips include POSIX-only checks and unavailable optional Bedrock support. The default suite excludes integration tests.

## Verified defects and gaps

P1 means high priority for trustworthy records or decisions. P2 means important reliability or evaluation work. Ordering reflects impact and dependencies, not estimated exploit likelihood.

### 1. P1 — Concurrent ledger operations can overspend available cash

**Evidence:** `tradingagents/ledger/accounting.py:133`, `:146`, `:161`, and `:203`. Withdrawals, fills, and reservations read state before their write transaction. A transaction around only the append does not protect the preceding validation.

**Reproduction:** two concurrent strict $80 buys against $100, using distinct operation identities, both succeed and leave **−$60 settled cash**. The synthetic reproduction synchronized both state reads before either append. The same pattern threatens share availability.

**Fix:** place read, validate, and append inside one `BEGIN IMMEDIATE` transaction, using the existing connection/transaction discipline. Include reservation release/consumption in that operation where appropriate. Test competing buys, sells, withdrawals, and reservations across independent connections. Exactly one conflicting operation should succeed.

### 2. P1 — Separate identical manual fills collapse into one event

**Evidence:** `tradingagents/ledger/accounting.py:91`, `:103`, `:122`, `:161`; `tradingagents/webapp/server.py:1621`. Default event identity hashes content, while the manual endpoint supplies no independent operation identity.

**Reproduction:** submit two distinct manual purchases of one SYNTH share at $80 on the same date, against $1,000. Both return the same event ID; holdings remain one share and cash $920. Two legitimate fills should produce two shares and $840. Identical manual contributions have the same identity problem.

**Fix:** assign an operation ID to each new user action; reuse that ID when retrying that action. Preserve broker/source IDs and suitable content deduplication for imports. Return whether an operation was newly applied or was a retry. Add tests for both legitimate duplicates and repeated delivery of one request.

### 3. P1 — Live calibration learns a different outcome from the original model

**Evidence:** `tradingagents/stocks/feedback.py:84`, `:121`; `stocks/daily.py:258`; `stocks/study.py:47`. Training predicts the barrier policy's net trade result, which can exit early. Feedback resolves fixed-horizon returns and uses them to recalibrate those policy predictions.

**Reproduction:** entry 100; two subsequent closes at 110; later closes at 90 through the horizon; daily sigma 0.01; round-trip cost 0.004. The shared barrier simulator exits at its target after two sessions for **+9.6% net**. Feedback's horizon-end result is **−10.4% net**. Calibration can learn the opposite answer.

**Fix:** freeze the policy and execution specification with every prediction, and resolve it through the same trade engine used to create labels. Store fixed-horizon returns as separately named diagnostics. Activate calibration only when prediction and outcome specifications match.

### 4. P1 — Calibration ignores historical cutoff and model identity

**Evidence:** `tradingagents/stocks/feedback.py:97`, `:111`; `stocks/daily.py:543`. Calibration reads all matured rows without a knowledge cutoff, model/policy/horizon filter, or a once-per-economic-observation selection rule. Historical `daily.run(as_of=...)` still applies it. Repeated runs for the same cutoff create distinct decision IDs and can give that economic observation extra weight.

**Impact:** a historical replay can consume outcomes observed later; changed models or ensembles can inherit another model's correction.

**Fix:** require an explicit knowledge cutoff and decision-specification ID. Select outcomes both matured and available by that cutoff, with compatible targets. Define one canonical observation per date/ticker/specification or explicit sampling weights. Store raw predictions, applied calibration version, and corrected predictions separately.

**Acceptance:** adding future outcomes cannot change an earlier replay; adding another model's outcomes cannot change calibration; repeating a cutoff does not inflate its effective sample size.

### 5. P1 — Validation can belong to a different decision pipeline

**Evidence:** `tradingagents/stocks/daily.py:107`, `:144`, `:150`, `:155`, `:377`, `:439`; `experiments/decision_study.py:120`; `stocks/shadow.py:89`.

The evidence gate checks several useful identities, but does not require live gates to match the study's chosen rule. The study orders candidates by expected net return, while live entries order by score percentile. Ensemble substitution changes ranking/maps without being covered by the fitted-model code hash. Shadow identity is largely its implementation version and exit policy. At `daily.py:456`, ensemble percentiles also feed a trade-quality model trained on the base XGBoost ranker's out-of-fold percentiles; that transfer needs matching ensemble out-of-fold training/calibration and evaluation.

**Fix:** distinguish the **training recipe**, **decision procedure**, and **individual fitted artifact/run**. The procedure includes ranking and ensemble construction, calibration and retraining rules, gates, candidate ordering, sizing, risk behavior, feature/data contract, and execution policy. Bind the historical study and forward cohort to that procedure, while recording each fitted model, ensemble-weight and calibration artifact hash on its individual decision. Routine scheduled refits under the same evaluated procedure should not restart the forward cohort; material procedure changes should. Keep reusable trained models when only dollars change. Train trade quality on out-of-fold scores from the ranking procedure actually used.

### 6. P1 — Local write API lacks request-origin defenses

**Evidence:** `tradingagents/webapp/server.py:1659`, `:1771` and the mutation routes. A synthetic handler accepted a settings write with a foreign `Origin`, foreign `Host`, and JSON sent as `text/plain`, returning 200 and executing the write.

**Fix:** validate allowed loopback Hosts and same-origin requests; require an unpredictable session/CSRF token for mutations; enforce JSON content type and bounded request bodies. Add negative tests to every mutation family. Loopback binding alone is insufficient server-side request validation. Actual cross-site browser delivery depends on browser protections and was not demonstrated here. These changes follow [OWASP's CSRF guidance](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html).

### 7. P1 — Docker build context can include personal records and extra env files

**Evidence:** `Dockerfile:10` copies the entire context into the builder, and `Dockerfile:28` copies `/build` into the runtime image. `.dockerignore` excludes `.env` and a few generated directories, but not `.env.enterprise`, other private env files, `my_*.json`, databases and sidecars, backups, bars, data, or recommendations.

**Impact:** building from this working directory can embed personal records or additional credential files even though Git ignores them. No image was built during this review.

**Fix:** prefer allowlisted source/package copies; explicitly exclude private env files, account records, SQLite files/sidecars, caches, and backups. Keep example env files available as needed. Use a synthetic build-context test to inspect the resulting image. Git ignore rules are separate from [Docker build-context exclusions](https://docs.docker.com/build/concepts/context/).

### 8. P2 — Cash edits report success without persisting

**Evidence:** `tradingagents/webapp/static/index.html:596`, `:1545`; `webapp/server.py:1780`. Holdings sends the cash field, but `/api/portfolio` adjusts only positions.

**Reproduction:** a temporary account with cash 10 accepts a holdings POST containing cash 1234 and returns 200; saved cash remains 10.

**Fix:** provide an audited cash adjustment/reconciliation operation with a clear effective date and reason, then return canonical saved state. Define whether a balance difference is an external flow or a correction; silently treating all changes as deposits would distort performance. Verify by saving, reloading, and checking the ledger and performance views.

### 9. P2 — Old jobs can overwrite a new job's progress

**Evidence:** `tradingagents/webapp/server.py:83`, `:107`. Workers receive shared `Job` state. Stopping the wait allows another run while the original worker continues; only final result assignment is generation-guarded.

**Reproduction:** the old worker overwrites the restarted job's progress and log.

**Fix:** create state per run ID and fence every status update. Distinguish stopping observation from canceling computation. Use cooperative cancellation where supported, and fence persistent side effects before allowing replacement work.

### 10. P2 — Request budget permits a call at exact exhaustion

**Evidence:** `tradingagents/ledger/ai_gateway.py:266`. The callback checks already consumed usage with a strict inequality and does not atomically reserve the upcoming request.

**Reproduction:** daily and monthly request limits of one, with one request reserved, allow **two** sequential starts/records; the third is blocked.

**Fix:** atomically claim each request before dispatch, accounting for in-flight calls. Estimate/limit token cost before dispatch where the provider supports it, then reconcile actual usage. Document that dollar estimates cannot guarantee a hard spend ceiling without an enforceable per-request upper bound. Test exact limits, parallel analysts, errors, and retries.

### 11. P2 — Failed Windows cache replacement still advances freshness metadata

**Evidence:** `tradingagents/dataflows/local_store.py:156`, `:169`, `:174`, `:206`. After all rename retries fail, the temporary file is removed but metadata is updated as though the replacement succeeded.

**Reproduction:** forced replacement failures leave the file ending 2026-10-26 while `bars_meta` claims 2026-10-27. A later read can regard stale data as current.

**Fix:** surface replacement failure and update metadata only after success. Reconcile metadata with actual file bounds/hash when inconsistent. Add a persistent `PermissionError` regression alongside the existing successful-write tests.

### 12. P2 — Early-close sessions use an ordinary-day order cutoff

**Evidence:** `tradingagents/ledger/recommendations.py:39`, `:57`, `:66`. The cutoff is always 15:45 New York time.

**Reproduction:** publication at 14:00 ET on 2026-11-27, with information through Nov 25, is assigned to Nov 27 and may execute that date. Regular trading actually closes at 13:00 ET that day, according to the [official NYSE/ICE calendar](https://ir.theice.com/press/news-details/2025/NYSE-Group-Announces-2026-2027-and-2028-Holiday-and-Early-Closings-Calendar/default.aspx).

**Fix:** use actual exchange open/close timestamps and a cutoff offset from each session's close. Test half-days, DST, unscheduled closures, and publication exactly at cutoff. The fixed 16:30 freshness rule in the local bar store also delays half-day availability.

### 13. P2 — Missing downside estimates pass the policy risk gate

**Evidence:** `tradingagents/stocks/decision_layer.py:57`, `:89`. Mean return is required; missing `q10_excess` skips the downside check.

**Reproduction:** with `validated=True`, mean excess 0.02/0.01 for next-close/limit policies, and both downside quantiles `None`, the result selects `next_close` and marks both choices eligible. Current generally unvalidated status reduces immediate exposure.

**Fix:** require finite downside estimates for eligible buys. Test `None`, NaN, infinity, missing features, and unavailable price paths. Keep an explicit abstention reason.

### 14. P2 — Research evidence and cached predictions have incomplete identities

**Evidence:** `tradingagents/research/engine.py:69`, `:400`; `experiments/stacking_study.py:99`. ETF evidence is selected by model name/horizon/universe; stacking caches use model names and quick/full suffixes.

**Impact:** a changed feature set, model implementation, data snapshot, or fold definition can reuse earlier calibration or out-of-fold scores.

**Fix:** content-addressed manifests containing code, data, feature/label schema, fold boundaries, availability cutoff, parameters, library versions, and seed. Reject incompatible cache/evidence reuse and display artifact identity in model diagnostics.

### 15. P2 — Statistical evidence can overstate independence

**Evidence:** `tradingagents/research/walkforward.py:161`, `:257`; `research/engine.py:217`. IC significance uses an IID t-statistic despite overlapping/serially dependent observations; Holm correction covers the current batch, rather than the entire exploratory history suggested by the module description. Those p-values influence evidence weights.

**Fix:** use dependence-aware block bootstrap or HAC inference at the date level; register the experiment/search family and report exploratory searches explicitly. Preserve chronological label purging. Keep selection and evaluation separate: [scikit-learn's nested-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) explains selection optimism, while [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) explains chronology. This project's overlapping labels still require its interval-based purging rather than blindly replacing it with an ordinary splitter.

### 16. P2/P3 — Provider controls and key persistence are inconsistent

**Evidence:** `tradingagents/llm_clients/azure_client.py:8` omits `max_tokens` from its forwarding allowlist. A mocked construction requesting 123 forwards no output limit. Existing output-cap tests do not exercise every provider adapter. `webapp/keys.py:63` changes process environment before file persistence succeeds.

**Fix:** contract-test actual constructor forwarding across all adapters, with explicit provider-supported exceptions. Persist keys atomically before changing process state, or roll back on failure. Test Windows credential ACLs using platform-appropriate mechanisms rather than relying on POSIX mode bits.

## Model improvements after correctness fixes

| Area | Recommended improvement | Evidence required before promotion |
|---|---|---|
| Model selection | Keep cash/SPY, momentum, and regularized linear baselines; compare the existing XGBoost ranker, LightGBM and ensemble under one frozen decision contract | Same dates/universe/execution/capital/costs; net excess and drawdown intervals; then a compatible forward cohort |
| Weak or tied scores | Show unique-score fraction, ties, effective tree count, prediction spread, missingness and drift; investigate early-stopping behavior | Stable improvements across chronological folds and selected candidates; do not infer quality from more distinct scores alone |
| Risk target | Decide whether the risk objective is a volatility-scaled event, fixed percentage/dollar loss, or policy downside; calibrate on the selected candidates | Reliability/Brier/log-loss versus base rates, tail outcomes, and uncertainty by date/regime |
| Uncertainty | Study rolling/conformal corrections using matured, compatible policy residuals | Coverage and width by year/regime/selection, with dependence and distribution shift disclosed |
| Fundamentals/data | Add missingness indicators; measure restatement, sector-history, unavailable delisting and missing-outcome sensitivity | Point-in-time feature ablations and explicit optimistic/pessimistic sensitivity bounds |
| Existing positions | Train continuation estimates on remaining horizon, existing barriers and actual lot state | Compare with the deterministic hold/exit policy in the same capital-constrained simulator |
| Optimizer | Return convergence status, iterations, primal/dual residuals and feasibility; compare with a trusted QP reference | Ill-conditioned covariance, tight stress caps, infeasible inputs, and solver disagreement cases |
| LLM agents | Benchmark per-role model/effort choices and ablate debate roles using frozen evidence packets | Grounding, schema validity, disagreement stability, latency, tokens/cost and incremental decision value |

There is insufficient evidence from a code scan to name a new architecture or provider model as the best performer. The existing catalog is broad enough for controlled comparisons. Foundation-model contamination and licenses need artifact-specific review before a study is treated as independent evidence.

Twenty forward trades can satisfy a configured minimum but do not by themselves demonstrate reliable alpha or rare-loss calibration. Report independent dates, overlap, sample uncertainty, and observed costs alongside the trade count. Previously inspected historical periods remain development evidence.

The current `daily.DEFAULT_RULES` sets uninformative risk behavior to **skip** (`stocks/daily.py:54`), while older audit documents describe default blocking. This is an explicit product choice, not automatically a defect. Resolve the intended policy and make settings, validation identity, documentation, and UI agree.

## Web UI improvements

### Decision information and financial clarity

1. Put a compact decision summary first on Trade and Ticker: action, reasons to wait, current decision source/version, information cutoff, freshness, evidence status, and available cash. The mobile ticker DOM currently places the full research column before the verdict/account/entry plan (`ticker.html:44`, `:249`).
2. Use one vocabulary for the active ranking source. `index.html:499` and `ticker.html:741` say displayed models decide nothing; `index.html:851` says the ensemble orders the Trade picks. Show champion/challenger/fallback status consistently.
3. Keep account equity, settled cash, reservations, sleeve targets, and recurring purchase amounts visibly distinct. State the effective date when saving an adjustment.
4. Present expected outcome, downside estimate, interval, net costs, and benchmark definition together. A score percentile is not a probability. A risk gate that was skipped needs a visible reason.
5. Give Models a comparison table with training/evaluation cutoffs, sample sizes, target, validation state, cost/latency, and compatible forward performance. Refresh the older screenshots and generated guides after these changes.

### Interaction, performance and accessibility

- **Load on demand:** dashboard startup invokes 16 feature loaders plus the initial state request, with further conditional/nested calls, including work for hidden tabs (`index.html:1837`). Cache/deduplicate reads, abort obsolete requests, use timeouts/backoff, and maintain one poller per job ID (`index.html:645`, `ticker.html:924`).
- **Preserve context:** retain selected ticker/tab and scroll position; show unsaved/saving/saved/error states. Confirm success from returned canonical state rather than HTTP 200 alone.
- **Keyboard access:** complete tab/panel associations and keyboard movement (`index.html:313`, `:653`); use actual sort buttons with `aria-sort` (`:870`); label fields and announce job status without stealing focus.
- **Reports:** use a native dialog or equivalent complete focus management for the ticker modal (`ticker.html:746`), including focus containment and return to its trigger. [W3C's dialog technique](https://www.w3.org/WAI/WCAG22/Techniques/html/H102) describes those behaviors.
- **Responsive tables:** constrain horizontal scrolling to table cards at all widths. Review the summary-first layout at 320, 390, 768 and 1440 pixels with real keyboard and browser checks.

These changes can be made within the current app. A framework migration should depend on hosting/team requirements, not be a prerequisite for fixing these defects.

## Code, delivery and maintainability

- Split the large server into route groups, services and job management. Extract shared API/error handling, formatting, tables, charts and modal utilities from the two HTML files. Add typed, finite-valued request/configuration boundaries.
- Add Windows CI, a research-extras CPU lane, and browser contract tests. Existing CI covers Linux Python versions and a timezone variant, but installs only dev extras. Preserve the clean-install and wheel smoke checks.
- Resolve the four platform test failures accurately. Gate POSIX-specific assertions, verify Windows behavior separately, and add suitable credential-permission coverage.
- Restore the lint gate. Many violations are mechanical, but closure-capture warnings deserve review. Avoid an unrelated whole-repository formatting rewrite in the same patch.
- Commit a reproducible application dependency lock or equivalent deployment constraints. `uv.lock` is currently ignored; Docker and CI resolve broad lower-bound requirements anew. Keep optional heavyweight model runtimes separate from the core install.
- Include prompt/schema/code identity in checkpoint compatibility (`graph/trading_graph.py:155`). Config-only identity can resume a workflow across material code changes.
- Give graph run logs immutable run IDs and atomic writes; the ticker/date JSON log is opened with `w` (`graph/trading_graph.py:468`, `:504`), so reruns overwrite provenance.
- Make destructive ledger migrations atomic or resumable. `ledger/store.py:197` commits retirement/deletion before subsequent restatement; add injected-failure recovery coverage for the boundary.
- Review warning handling: all-NaN future paths, flat-score calibration, and poorly conditioned fits should produce explicit diagnostics/abstentions when relevant. Do not globally hide warnings as a substitute.
- Update the README to introduce the current web app and connect installation, research extras, CLI, architecture, and operating documentation. Existing screenshots depict older navigation and branding.

## Suggested implementation order

| Stage | Work | Completion evidence |
|---|---|---|
| 1 — Record correctness and boundaries | Findings 1, 2, 6, 7, 8, 10, 11, 12; platform tests and lint | Concurrent/identity tests, negative HTTP tests, cash round trip, failed-write tests, half-day boundaries, clean safe build context |
| 2 — Decision/evidence contract | Findings 3, 4, 5, 13, 14, 15; exact shared execution outcomes | Future-data invariance, full-spec invalidation tests, policy-aligned feedback, compatible artifact manifests |
| 3 — UI structure and reliability | Summary-first pages, source/status consistency, isolated jobs, on-demand reads, keyboard support | Browser checks for save/reload/retry/failure, no stale overwrite, keyboard flow, four viewport sizes |
| 4 — Controlled research | Existing model/LLM ablations, diagnostics, continuation and uncertainty studies | Frozen experiments, declared search family, comparable net metrics, prospective records tied to the winning specification |

The product decisions and ready-to-use prompts are in [IMPROVEMENT_PROMPTS_2026-10-28.md](IMPROVEMENT_PROMPTS_2026-10-28.md). Stages 1–2 can begin without selecting a new model provider or visual framework.

## Audit artifacts and limitations

No application source was changed. The two new Markdown documents are the deliverables. No live browser rendering, image build, external integration run, performance-training campaign, private data audit, or measured strategy-performance claim is included.

One synthetic fixture remains at `C:\Users\santo\AppData\Local\Temp\tradingagents-web-audit-qdi91ztf\portfolio.sqlite3`. Automatic approval review rejected cleanup with **“blocked by policy”**; cleanup attempts stopped. It contains synthetic audit data, not the user's account records.
