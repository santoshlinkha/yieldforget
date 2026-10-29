# Models, outputs, redundancy and web display review

Review date: 2026-10-28. Source: `3f72cbd`, with no tracked application changes. This follow-up extends [the repository audit](REPO_REVIEW_2026-10-28.md) and addresses the web interface as displayed to the user, including wording, units, hierarchy, tables, colors, and navigation.

## Main assessment

There are several useful research and decision pipelines, but they are presented as though their forecasts and performance records were interchangeable. They are not: targets, units, entry clocks, exit policies, and evidence differ. The most useful improvement is a shared model/output contract plus a clearer distinction between the active decision procedure, diagnostic models, research challengers, and historical compatibility features.

No model should be removed merely because it failed to win one inspected historical study. Conversely, unused compatibility code should not keep competing with the current decision workflow in the UI.

## What was checked

- Rechecked the unchanged source tree and previous audit. Traced model recipes through training, output storage, API responses, and current HTML renderers.
- Reviewed all major model families and the current eight dashboard tabs plus the ticker page. Inspected saved research JSON schemas and read only model-related tables from the local database using SQLite `mode=ro`. Account events, holdings, balances, credentials, and transaction contents were not part of those database queries.
- Reproduced probability semantics, Trade sizing differences, and CSS contrast with small synthetic calculations. No paid inference or large retraining was run.
- Created an isolated synthetic visual-QA fixture under `results/review-2026-10-28/`; it did not use the personal database. The fixture server was stopped after the rendering attempt.
- **Current browser screenshots remain unavailable.** Computer-use reported that its native pipe was unavailable. Automatic approval review then rejected launching an isolated headless Chrome renderer with **“blocked by policy.”** No alternative launch was attempted after rejection. Layout recommendations below are based on current markup/CSS, not claims of completed pixel-level browser verification.
- The earlier complete test run on unchanged source remains **1,662 passed, 4 failed, 4 skipped, 1 deselected**, with 100 subtests passed and 111 lint violations. It was not repeated unnecessarily. The four known failures concern Windows-specific assumptions.

## Saved-output snapshot

These counts describe the files/database inspected during this review, not investment performance. Another running application can subsequently add records. Aggregated details were saved to `results/review-2026-10-28/output_inventory.json` without account records or per-stock predictions.

| Record | Observed |
|---|---:|
| Forecast runs | 88 |
| Forecast rows | 8,620 |
| Stock research runs / forecast rows | 19 / 6,045 |
| Stock decisions | 33, all experimental |
| Model-comparison runs | 6 |
| Research trials | 36 |
| Recorded model predictions / matured outcomes | 8,006 / 0 |
| Shadow snapshots | 33 |
| Agent-run records | 1 |

Forecast checks found **zero** crossed `q05/median/q95` intervals, out-of-range probabilities, nonpositive scales, missing core forecasts, or nonfinite numeric values in the inspected forecast fields. This is numerical/schema integrity, not evidence of forecast accuracy or calibrated intervals.

There are **2,003 repeated date/ticker/horizon/model-key groups** in `model_predictions`. Repeated runs are legitimate records, but calibration needs an explicit economic-observation selection/weighting rule before using them. There are currently no matured outcomes, so the live recalibration defects identified in the earlier audit are not yet supported by any forward outcome sample in this table.

The latest model-comparison payload declares 14 model entries. Most have outputs for 500 scored stocks; per-stock Ridge has 466, analyst-upside has 492, and earnings-surprise has 496. **The ensemble has zero per-stock outputs and `ensemble` metadata is absent**, despite being declared in the model list; the payload has no model error notes. Treat this as an availability/coverage state requiring explicit display, not proof that the ensemble computation failed. The app already supports a base-ranker fallback, which should be visible consistently.

## Model and consumer inventory

| Family / recipes | Actual outputs and consumers | Lifecycle recommendation |
|---|---|---|
| Forecast engine: zero, training mean, volatility prior, momentum, Elastic Net, XGBoost regression/ranking/quantiles, Chronos-2, `+chronos` variants | Horizon log excess over cash, quantiles, P(positive total return), estimation spread; `forecast_runs` / `forecasts`; Forecasts, ticker, fund verdict/context | Keep baselines and retained challengers. Present as forecasting, separately from stock policy decisions. `research/engine.py:56`, `:84`; web `server.py:464`, `:690`, `:1345`. |
| Stock signal pipeline: Elastic Net, XGB quantiles, relative-return regression, trend classifier, drawdown quantiles, four event classifiers | Returns, trend/breakout/breakdown/loss/drawdown probabilities; stock research tables, Signals/screener, entry context, ideas, CLI | Active diagnostics. Share metadata and evaluation infrastructure while preserving distinct event definitions. `stocks/pipeline.py:70`, `:171`; `server.py:560`, `:632`. |
| Daily stock decision: XGB ranker, policy-P&L map, logistic trade quality with calibration, position-risk XGB | Candidate ranks, expected arithmetic net policy P&L, P(policy profit), position risk, sized actions and deterministic lot exits | Main decision procedure. Fix feedback targets, chronology, evidence identity and sizing consistency first. `stocks/daily.py:253`, `:281`, `:367`. |
| Model zoo: two XGB rankers, LightGBM rank/regression, pooled/per-stock Ridge, Mamba, Chronos price-only/covariates, four baselines, learned ensemble | Per-model percentiles and heterogeneous forecasts; Models, ticker, screener; ensemble can replace the daily rank/maps | Active comparison plus a possible decision input. Make coverage, target, active/fallback state and compatible OOF calibration explicit. `stocks/model_zoo.py:33`, `:257`; `daily.py:439`, `:529`. |
| Policy outcome layer: LightGBM plus linear/logistic baselines | P(target/stop/expiry/no fill), expected arithmetic policy excess over cash, downside quantile | Distinct competing policy family, not a duplicate terminal-loss predictor. Fix missing-downside eligibility and compatible evidence. `stocks/outcome_models.py:56`; `policy_study.py:335`; `decision_layer.py:127`. |
| Sector model: LightGBM LambdaRank and regression | Sector rank and expected log excess over SPY, using its next-close horizon; sector view | Keep reference-only until compatible evidence supports promotion. A selected stock within a sector is a rank-based pick, not a separately estimated winner probability. `research/sector_model.py:103`, `:117`; `server.py:317`. |
| Risk research: trailing/sample/EWMA/GARCH volatility; sample/Ledoit–Wolf/factor/EWMA covariance | Risk forecasts and comparative study metrics | Research-only in the found callsites. Do not imply these automatically configure current production risk. `research/risk_models.py:72`, `:157`; `experiments/risk_forecast_study.py`. |
| Sequence/pretrained challengers: TimesFM 2.5, Kronos-mini, PatchTST, LSTM | Experimental scores/returns in dedicated runners | Research-only, not dead merely because absent from the main page. Keep reproducible runners for retained hypotheses; archive concluded study designs/results deliberately. `experiments/challenger_stock_study.py:62`; `challenger_study.py:28`, `:51`. |
| LLM team: four analysts, bull/bear, research manager, trader, three risk roles, portfolio manager, reflector | Reports, proposals, ratings, entry plans, reflections | Thirteen roles, often sharing underlying models; not thirteen independently trained sources. Benchmark incremental role value and preserve role/provider provenance. `llm_clients/roles.py:20`, `:65`. |
| Sentiment, financial quality, moat, screening | FinBERT or word-list sentiment, deterministic checks, optional source screening | Label classification versus heuristics; surface fallback status. Keep company quality separate from a validated return signal. `stocks/news.py:111`; `research/quality.py:287`; `moat.py:209`; `agents/post_screen.py:68`. |

The forecast engine's `expected_returns()` has no production caller found; it is covered by tests. The current web is therefore not accurately described as a live optimizer automatically driven by that helper. Provider catalogs/capability adapters were reviewed structurally in the earlier audit; every remote model ID and account entitlement was **not** checked by live API calls.

## New output and display findings

### D1 — The ticker's probability label names the wrong event

`research/forecast.py:110` calculates P(total return > 0), including cash return. `ticker.html:589–591` calls the value the chance of beating cash. Those events differ.

A synthetic residual distribution `[-2,-1,0,1,2]`, zero predicted excess, positive cash return and scale 0.1 produces `p_gain=0.6`; the probability of positive excess using the same distribution is 0.4. The example establishes the semantic difference, not a realistic forecast.

**Change:** expose explicitly named `p_total_positive` and, if required, a separately computed `p_excess_positive`. Render the actual event, reference asset and horizon. Use this contract across Forecasts and ticker.

### D2 — A single renderer mixes different targets and units

The model tables describe forecasts as 20-session log returns over cash (`ticker.html:741`, `index.html:865`). Ordinary supervised zoo return recipes use that target, but the ensemble maps percentiles to **arithmetic barrier-policy net P&L**, with a separate excess-versus-SPY value (`stocks/model_zoo.py:305`). Directly formatting both as “vs cash” is misleading.

**Change:** attach target/unit/benchmark/entry/exit metadata per output. Show a primary rank comparison separately from return comparisons. Only compare returns in one column after deliberate conversion to a common evaluated policy outcome.

### D3 — Chronos stock forecasts have a different entry clock

`research/foundation.py:68–75` forecasts the final future step relative to the current last close. The zoo requests a 20-step forecast, whereas stock labels use entry at `t+1` and exit at `t+21` (`stocks/targets.py:3`). The common table description claims next-close timing.

**Change:** align the model forecast to the evaluated entry/exit clock or label it as current-close-to-horizon research. Do not assume an extra step alone estimates a conditional next-close return correctly; evaluate the chosen transformation.

### D4 — Trade's prominent limit instruction conflicts with its canonical policy

`index.html:691` foregrounds a buy limit at the last close. Backend `trade.py:44–45` supplies a next-close-at-whatever-price entry and an execution specification with no limit price. The compatibility field `buy_limit` still exists; this is a semantic mismatch, **not** a missing-field crash. Target/stop dollars are illustrative levels from the reference, while actual levels depend on the fill.

**Change:** display the canonical entry text, dated reference close, fill-relative percentages and close-confirmed exit timing. Keep patient limits under a visibly separate untested-policy panel. Recording a completed purchase should ask for the actual fill price.

### D5 — Trade candidate sizing can diverge from the saved decision

`trade._candidate()` uses the entire supplied budget to set `dollars` and `shares`; it does not preserve the saved entry's size. `daily.entries()` applies slot and concentration limits first (`daily.py:413–429`).

**Synthetic reproduction:** a saved candidate sized at $25 / 0.25 shares, with reference price $100 and a $1,000 supplied budget, becomes $1,000 / 10 shares in the Trade-view payload. The current main card does not display that share size, reducing immediate UI exposure; the API contract nevertheless conflicts with the claim that it reflects the saved decision.

**Change:** distinguish illustrative budget capacity from permitted order size. Use one account-aware sizing service, retain the decision's gates/reasons, and explicitly revalidate when cash or holdings change.

### D6 — Track record measures another outcome and hides costs in its main average

`stocks/track_record.py:158`, `:203–212` follows fixed-horizon closes for picks and policy suggestions rather than replaying the barrier exit policy. A limit record also has its own simplified path. `summary()` averages raw `return`, while “correct” subtracts the estimated round-trip cost (`:207`, `:212`, `:225`). The headline average and the correctness label therefore use different cost definitions. This is separate from the proper shared-engine shadow record.

**Change:** provide named records: fixed-horizon diagnostic, actual policy replay, forward capital account, and actual recorded fills. Resolve policy outcomes through the shared execution engine. Lead with net return and net excess versus the same-date benchmark; state the costs and denominator. Keep diagnostic hit rates as secondary context.

### D7 — Displayed coverage and lifecycle are incomplete

The latest saved zoo has partial model coverage and no ensemble scores, yet declares the full catalog. `renderZoo()` defaults to an equal-weight average of available fitted-model percentiles (`index.html:855`) rather than the learned ensemble. Users can read the generic “Average” as the active decision score; missing models can also change the effective average per stock.

**Change:** show model coverage and availability; label “Equal-weight mean percentile (N models)” separately from the active ensemble. Default the decision-oriented view to the actual active recipe. Include compatible development dates, forward sample size and fallback reason in each model row.

### D8 — Small text has insufficient contrast in specific theme pairs

Current CSS pairs white button text with dark-theme blue `#60a5fa`, a calculated contrast ratio of **2.542:1**. Light-mode positive text `#059669` on white is **3.768:1**. Ordinary small text requires 4.5:1 under [WCAG's contrast criterion](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).

**Change:** separate link/accent color from button background color, or use dark text on the bright blue. Darken light-theme green for small numbers. Add a text/icon sign as well as color, and verify the actual computed states in-browser.

### D9 — Forecast loading provenance overstates revision pinning

The challenger metadata advertises a pinned Chronos revision, but the shared loader calls `from_pretrained()` without a `revision=` argument (`research/foundation.py:34–39`). `Chronos2Score` wraps that shared forecaster; it is not a separate architecture.

**Change:** pin the shared loading contract or record the actual resolved revision. Store model artifact, package version, preprocessing and clock metadata in every run.

### D10 — One duplicate transport can expose a credential in error logs

`stocks/macro.py:58–79` sends `FRED_API_KEY` as a query parameter and logs raw request exceptions. Such exceptions can include the request URL. The agent-facing FRED path already uses `dataflows.net.get_scrubbed`.

**Change:** share scrubbed transport, timeout/retry behavior and redaction tests while preserving the stock client's vintage semantics. No real credential or log contents were inspected for this finding.

The earlier P1 findings remain: ledger atomicity and event identity; policy-mismatched feedback; calibration cutoff; incomplete decision evidence identity; write-origin controls; Docker private-data exclusions. This follow-up adds display/output issues rather than superseding those priorities.

## What to consolidate, deprecate or retain

| Candidate | Repository evidence | Recommendation |
|---|---|---|
| `optimizer/auto_invest.py` | Only test references found (`test_optimizer.py:443`, `:457`) | Strong archive/deprecation candidate; check public API commitments first. |
| `optimizer/tracker.py` | Only test imports found; current web does not call it | Archive retired tracker code deliberately; preserve historical tracker files. |
| Old web scaffolding | `tracker_path`, `backtest`/`tracker` jobs, unpopulated `last_targets`/`last_rows` in `server.py:178–194` | Remove obsolete fields and old-plan assumptions after checking API consumers. |
| Risk JSON persistence | `optimizer/risk.py:66–114` has no production persistence callers; risk definitions remain used | Split active risk definitions from retired JSON settings persistence. |
| Old optimizer publishing | `recommendations.save/official/decide` have no production callers found; history/load remain live | Keep archive reading and stored data. Deprecate inactive publishing separately; extract shared provenance helpers. |
| `/api/ratings` | Still implemented, but no current static UI caller; UI uses `/api/ai-decisions` | Mark compatibility status and deprecate through a versioned API policy. |
| Old stock ideas/sizing | Still reachable from CLI/API; overlaps daily decision responsibilities | Migrate callers toward one decision service before deprecation. It is not dead today. |
| Zoo training recipes | LightGBM/Ridge study implementations duplicate live recipe implementations | Share adapters/recipes between study and inference; avoid silent implementation drift. |
| Model artifact writers | Forecast engine, stock pipeline, daily models and policy bundles implement similar storage logic | Share manifests, hashing, atomic writes, version checks and retention; preserve distinct model semantics. |
| Mutable market caches | Finalized bars, analyst OHLCV and challenger OHLCV have separate paths/freshness behavior | Unify provenance/freshness behind interfaces. Keep immutable ledger price snapshots separate. |
| Multiple data engines | SQLite model/ledger/news stores and DuckDB training sets serve different workloads | Standardize connection, backup and manifest policies; do not combine stores merely because there are several. |
| Uncalled helpers | `research.engine.expected_returns`, `stocks.models.event_model`, `decision_layer.portfolio` lack production callers found | Classify as test/library/compatibility surface before deleting. Static searches cannot rule out external users. |
| Two backtests | CLI graph-rating evaluator and optimizer portfolio simulator answer different questions | Keep both mechanics, name their outcomes clearly. The optimizer simulator still supports research experiments. |
| Report writers | `reporting.py` / `report_html.py` are live via CLI and web agent analysis | Keep the shared writer; consolidate new output formats around it. |
| Old architecture documents | `HOW_IT_WORKS.md:225–228` and parts of `SYSTEM_REFERENCE.md` describe retired tracker/optimizer UI behavior | Update authoritative Markdown, label retired workflows, then regenerate PDFs and derived prompts. |

No source, history, model artifact or user record was deleted during this review.

## Proposed web display

These are concrete presentation proposals from current structure. Their visual fit must be checked in a rendered session.

| Screen | Proposed display |
|---|---|
| Trade | One “Today's decision” summary: permitted action, top candidate, reason to wait, evaluated entry clock, allowed size, model/evidence status and cutoff. Put account readiness beside it. Move detailed market/sector research below a collapsed section or into Research. Use “Top model candidate” in experimental mode. |
| Portfolio | State whether KPIs cover stocks only or the full account. Current “Portfolio value” excludes SGOV/cash (`server.py:285–291`; `index.html:802`). Use “Stock holdings value” or a full total with breakdown. Put holdings before a collapsed “Record trade” form; label “Paid” as open cost basis. |
| Screener | Compact search/filter/view toolbar and one refresh control. Default to ticker/name, rank, quality, analyst view, valuation and data status. Move component calculations and methodology into columns/row details. Keep ticker/name sticky. |
| Forecasts | Separate Results, Configure run and Validation. Default to ticker, exact expected-outcome label, an interval with a zero reference, event probability and cutoff. Expand constituent models per stock. Put hardware/runtime under Run details. |
| Decisions | Use for history and analysis rather than duplicating today's Trade page. Clearly separate Quant decision from Agents' opinion. Move provider budgets into Settings; retain a run cost estimate near Run. |
| Models | Start with a model-health/lifecycle list: active decision source, challenger/context state, coverage, exact target/clock, calibration, evidence dates, artifact and runtime. Make the wide stock-by-model matrix a secondary comparison with selectable columns. |
| Track record | Lead with the exact net outcome, benchmark, matured sample and uncertainty. Separate policy, fixed-horizon diagnostic, agent and actual-account records. Use Open/Matured/All filters. Name the row action Hide, since the saved record remains. |
| Settings | Group Account, Data & universe, AI providers & cost. Show connected providers first. Collapse role overrides/pricing into Advanced; show section-level save state confirmed from canonical persisted values. |
| Ticker | Full-width ticker/price/position/action/evidence summary first, especially on mobile. Follow with Price, Research, Models and Reports sections. Keep four main stats visible; expand the rest. Add chart range controls and a visible log-scale indicator. Merge repeated entry panels with quant/agent provenance. |

Global presentation changes: strengthen heading/body hierarchy instead of shrinking long explanatory copy; use contained scrolling and sticky identity columns in wide tables; expose mobile navigation overflow; show stale data with timestamps while refreshing; make empty states explain the next action. Standardize trading-session horizons and rank notation. “Worst drop ever” should become “Maximum drawdown since [history start].” Preserve output-specific probability definitions rather than relabeling every probability “Up.”

## Implementation order and next-session checks

1. **Correct the output contract:** D1–D7, earlier feedback/validation defects, active/fallback identity. Add focused API-to-renderer contract tests and deterministic outcome examples.
2. **Protect records and infrastructure:** earlier ledger, cache, API, budget and packaging defects; D9–D10; shared manifests/transport.
3. **Simplify the displayed workflow:** summary-first Trade/Ticker, distinct model lifecycle, uncertainty presentation, clear cash/account labels, named performance records.
4. **Deprecate deliberately:** start with test-only optimizer tracker/auto-invest, dead web fields and compatibility publishing. Preserve historical data and external API migration paths.
5. **Complete rendered QA in another session:** desktop/tablet/mobile, light/dark, populated/empty/error/stale states, every tab and ticker; keyboard navigation and report dialogs; save/reload, active-model fallback and long-job transitions.

Use [REVIEW_SESSION_HANDOFF.md](REVIEW_SESSION_HANDOFF.md) to continue without repeating completed work. Product/model-budget questions and implementation prompts are in [IMPROVEMENT_PROMPTS_2026-10-28.md](IMPROVEMENT_PROMPTS_2026-10-28.md).
