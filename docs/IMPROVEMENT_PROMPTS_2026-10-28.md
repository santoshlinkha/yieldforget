# Improvement prompts — 2026-10-28

Use these with [the repository review](REPO_REVIEW_2026-10-28.md). They separate missing product decisions from implementation. The defaults below are proposals, not inferred user authorization to change trading behavior.

Also read [the model/output/display follow-up](MODEL_OUTPUT_DISPLAY_REVIEW_2026-10-28.md) and [session handoff](REVIEW_SESSION_HANDOFF.md), which record the newer unit/timing/display findings, archive candidates, output checks and pending rendered QA.

## 1. Information needed from the owner

Copy this prompt and answer the fields you know. Unknowns can remain unknown; correctness fixes do not require every answer.

```text
Help define the next version of this TradingAgents/Yieldforge application.
Read docs/REPO_REVIEW_2026-10-28.md and the current code first. Use connected or
repository context before asking me for facts already available. Separate
confirmed requirements from proposed defaults. Do not ask for API keys,
passwords, full account identifiers, or private holdings to answer these questions.

My answers:
1. Main purpose: [personal research / paper decision support / manual brokerage
   decision support / multi-user product]. Primary user skill level: [ ].
2. Hosting: [local Windows only / local Docker / private server / public service].
   Users and devices: [ ]. Mobile use important? [ ].
3. Asset universe and account constraints: [ ]. Existing restrictions documented
   in the repo that remain valid: [ ]. Intended holding horizons: [ ].
4. Primary optimization objective: [net excess versus cash-flow-matched SPY /
   absolute net return / drawdown-aware growth / another defined objective].
   Maximum acceptable loss/drawdown and how it is measured: [ ].
5. When the risk estimate is unavailable or uninformative:
   [block new entries / permit paper candidates with a visible warning /
   another explicitly described behavior]. What should validated mode do? [ ].
6. Required behavior of cash edits: [record a dated external flow / correct a
   reconciliation balance / both as separate operations]. Preferred audit trail: [ ].
7. Data/model spend: monthly data budget [ ]; monthly AI budget [ ]; maximum
   latency for a normal decision [ ]; maximum scheduled retraining time [ ].
   Hardware constraints and whether cloud compute is acceptable: [ ].
8. Which outputs matter most: [candidate ranking / actionable entry/exit plans /
   portfolio allocation / track record / research explanations]. Rank the top 3: [ ].
9. UI preference: [keep current appearance with better organization / compact
   trading workspace / simplified investing dashboard]. Must-keep features: [ ].
10. Promotion: what independent forward evidence, interval, calibration, cost,
    and operational checks should a candidate satisfy? [ ]. Unknown is acceptable;
    propose a defensible evaluation protocol without asserting profitability.

Return a one-page product brief, a table of unresolved decisions with recommended
defaults and consequences, and an implementation backlog tied to the audit IDs.
Ask only the questions that materially change the immediate implementation.
```

Recommended starting assumptions for planning: preserve the current local/manual workflow; preserve account records; keep model research experimental until compatible evidence exists; make risk policy explicit; focus the UI on the current decision and its limitations. Confirm changed financial policy before adopting it.

## 2. Code and accounting implementation prompt

```text
Implement Stage 1 of docs/REPO_REVIEW_2026-10-28.md in this repository.
First inspect current git status and applicable repository instructions, then
verify each finding against current code. Preserve existing unrelated changes.
Use isolated temporary fixtures; do not modify personal databases, caches,
portfolio JSON files, credentials, or brokerage state. Do not call paid models.

Work in small reviewable groups:
A. Make ledger validation plus append atomic across connections. Distinguish new
   manual operation identity from retried delivery and broker-import identity.
B. Protect local write APIs with Host/origin/session-token validation, strict
   JSON parsing and bounded bodies. Keep documented local workflows usable.
C. Fix false-success cash edits using an explicit audited operation and canonical
   response; do not invent external contributions from ambiguous corrections.
D. Claim AI request capacity before dispatch, fence job state by run ID, update
   cache freshness only after successful persistence, and use actual session close
   timestamps for order cutoffs.
E. Exclude secrets/account data from Docker contexts; repair platform-specific
   tests and lint; cover output-cap forwarding across provider constructors.

Add meaningful regressions: competing $80 buys with $100; independent identical
manual fills versus retry of one operation; foreign-origin text/plain writes;
cash save/reload; request budget exactly one; old worker updating a new run;
permanent cache rename failure; early-close publication; provider output caps.
Cover withdrawals/sales/reservations where the same invariant applies.

Do not silently choose a new cash-reconciliation or risk policy. Complete all
independent fixes and ask for only the material unresolved product decision.
Run relevant tests, then the full isolated suite and lint once changes stabilize.
Report changes, evidence, remaining failures, and migration/compatibility effects.
```

## 3. Model correctness and research prompt

```text
Implement Stage 2 of docs/REPO_REVIEW_2026-10-28.md, then propose controlled
research using the existing model families. Verify findings against current code.
Do not run paid inference or large studies without an agreed resource budget.

Create distinct identities for the training recipe, complete decision procedure,
and individual fitted artifact/run. The procedure includes ranking and ensemble
construction, calibration/retraining rules, gates, ordering, sizing, risk fallback,
feature/data availability, costs and execution/exit policy. Record each fitted
model, weight/map and calibration artifact hash on individual decisions. Routine
scheduled refits under the same evaluated procedure retain the forward cohort;
material procedure changes start a new one. Reuse trained models when only dollars
change, but invalidate incompatible decision evidence. Train/calibrate trade
quality using out-of-fold scores from the ranking procedure actually deployed,
including the ensemble when active.

Resolve frozen predictions through the same policy simulator that creates labels.
Store fixed-horizon diagnostics separately. Calibration must use only compatible
outcomes available by its knowledge cutoff, with a documented deduplication and
sampling rule. Preserve raw predictions and record calibration versions.

Bind studies, ETF calibration, out-of-fold caches and forward shadow cohorts to
compatible artifact manifests. Require finite downside estimates for eligible
buys. Preserve date-based splitting and label-interval purging. Make statistical
inference dependence-aware and declare the experiment/search family.

Add regressions demonstrating that:
- the +9.6% barrier-policy example does not become a -10.4% calibration label;
- future outcomes cannot change past decisions;
- another model's outcomes and duplicate reruns cannot contaminate calibration;
- changing ranking methods, gate rules, calibration procedures, or ensemble
  construction invalidates incompatible validation, while scheduled refits under
  the same evaluated procedure retain their cohort and record new artifact hashes;
- missing/NaN/infinite downside estimates produce explicit abstention;
- altered folds/features/data reject stale OOF artifacts.

Then design a modest champion/challenger study comparing cash/SPY, momentum,
regularized linear, XGBoost, LightGBM and the current ensemble under identical
universe, dates, execution, capital and costs. Include prediction ties/spread,
missingness, calibration, net returns, drawdown, turnover, intervals and latency.
Distinguish exploratory results from untouched prospective evidence. Do not
promote a model merely for a better retrospective rank correlation or more trees.

Deliver the code changes, test evidence, artifact specification and a reproducible
study plan. Identify product choices about risk/objectives before changing them.
```

## 4. Web UI implementation prompt

```text
Improve the existing Yieldforge web UI using docs/REPO_REVIEW_2026-10-28.md.
Start by reading both HTML pages, server routes and tests, then render the current
UI with isolated synthetic fixtures. Keep user data and credentials untouched.
Use the existing architecture unless a demonstrated requirement justifies a move.

Make Trade and Ticker summary-first on desktop and mobile:
- action and reasons to wait;
- active decision source/version, fallback, cutoff, freshness and validation state;
- account equity, settled/unreserved cash and reservations with distinct labels;
- entry/exit timing, expected outcome, downside, costs and benchmark definition;
- research and full diagnostics available through progressive disclosure.

Make Models a consistent champion/challenger diagnostic view. Remove contradictory
claims about whether the ensemble drives Trade. Show sample sizes and evaluation
periods beside performance/calibration. Keep score ranks distinct from probabilities.

Extract shared API, formatting, table, chart and dialog utilities. Lazy-load tabs,
deduplicate reads, use AbortController/timeouts/backoff, and poll per run ID.
Provide clear saving/saved/error states based on canonical saved responses.
Complete keyboard tabs, sort buttons/aria-sort, form labels, live status and modal
focus containment/return. Constrain wide tables to their cards.

Verify at 320/390/768/1440px, with keyboard and simulated network failures.
Exercise cash save/reload, repeated manual action versus retry, stale job results,
model-source consistency and missing risk estimates. Capture updated screenshots
and refresh affected documentation after the final UI is verified.
Deliver implementation, focused browser/test evidence and unresolved design choices.
```

## 5. LLM role/model selection prompt

```text
Design a reproducible evaluation of the current TradingAgents LLM role assignments.
Inspect provider adapters, role configuration, prompts, structured fallback and
request/usage logging. Read the repository review's model recommendations.

Use frozen, versioned evidence packets and explicit tasks. Compare a small set of
available models/effort levels within the agreed budget. Verify provider/model
capabilities against official documentation when choosing concrete current IDs.
Do not assume a larger model, more debate rounds, or more providers improves results.

Evaluate analyst grounding/citations, numerical consistency, schema/entry-plan
validity, missing-evidence abstention, resistance to instruction-like text in
retrieved evidence, stability over repeated runs, latency and measured tokens/cost.
Ablate the bull/bear and risk debate roles to measure incremental value. Keep
research explanation quality separate from realized trading-policy performance.

Return a benchmark protocol, artifact schema, scoring rubric, budget estimate,
and candidate role assignment. Run it only after a resource budget is established.
Do not use private account information as benchmark material.
```

## 6. Resume the remaining review in another session

```text
Resume the repository review from docs/REVIEW_SESSION_HANDOFF.md. Read the
model/output/display follow-up before starting, and check current git status.
Preserve completed findings and artifacts; rerun checks only if source or evidence
has changed. This remains an audit task, not authorization to delete old modules
or change financial behavior.

Complete the actual displayed-web review using a supported browser-control tool
and isolated synthetic data. Inspect all eight dashboard tabs and Ticker in light
and dark themes at desktop, tablet and phone widths. Verify the probability,
return-unit, entry/exit timing, sizing, model availability and performance labels
against their backend definitions. Check keyboard interactions and populated,
empty, stale, loading and error states. Do not work around earlier approval
rejections or claim screenshots exist when they do not.

Update the audit with observed screenshots, a short visual issue list and an
ordered backlog. Classify every cleanup candidate as active, research-only,
compatibility, archive-only or uncalled; preserve historical records and confirm
external compatibility requirements before recommending removal.
```
