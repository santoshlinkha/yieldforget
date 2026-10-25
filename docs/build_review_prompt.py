"""Build docs/REVIEW_PROMPT.md: a self-contained prompt asking another AI (ChatGPT, Gemini) to review the system.

Run from the TradingAgents folder:  PYTHONPATH=. .venv/Scripts/python.exe docs/build_review_prompt.py
It pastes the current SYSTEM_REFERENCE.md and the source of the functions a reviewer most
needs to see, so the prompt always matches the code it describes.
"""

import inspect
from pathlib import Path

HERE = Path(__file__).parent

INTRO = """# Review request: a personal quantitative portfolio and stock-research system

You are reviewing a working personal investment-research system. Act as **a senior
quantitative researcher and a senior ML engineer who has seen many backtests fail in live
trading**. Your job is to find what is wrong, weak, unproven or dangerous, and to say
precisely how to fix or test it. Praise is not useful; specific, checkable criticism is.

## Who it is for and what is at stake

One person's **Robinhood Roth IRA, about $5,844**: a cash account (buys need settled cash,
sales settle T+1, no margin, no shorting, options at most level 2). It is split into two
bookkeeping sleeves: **Investing** (about $4,844, long-term ETFs and stocks) and **Trading**
($1,000 initially, then its own P&L and transfers). The system plans, forecasts and
records; **it never places orders**; the owner trades by hand. Contributions are about $27.61
per trading day (the Roth IRA limit is $7,500 a year under age 50).

The owner wants to know: (1) whether any of the forecasting is good enough to influence real
money, (2) whether the evidence is trustworthy (leakage, multiple testing, survivorship,
statistical power), (3) what to fix first, and (4) which next experiment is most worth its
cost.

## How to answer

Use exactly these sections, in this order:

1. **Verdict (at most 6 lines).** Is any forecast fit to influence allocations? Which, at
   what horizon, under what conditions? If none, say so and say what evidence would change it.
2. **Top issues (a table of 10 to 15 rows)**, most severe first, with columns: Severity
   (critical / high / medium / low), Area, Issue (specific; cite the section or function),
   Why it matters (what could go wrong, in money or in false confidence), Fix (concrete),
   How to verify the fix (a test or a measurement).
3. **Leakage and timing audit.** For each item in the checklist below, answer *pass*,
   *fail* or *uncertain*, with one or two sentences of reasoning grounded in the reference
   or the code excerpts.
4. **Target definitions.** Are the trend, breakout, sideways, large-loss and drawdown
   targets well defined, non-degenerate and useful? Propose better definitions or thresholds
   if warranted, and say how to choose them without peeking at the final test.
5. **Statistical validity.** Comment on: folds and purging; development vs final-test design
   and its power (count the test dates); bootstrap block length; Holm correction scope and the
   total number of trials; ablation design; whether the portfolio test is realistic (costs,
   turnover, overlapping holds, T+1, next-close execution, a $1,000 sleeve).
6. **Models.** Are the model choices sensible for the data size and signal-to-noise? Is
   anything over- or under-engineered? Comment on the pooled design, ranking within date,
   missing-as-zero, the ensemble weighting, probability calibration (no isotonic/Platt step),
   the Chronos-2 stacking and contamination handling, GJR-GARCH vs EWMA. Use the inventory in section 14.
7. **Portfolio integration and risk.** Are the recommendation rules (net expected return > 0,
   evidence backed, P(large loss) <= 15%, at most 5 names, <= 25% of the sleeve each, settled
   cash only, no shorts, hold through the horizon) sound for a $1,000 Trading sleeve inside a
   Roth IRA? What would you change? How should stock picks interact with the ETF optimizer?
8. **Decision layer and AI (sections 13-14).** Is a verdict that comes only from the 20-session ranker
   (BUY = top fifth with gates; AVOID = bottom fifth) defensible given section 12's evidence, the score ties
   and the non-monotone score-to-P&L map? Are the signal warnings (lift plus top-fifth plus flat-model test)
   sound? Is the entry plan (dip quantiles, barrier exits) a reasonable use of the drawdown model? How should
   the LLM agents' ratings be evaluated against the model before either is trusted, and is the evidence
   packet (13.7) adequate for grounded LLM analysis? Should leveraged ETFs be allowed under the account rules?
9. **Data.** What free or cheap data would most improve point-in-time correctness or signal
   (point-in-time index membership, analyst revisions with timestamps, earnings calendars,
   options-implied volatility, short interest, insider transactions)? Be specific about sources
   and their point-in-time properties; say when you are not sure a source is point-in-time.
10. **Next experiments, ranked** by expected value of information per hour of work, each with
   a hypothesis, the exact test, and the decision it would change. Include whether a compact
   sequence model (PatchTST, VSN+LSTM) or a pretrained model (Chronos-2 fine-tuning, TimesFM)
   is worth testing next given the measured results, and why.
11. **Engineering.** Reproducibility, caching, failure isolation, append-only records, cost
    controls: anything fragile or missing?
12. **Questions for the owner** that would change your advice.

Rules for your answer: be concrete and cite sections (e.g. "6.2 Trend state") or function
names. Do not invent numbers; if you estimate, label it an estimate. If something cannot be
judged from what is given, say what you would need. Prefer the simplest fix that addresses
the risk. Assume the owner is technical and will implement your suggestions.

## Leakage and timing checklist (answer each in section 3)

1. SEC facts used from the day after the filing *date* (EDGAR gives no acceptance time in
   company facts). Is next-day enough? Any case where a 10-K/A or late XBRL revision leaks?
2. Restatements: the original value stands until the restatement's filing date.
3. FRED/ALFRED vintages: latest vintage published strictly before the date, from the next day.
4. Yahoo prices are **adjusted for later dividends and splits**. Returns are unaffected, but
   level-based features (distance from 52-week high, range position, moving-average gaps) and
   the raw-price reconstruction for valuation use adjusted series. Any leak or bias?
5. Universe: today's S&P 500, each stock from its "date added". Remaining bias?
6. Decision timing: features through the close of t; tradable return from the close of t+1.
   Path events measured from the close of t over t+1..t+h. Consistent?
7. Purging: training rows only where the tradable label ended before the test block; the
   inner validation split (last fifth of training dates) is also purged.
8. Cross-sectional ranking within each date: any leak from ranking against stocks that
   later entered the index, or from the requested stocks being added to the cutoff ranking?
9. Cost estimate (Abdi-Ranaldo) uses the next day's mid; the window stops a day before t.
10. The models used in the batch are fitted on all rows whose labels ended by the cutoff;
    the evidence attached to them comes from the walk-forward study. Is that evidence
    representative of the deployed model?
11. News: collected prospectively, not used in training; sentiment from FinBERT on headlines.
12. Analyst data: excluded from stock training because Yahoo's lists are today's versions.
13. Thresholds: defined in code before the study, versioned (now `thr-016cb9da59`), not tuned.
14. Verdict scores come from the daily decision fitted on rows whose labels ended by the cutoff; stocks
    outside the index are ranked against members without changing members' ranks. Any leak?
15. Fund verdicts use the ETF engine's combined forecast; its weights come from walk-forward ICs. Any
    look-ahead in choosing the weights?
16. The evidence packet cuts every series at the cutoff before computing; news needs a known publication
    time by the cutoff's end of day (New York); SEC facts count from the day after filing; FRED uses
    vintages published before the cutoff or the series' lag. Any remaining leak (e.g. a profile or analyst
    value that is today's, or first_seen after the cutoff)?
17. Agent runs analyze a past day with tools that withhold present-day profile values; is anything in
    their tool data from after the analysis day?

---

# The system (full reference)

"""

OUTRO = """
---

# Appendix: source of the critical functions

These are copied from the code at build time. Review them for leakage, off-by-one errors,
and mismatches with the reference above.

"""


def excerpts() -> str:
    from tradingagents.research import walkforward
    from tradingagents.ledger import capital
    from tradingagents.agents import packet
    from tradingagents.ledger import ai_gateway
    from tradingagents.llm_clients import roles
    from tradingagents.stocks import daily, fundamentals, ideas, macro, panel, positions, signals, targets, verdict

    parts = [("Targets and labels (tradingagents/stocks/targets.py)", [targets.Thresholds, targets.labels,
                                                                     targets.current_trend]),
             ("Point-in-time fundamentals (tradingagents/stocks/fundamentals.py)", [fundamentals._features,
                                                                                   fundamentals.history,
                                                                                   fundamentals.valuation]),
             ("Economic vintages (tradingagents/stocks/macro.py)", [macro.as_known]),
             ("Cost estimate and panel (tradingagents/stocks/panel.py)", [panel.spread_cost, panel.build]),
             ("Walk-forward split (tradingagents/research/walkforward.py)", [walkforward.walk_forward]),
             ("Recommendation sizing (tradingagents/stocks/ideas.py)", [ideas.make]),
             ("Position engine (tradingagents/stocks/positions.py)", [positions.Policy, positions.trades]),
             ("Daily decision gates and exits (tradingagents/stocks/daily.py)", [daily.evidence, daily.review_lot,
                                                                                daily.entries]),
             ("Capital settings (tradingagents/ledger/capital.py)", [capital.reallocation, capital.daily_plan]),
             ("The verdict (tradingagents/stocks/verdict.py)", [verdict.verdict, verdict.ai_decision]),
             ("Signals and the entry plan (tradingagents/stocks/signals.py)", [signals.peers, signals.describe,
                                                                             signals.entry_plan]),
             ("The evidence packet (tradingagents/agents/packet.py)", [packet.assemble, packet._prices, packet._news,
                                                                      packet._known_at, packet._macro, packet._fit]),
             ("Per-role agent models (tradingagents/llm_clients/roles.py)", [roles.resolve, roles.suggest]),
             ("AI gateway: budget, prices, cache (tradingagents/ledger/ai_gateway.py)", [ai_gateway.Gateway.run,
                                                                                         ai_gateway.UsageCallback._which])]
    out = []
    for title, fns in parts:
        out.append(f"## {title}\n")
        for fn in fns:
            out.append("```python\n" + inspect.getsource(fn).rstrip() + "\n```\n")
    return "\n".join(out)


def main():
    reference = (HERE / "SYSTEM_REFERENCE.md").read_text(encoding="utf-8")
    text = INTRO + reference + OUTRO + excerpts()
    (HERE / "REVIEW_PROMPT.md").write_text(text, encoding="utf-8")
    try:
        import tiktoken

        tokens = f"{len(tiktoken.get_encoding('o200k_base').encode(text)):,} tokens (o200k_base)"
    except ImportError:
        tokens = f"about {len(text) // 4:,} tokens (estimate)"
    print("wrote", HERE / "REVIEW_PROMPT.md", f"({len(text):,} characters, {tokens})")


if __name__ == "__main__":
    main()
