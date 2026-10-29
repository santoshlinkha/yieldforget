"""One plain verdict per stock (BUY, WATCH, HOLD, SELL or AVOID) from the app's own 20-session model.

The verdict comes only from the internal quant model of the daily decision (``stocks.daily``), fitted on
this machine: it ranks every S&P 500 stock on how a 20-session trade should do after trading costs, and
for each it gives the expected net return, the chance the trade ends in profit, and the chance of a large
loss while held. No AI rating decides anything; the agents' rating, the quality and Wall Street screen and
the breakout / breakdown signals are shown beside the verdict as context.

Rules (``score`` = the stock's percentile among today's ranked stocks, 1 = the best):

- **Not held**
  - **BUY**: the daily decision says ENTER, or score in the top fifth (0.80+), expected net return after
    costs above zero, P(profit) at least 50% and P(large loss) at most 15%.
  - **AVOID**: score in the bottom fifth (0.20 or lower), or an expected net loss after costs.
  - **WATCH**: everything in between; the failed tests say what is missing.
- **Held**
  - **SELL**: the daily decision says EXIT or REDUCE for the lot, or score in the bottom fifth with an
    expected net loss after costs.
  - **HOLD**: everything else.

The model is experimental: its ranking showed a small edge in the walk-forward study, but trading on it did
not beat holding SPY, so the verdict is the model's view, not a proven forecast. The app recommends; you
decide.
"""

from __future__ import annotations

from datetime import date

TOP = 0.80                  # score percentile a BUY needs
BOTTOM = 0.20               # at or below: AVOID (not held) / SELL with an expected loss (held)
MIN_P_WIN = 0.50
MAX_P_RISK = 0.15
RATING_DAYS = 30            # an agent rating older than this is not shown as current
MIN_QUALITY_BUY, MAX_CONSENSUS_BUY, MIN_UPSIDE_BUY = 75, 2.2, 0.10   # context marks only
BAD_SIGNALS = ("breakdown", "crash", "drawdown")


def _fresh(rating: dict | None, today: date) -> dict | None:
    if not rating or not rating.get("rating") or not rating.get("date"):
        return None
    try:
        age = (today - date.fromisoformat(str(rating["date"])[:10])).days
    except ValueError:
        return None
    return {**rating, "age_days": age} if 0 <= age <= RATING_DAYS else None


def _check(name, ok, detail, decides=True):
    return {"name": name, "ok": ok, "detail": detail, "decides": decides}


def verdict(held: bool, decision: dict | None, signal: dict | None, screen: dict | None, rating: dict | None,
            today: date | None = None, score: dict | None = None, horizon: int = 20, validated: bool = False) -> dict:
    """``{"verdict", "action", "label", "validated", "summary", "checks", "context", "held", "model"}``.

    ``verdict`` is the model's view; ``action`` is what the evidence gate allows. While the model is not
    validated (experimental mode; funds always), a model BUY is action WAIT and a model-driven SELL is HOLD,
    and ``label`` says so ("BUY (model view, not validated)"); a rule-based exit from the daily decision's
    lot review (EXIT / REDUCE) and the decision's own ENTER keep their action. ``checks`` decide the verdict;
    ``context`` is shown beside it and changes nothing. ``ok`` None means no data."""
    today = today or date.today()
    screen, rating = screen or {}, _fresh(rating, today)
    action = (decision or {}).get("action")
    s = score or {}
    # The position among today's stocks (ties broken), not the tied score, decides the fifths.
    pct = s.get("rank_pct", s.get("score_pct"))
    net, p_win, p_risk = s.get("expected_net"), s.get("p_win"), s.get("p_risk")

    checks = []
    if s:
        fund = s.get("kind") == "fund"
        peers = "funds" if fund else "S&P 500 stocks" + ("" if s.get("member", True) else ", where it would stand")
        checks = [
            _check("Model rank", pct >= TOP, f"#{s['rank']} of {s['of']} {peers} (ahead of {pct:.0%}; "
                                             f"BUY needs the top fifth, AVOID is the bottom fifth)"),
            _check("Expected return", net > 0,
                   f"{net:+.2%} log return above cash over {horizon} sessions from the latest close, before "
                   f"costs (BUY needs above zero)" if fund else
                   f"{net:+.2%} over {horizon} sessions after costs of {s.get('cost_rt') or 0:.2%} (BUY needs above zero)"),
            # A fund's probability is of a positive total return (cash yield included), not of beating cash.
            _check("Chance of a positive total return (before costs)" if fund else "Chance of profit after costs",
                   p_win >= MIN_P_WIN, f"{p_win:.0%} (BUY needs {MIN_P_WIN:.0%}+)"),
        ]
        if not fund:
            # A stock needs its risk estimate: missing or unknown fails the gate, it is never skipped.
            # One the model gives every stock alike (its training base rate) is passed, as the daily decision's
            # default (risk_gate_when_uninformative = skip) passes it, and said to be no assessment of this stock.
            known = p_risk is not None and p_risk == p_risk
            if known and s.get("p_risk_informative") is False:
                detail = (f"{p_risk:.0%}, the same for every stock (the risk model's training base rate): passed, but "
                          f"it says nothing about whether this one is riskier")
            elif known:
                detail = (f"{p_risk:.0%} at the exit of a {horizon}-session trade (terminal, not the worst point "
                          f"while held; BUY needs {MAX_P_RISK:.0%} or less)")
            else:
                detail = "unknown: BUY needs a risk estimate"
            checks.append(_check("Chance of a large loss at the exit",
                                 known and p_risk <= MAX_P_RISK, detail))
    if action:
        checks.append(_check("Daily decision", action not in ("EXIT", "REDUCE"),
                             f"{action}{' (' + decision['paper_action'] + ')' if decision.get('paper_action') else ''}"))

    warnings = {w["kind"]: w["text"] for w in (signal or {}).get("warnings", [])} \
        if signal and signal.get("status") == "ok" else {}
    bad = [v for k, v in warnings.items() if k in BAD_SIGNALS]
    q, c, u = screen.get("quality_score"), screen.get("analyst_mean"), screen.get("target_upside")
    context = [
        _check("Breakdown or crash warning", None if not signal or signal.get("status") != "ok" else not bad,
               "no signals saved" if not signal or signal.get("status") != "ok" else ("; ".join(bad) or "none"), False),
        _check("Quality score", None if q is None else q >= MIN_QUALITY_BUY,
               "not screened" if q is None else f"{q:.0f} of 100", False),
        _check("Wall Street", None if c is None else c <= MAX_CONSENSUS_BUY,
               "no analysts" if c is None else f"consensus {c:.2f} (1 strong buy to 5 strong sell)"
               + (f", target {u:+.0%}" if u is not None else ""), False),
        _check("Agents' rating", None if not rating else None,
               "no recent agent analysis" if not rating else f"{rating['rating']} on {rating['date']} "
                                                             f"({rating['age_days']} days ago); not used", False),
    ]

    if held:
        if action in ("EXIT", "REDUCE"):
            v, why = "SELL", f"the daily decision says {action}"
        elif s and pct <= BOTTOM and net < 0:
            v, why = "SELL", f"the model ranks it #{s['rank']} of {s['of']} and expects {net:+.2%} after costs"
        else:
            v = "HOLD"
            if not s:
                why = "held; no model view (the model has not scored this ticker), so nothing says to sell"
            else:
                peers = "funds" if s.get("kind") == "fund" else "stocks"
                where = f"ranks #{s['rank']} of {s['of']} {peers}"
                why = (f"held; the model {where}, in the top fifth: a candidate to add to" if pct >= TOP else
                       f"held; the model {where}, in the bottom fifth, but still expects {net:+.2%}: "
                       f"weakest of what you hold, though no sell signal" if pct <= BOTTOM else
                       f"held; the model {where}, in the middle: no reason to sell or add")
    elif action == "ENTER":
        v, why = "BUY", "the daily decision says ENTER"
    elif not s:
        v, why = "WATCH", "no model view: the model has not scored this ticker"
    elif pct <= BOTTOM or net < 0:
        v = "AVOID"
        why = (f"the model ranks it in the bottom fifth (#{s['rank']} of {s['of']})" if pct <= BOTTOM
               else f"the model expects {net:+.2%} after costs over {horizon} sessions")
    elif all(x["ok"] for x in checks):
        v, why = "BUY", f"the model ranks it #{s['rank']} of {s['of']} and every test passes"
    else:
        missing = [x["name"].lower() for x in checks if x["ok"] is False]
        v, why = "WATCH", "the model's view is not strong enough: " + ", ".join(missing)
    validated = bool(validated) and not (s and s.get("kind") == "fund")
    rule_based = action in ("EXIT", "REDUCE", "ENTER")
    act = v if validated or rule_based else {"BUY": "WAIT", "SELL": "HOLD"}.get(v, v)
    label = v if act == v else f"{v} (model view, not validated)"
    return {"verdict": v, "action": act, "label": label, "validated": validated, "summary": why,
            "checks": checks, "context": context, "held": held,
            "model": {"horizon": horizon, **s} if s else None,
            "rules": {"top": TOP, "bottom": BOTTOM, "min_p_win": MIN_P_WIN, "max_p_risk": MAX_P_RISK,
                      "rating_days": RATING_DAYS}}


AI_DECISIONS = {"Buy": "BUY", "Overweight": "BUY", "Hold": "HOLD", "Underweight": "SELL", "Sell": "SELL"}


ABSTAIN_MARK = "**Insufficient evidence**"   # what decision_markdown writes when an answer abstained


def rating_on_record(entry: dict) -> dict:
    """A memory-log entry as the rating the screens use: the rating, its date and whether the analysis
    abstained (its decision text carries the abstention mark; the rating was recorded only because the format
    requires one)."""
    return {"rating": entry.get("rating"), "date": entry.get("date"),
            "abstained": ABSTAIN_MARK in str(entry.get("decision") or "")}


def ai_decision(rating: dict, held: bool) -> dict:
    """The agents' rating as an action: Buy / Overweight -> BUY, Hold -> HOLD when held (WATCH when not),
    Underweight / Sell -> SELL when held (AVOID when not), with the rating's age; ABSTAIN when the analysis said
    the evidence was insufficient. Shown as the agents' view; it decides nothing."""
    r = rating.get("rating")
    action = "ABSTAIN" if rating.get("abstained") else AI_DECISIONS.get(r, "REVIEW")
    if action == "SELL" and not held:
        action = "AVOID"
    if action == "HOLD" and not held:
        action = "WATCH"
    try:
        age = (date.today() - date.fromisoformat(str(rating.get("date"))[:10])).days
    except ValueError:
        age = None
    return {"action": action, "rating": r, "date": rating.get("date"), "age_days": age,
            "stale": age is None or age > RATING_DAYS, "abstained": bool(rating.get("abstained"))}
