"""Run the analysis by hand in a chat app (ChatGPT, Gemini, Claude): one prompt out, one answer back.

``build_prompt`` sends the TradingAgents team's own prompts (``agents.upstream_prompt``): every agent from
the market analyst to the portfolio manager, in the graph's order, with the analysts' tool results pasted
under their prompts. It carries no holdings and none of the app's own verdicts, ranks or signals, so the
chat's judgment is independent of the app's. The chat answers each agent as the API would (the structured steps
in their schemas), and ``final_state`` turns the answer into the same final state an API run saves, so the same
report files are written. ``parse_answer`` still reads an older answer's single JSON block (rating, summary,
thesis, target, horizon, trader's proposal).
No API is called: the model runs in the user's own chat subscription.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.agents.upstream_prompt import PROMPT_FORMAT, build, trader_line

RATINGS = [r.value for r in PortfolioRating]
PROMPT_VERSION = PROMPT_FORMAT


def build_prompt(ticker: str, day: str, cache_dir: Path | None = None) -> str:
    """The whole prompt for one instrument and cutoff: each agent's own prompt and its tool data, in order.
    Data comes through the configured vendors, which read the local store first."""
    return build(ticker.upper(), day)


_RESPONSE = re.compile(r"^#{1,4}\s*Response\s+(\d+)\s*[:.\-–]?[^\n]*$", re.M | re.I)


def parse_steps(text: str) -> dict[str, str]:
    """Each '## Response N' section of an answer, by step key (``upstream_prompt.STEPS``)."""
    from tradingagents.agents.upstream_prompt import STEPS

    marks = list(_RESPONSE.finditer(text or ""))
    out = {}
    for i, m in enumerate(marks):
        n = int(m.group(1))
        if 1 <= n <= len(STEPS):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            out[STEPS[n - 1][0]] = text[m.end():end].strip()
    return out


def _json_of(section: str) -> dict | None:
    found = re.findall(r"```(?:json)?\s*(\{.*\})\s*```", section, flags=re.S) or re.findall(r"(\{.*\})", section, re.S)
    for raw in reversed(found):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def final_state(text: str, ticker: str, day: str) -> tuple[dict, dict] | None:
    """An answer in the per-step form, as the graph's final state (what an API run saves): each structured step's
    JSON validated against its schema and rendered by the same function the API run uses; free-text steps as
    written; debate turns labelled as the agents label them. ``(state, notes)``, or None when the answer has no
    '## Response N' sections."""
    from pydantic import ValidationError

    from tradingagents.agents import schemas as S
    from tradingagents.agents.upstream_prompt import STRUCTURED_STEPS, step_schema

    steps = parse_steps(text)
    if not steps:
        return None
    render = {"sentiment": S.render_sentiment_report, "research_manager": S.render_research_plan,
              "trader": S.render_trader_proposal, "portfolio_manager": S.render_pm_decision}
    out, notes, parsed = {}, {}, {}
    for step, body in steps.items():
        if step in STRUCTURED_STEPS:
            data = _json_of(body)
            try:
                obj = step_schema(step).model_validate(data or {})
                parsed[step] = obj
                out[step] = render[step](obj)
            except ValidationError as exc:   # kept as written, and said so
                out[step] = body
                notes[step] = f"not in the schema ({exc.error_count()} problem(s)); saved as written"
        else:
            out[step] = body
    say = lambda who, key: f"\n{who}: {out[key]}" if out.get(key) else ""  # noqa: E731
    bull, bear = say("Bull Analyst", "bull"), say("Bear Analyst", "bear")
    agg, con, neu = say("Aggressive Analyst", "aggressive"), say("Conservative Analyst", "conservative"), \
        say("Neutral Analyst", "neutral")
    state = {"company_of_interest": ticker, "trade_date": day,
             "market_report": out.get("market", ""), "sentiment_report": out.get("sentiment", ""),
             "news_report": out.get("news", ""), "fundamentals_report": out.get("fundamentals", ""),
             "investment_debate_state": {"bull_history": bull, "bear_history": bear, "history": bull + bear,
                                         "judge_decision": out.get("research_manager", "")},
             "investment_plan": out.get("research_manager", ""),
             "trader_investment_plan": out.get("trader", ""),
             "risk_debate_state": {"aggressive_history": agg, "conservative_history": con, "neutral_history": neu,
                                   "history": agg + con + neu, "judge_decision": out.get("portfolio_manager", "")},
             "final_trade_decision": out.get("portfolio_manager", "")}
    return state, {"notes": notes, "parsed": parsed, "steps": sorted(steps)}


def decision_from_steps(meta: dict) -> dict | None:
    """The portfolio manager's decision and the trader's proposal from a per-step answer, in ``parse_answer``'s
    shape; None without a valid portfolio manager answer."""
    pm, tr = meta["parsed"].get("portfolio_manager"), meta["parsed"].get("trader")
    if pm is None:
        return None
    d = {"rating": pm.rating.value, "executive_summary": pm.executive_summary,
         "investment_thesis": pm.investment_thesis, "price_target": pm.price_target, "time_horizon": pm.time_horizon,
         "insufficient_evidence": False}
    if tr is not None:
        d["trader"] = {"action": tr.action.value, "entry_price": tr.entry_price, "stop_loss": tr.stop_loss,
                       "position_sizing": tr.position_sizing}
    return d


def parse_answer(text: str) -> dict:
    """The portfolio manager's JSON from a pasted answer: the last JSON object with a valid ``rating``."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.S)
    if not candidates:   # no fenced block: any brace-delimited object that names a rating
        candidates = re.findall(r"(\{[^{}]*\"rating\"[^{}]*\})", text or "", flags=re.S)
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rating = str(data.get("rating", "")).strip().title()
        if rating in RATINGS:
            return {**data, "rating": rating, "insufficient_evidence": bool(data.get("insufficient_evidence"))}
    m = re.search(r"\b(?:final\s+)?rating\b[^A-Za-z]{0,12}(Buy|Overweight|Hold|Underweight|Sell)\b", text or "", re.I)
    if m:
        return {"rating": m.group(1).title(), "executive_summary": "", "investment_thesis": "",
                "insufficient_evidence": False}
    raise ValueError("no rating found: the answer must end with the JSON block the prompt asks for "
                     f"(rating one of {', '.join(RATINGS)})")


def decision_markdown(d: dict) -> str:
    """The same shape the agents' portfolio manager writes (5_portfolio/decision.md), plus the trader's
    proposal and any optional fields the answer gives."""
    out = (f"**Rating**: {d['rating']}\n\n**Executive Summary**: {d.get('executive_summary') or 'not provided'}\n\n"
           f"**Investment Thesis**: {d.get('investment_thesis') or 'not provided'}\n\n"
           f"**Price Target**: {d.get('price_target') if d.get('price_target') is not None else 'not provided'}\n\n"
           f"**Time Horizon**: {d.get('time_horizon') or 'not provided'}\n")
    if trader_line(d):
        out += "\n" + trader_line(d)
    if d.get("insufficient_evidence"):
        out += ("\n**Insufficient evidence**: the analysis abstained; the rating above is recorded only because "
                "the answer format requires one.\n")
    if d.get("evidence_ids"):
        out += f"\n**Evidence cited**: {', '.join(map(str, d['evidence_ids']))}\n"
    if d.get("entry_plan"):
        from tradingagents.agents.schemas import render_entry_plan

        out += "\n" + render_entry_plan(d["entry_plan"]) + "\n"
    return out


def entry_plan_of(d: dict) -> dict | None:
    """The answer's entry plan, normalised like the agents' (placeholders and formatted prices cleaned);
    as given when it does not fit the schema, so the local check reports what is wrong with it."""
    from pydantic import ValidationError

    from tradingagents.agents.schemas import EntryPlan

    raw = d.get("entry_plan")
    if not isinstance(raw, dict):
        return None
    try:
        return EntryPlan.model_validate(raw).model_dump(mode="json")
    except ValidationError:
        return raw
