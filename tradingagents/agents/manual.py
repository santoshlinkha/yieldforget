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


# A response heading in whatever form a chat app writes it: "## Response 12: ...", "**Response 12 of 12 - ...**",
# "### Request 12 — ...", "Response 12." on its own line.
_RESPONSE = re.compile(r"^[ \t>]*(?:#{1,6}[ \t]*)?[*_]{0,3}[ \t]*(?:Response|Request|Step)[ \t]+(\d{1,2})\b(?:[ \t]+of[ \t]+\d+)?"
                       r"[^\n]*$", re.M | re.I)
# The fields that tell which agent a JSON object answers, when the headings cannot be matched.
_KEYS = {"portfolio_manager": {"rating", "executive_summary"}, "trader": {"action", "reasoning"},
         "research_manager": {"recommendation", "strategic_actions"}, "sentiment": {"overall_band", "narrative"}}


def parse_steps(text: str) -> dict[str, str]:
    """Each response section of an answer, by step key (``upstream_prompt.STEPS``). A step answered twice keeps
    its last section."""
    from tradingagents.agents.upstream_prompt import STEPS

    text = text or ""
    marks = list(_RESPONSE.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        n = int(m.group(1))
        if 1 <= n <= len(STEPS):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end].strip()
            if body:
                out[STEPS[n - 1][0]] = body
    return out


def json_objects(text: str) -> list[dict]:
    """Every JSON object in ``text``, fenced or not, in order (a proper decoder, so nested braces are fine)."""
    dec, out, i = json.JSONDecoder(), [], 0
    text = (text or "").replace("“", '"').replace("”", '"')
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end


def _clean(data: dict) -> dict:
    """Markdown and stray spacing out of a chat's JSON values ("**Buy**" -> "Buy"); enum spellings normalised."""
    out = {}
    for k, v in data.items():
        if isinstance(v, str):
            v = re.sub(r"^[\s*_`]+|[\s*_`]+$", "", v)
            if k in ("rating", "recommendation", "action", "overall_band"):
                v = v.title().replace("Mildly bullish", "Mildly Bullish")
            if k == "confidence":
                v = v.lower()
            if k in ("price_target", "entry_price", "stop_loss", "overall_score"):
                v = v.replace("$", "").replace(",", "").strip() or None
        out[k] = v
    return out


def _json_of(section: str) -> dict | None:
    found = json_objects(section)
    return found[-1] if found else None


def final_state(text: str, ticker: str, day: str) -> tuple[dict, dict] | None:
    """An answer in the per-step form, as the graph's final state (what an API run saves): each structured step's
    JSON validated against its schema and rendered by the same function the API run uses; free-text steps as
    written; debate turns labelled as the agents label them. ``(state, notes)``, or None when the answer has no
    '## Response N' sections."""
    from pydantic import ValidationError

    from tradingagents.agents import schemas as S
    from tradingagents.agents.upstream_prompt import STRUCTURED_STEPS, step_schema

    steps = parse_steps(text)
    objects = json_objects(text)
    # The older form (one summary block holding the rating and a nested "trader"): read by ``parse_answer``.
    if len(steps) < 2 and any("rating" in o and isinstance(o.get("trader"), dict) for o in objects):
        return None
    # A structured step whose heading was not found: the last JSON object in the answer with that agent's fields.
    for obj in objects:
        for step, keys in _KEYS.items():
            if keys <= set(obj) and step_schema(step) is not None and (step not in steps or not _json_of(steps[step])):
                steps[step] = "```json\n" + json.dumps(obj) + "\n```"
    if not steps:
        return None
    render = {"sentiment": S.render_sentiment_report, "research_manager": S.render_research_plan,
              "trader": S.render_trader_proposal, "portfolio_manager": S.render_pm_decision}
    out, notes, parsed = {}, {}, {}
    for step, body in steps.items():
        if step in STRUCTURED_STEPS:
            data = _json_of(body)
            try:
                obj = step_schema(step).model_validate(_clean(data or {}))
                parsed[step] = obj
                out[step] = render[step](obj)
            except ValidationError as exc:   # kept as written, and said so
                out[step] = body
                notes[step] = ("no JSON object found" if not data else
                               "not in the schema: " + "; ".join(f"{'.'.join(map(str, e['loc']))} {e['msg']}"
                                                                  for e in exc.errors()[:3])) + "; saved as written"
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
         "investment_thesis": pm.investment_thesis, "price_target": pm.price_target, "stop_loss": pm.stop_loss,
         "time_horizon": pm.time_horizon,
         "insufficient_evidence": False}
    if tr is not None:
        d["trader"] = {"action": tr.action.value, "entry_price": tr.entry_price, "stop_loss": tr.stop_loss,
                       "price_target": tr.price_target, "position_sizing": tr.position_sizing}
    return d


def parse_answer(text: str) -> dict:
    """The portfolio manager's decision from a pasted answer: the last JSON object with a valid ``rating``
    (fenced or not), else a plain "Rating: ..." line."""
    for data in reversed(json_objects(text)):
        data = _clean(data)
        rating = str(data.get("rating", "")).strip().title()
        if rating in RATINGS:
            return {**data, "rating": rating, "insufficient_evidence": bool(data.get("insufficient_evidence"))}
    m = re.search(r"\b(?:final\s+)?rating\b[^A-Za-z]{0,12}(Buy|Overweight|Hold|Underweight|Sell)\b", text or "", re.I)
    if m:
        return {"rating": m.group(1).title(), "executive_summary": "", "investment_thesis": "",
                "insufficient_evidence": False}
    found = sorted(parse_steps(text))
    raise ValueError("no portfolio manager decision found: Response 12 must be one JSON object with \"rating\" "
                     f"(one of {', '.join(RATINGS)}), \"executive_summary\" and \"investment_thesis\". "
                     + (f"Found responses for: {', '.join(found)}." if found else
                        "No '## Response N' headings were found either; paste the whole answer."))


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
