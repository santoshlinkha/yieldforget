"""Run the analysis by hand in a chat app (ChatGPT, Gemini, Claude): one prompt out, one answer back.

``build_prompt`` sends one compact evidence packet (``agents.packet``): stable, grounded instructions first,
then the instrument's canonical data as of the analysis day, read from local caches only. It carries data
only: not whether the stock is held, and none of the app's own verdicts, ranks or signals, so the chat's
judgment is independent of the app's. ``parse_answer`` reads the JSON block back; the contract is the
portfolio manager's (rating, executive summary, investment thesis, price target, time horizon), with two
optional additions the packet's instructions ask for (``evidence_ids``, ``insufficient_evidence``).
No API is called: the model runs in the user's own chat subscription.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tradingagents.agents.packet import PACKET_VERSION
from tradingagents.agents.packet import build_prompt as _packet_prompt
from tradingagents.agents.schemas import PortfolioRating

RATINGS = [r.value for r in PortfolioRating]
PROMPT_VERSION = PACKET_VERSION


def build_prompt(ticker: str, day: str, cache_dir: Path, limits: dict | None = None, with_packet: bool = False):
    """The whole prompt for one instrument and cutoff (instructions, then the evidence packet); with
    ``with_packet``, ``(text, packet)`` so the caller can save the packet the answer will cite."""
    text, packet = _packet_prompt(ticker.upper(), day, cache_dir, limits)
    return (text, packet) if with_packet else text


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
    """The same shape the agents' portfolio manager writes (5_portfolio/decision.md), plus the packet's
    optional fields when the answer gives them."""
    out = (f"**Rating**: {d['rating']}\n\n**Executive Summary**: {d.get('executive_summary') or 'not provided'}\n\n"
           f"**Investment Thesis**: {d.get('investment_thesis') or 'not provided'}\n\n"
           f"**Price Target**: {d.get('price_target') if d.get('price_target') is not None else 'not provided'}\n\n"
           f"**Time Horizon**: {d.get('time_horizon') or 'not provided'}\n")
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
