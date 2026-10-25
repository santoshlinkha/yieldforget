"""Run the analysis by hand in a chat app (ChatGPT, Gemini, Claude): one prompt out, one answer back.

``build_prompt`` gathers what the agents' tools would fetch, as of the analysis day (prices, indicators,
the verified snapshot, fundamentals, company and global news, a few macro series) and asks the chat
model to play the whole team and end with the portfolio manager's decision as a JSON block. It carries
raw data only: not whether the stock is held, and none of the app's own verdicts, ranks or signals, so the
chat's judgment is independent of the app's. ``parse_answer`` reads that block back. No API is called by either: the data comes from
the same cached vendors the agents use, and the model runs in the user's own chat subscription.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

from tradingagents.agents.schemas import PortfolioRating
from tradingagents.agents.tools import compact_indicator, compact_prices

INDICATORS = "close_10_ema,close_50_sma,close_200_sma,rsi,macd,macdh,boll_ub,boll_lb,atr,vwma"
MACRO = ("fed_funds_rate", "10y_treasury", "cpi", "unemployment")
LIMITS = {"fundamentals": 14000, "statement": 4000, "news": 5000, "global_news": 3500, "macro": 900}   # characters
RATINGS = [r.value for r in PortfolioRating]


def _cut(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + f"\n… [{len(text) - n:,} more characters left out]"


def _fetch(label: str, fn) -> str:
    try:
        out = fn()
        return out if isinstance(out, str) and out.strip() else f"({label}: no data)"
    except Exception as exc:  # one missing source must not stop the prompt
        return f"({label} unavailable: {str(exc)[:160]})"


def build_prompt(ticker: str, day: str, company: str = "") -> str:
    from tradingagents.dataflows.router import route_to_vendor
    from tradingagents.dataflows.vendors.yahoo.snapshot import build_verified_market_snapshot

    d = date.fromisoformat(day)
    start = (d - timedelta(days=120)).isoformat()
    week = (d - timedelta(days=7)).isoformat()
    prices = _fetch("prices", lambda: compact_prices(route_to_vendor("get_stock_data", ticker, start, day), 60))
    indicators = "\n\n".join(_fetch(i, lambda i=i: compact_indicator(route_to_vendor("get_indicators", ticker, i, day, 10)))
                             for i in INDICATORS.split(","))
    snapshot = _fetch("snapshot", lambda: build_verified_market_snapshot(ticker, day, 5))
    # The profile tool withholds today-only values (multiples, market cap) for a past day; the statements
    # come as filed by the day, as the fundamentals analyst's own tools serve them.
    fundamentals = _cut("\n\n".join([
        _fetch("fundamentals", lambda: route_to_vendor("get_fundamentals", ticker, day)),
        *(_cut(_fetch(name, lambda tool=tool: route_to_vendor(tool, ticker, "quarterly", day)), LIMITS["statement"])
          for name, tool in (("income statement", "get_income_statement"), ("balance sheet", "get_balance_sheet"),
                             ("cash flow", "get_cashflow")))]), LIMITS["fundamentals"])
    news = _cut(_fetch("company news", lambda: route_to_vendor("get_news", ticker, week, day)), LIMITS["news"])
    world = _cut(_fetch("global news", lambda: route_to_vendor("get_global_news", day, 7, 10)), LIMITS["global_news"])
    macro = "\n\n".join(_cut(_fetch(m, lambda m=m: route_to_vendor("get_macro_indicators", m, day, 365)), LIMITS["macro"])
                        for m in MACRO)
    who = f"{ticker}" + (f" ({company})" if company else "")
    # Raw data only: no holding status and none of the app's own verdicts, ranks or signals, so the chat
    # model's judgment is its own and can be compared with the app's rather than echo it.
    return f"""You are a trading research team analyzing {who} as of the market close on {day}. Treat {day} as today: use only the data below, which stops at that date, and never invent a number that is not in it. When the data does not support a claim, say so. Judge it as an investment for a long-only retirement account (no shorting, no leverage, no day trading), without assuming whether it is already owned.

Work through these roles in order, writing each as its own section:
1. Technical analyst: trend, momentum, support and resistance from the prices and indicators; quote exact values from the data.
2. Fundamentals analyst: business quality, growth, margins, balance sheet and valuation from the fundamentals.
3. News and economy analyst: what the company news and the global and macro data mean for this stock now.
4. Sentiment analyst: the tone of the news, analyst views and positioning in the data.
5. Bull researcher, then bear researcher: the strongest evidence-based case for and against owning it, each answering the other.
6. Research manager: weigh the debate and commit to a side; conflicting arguments alone are not a reason to Hold.
7. Trader: a concrete plan (entry, stop-loss as a price, position size relative to a standard position).
8. Risk debate: an aggressive, a conservative and a neutral risk analyst each judge the trader's plan.
9. Portfolio manager: the final decision.

Rating scale (the portfolio manager uses exactly one): Buy (strong conviction to enter or add), Overweight (favorable; increase gradually), Hold (keep the current position; no action), Underweight (reduce; take partial profits), Sell (exit, or avoid entry).

End your answer with the portfolio manager's decision as one JSON code block, exactly in this shape:
```json
{{"rating": "Buy | Overweight | Hold | Underweight | Sell", "executive_summary": "2-4 sentences: entry, sizing, key risk levels, horizon", "investment_thesis": "the evidence that decided it", "price_target": null, "time_horizon": "e.g. 3-6 months"}}
```

=== PRICES (daily, compact) ===
{prices}

=== TECHNICAL INDICATORS (last 10 days) ===
{indicators}

=== VERIFIED SNAPSHOT (source of truth for exact price and indicator values) ===
{snapshot}

=== FUNDAMENTALS ===
{fundamentals}

=== COMPANY NEWS (last 7 days) ===
{news}

=== GLOBAL AND ECONOMIC NEWS (last 7 days) ===
{world}

=== MACRO DATA (FRED) ===
{macro}
"""


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
            return {**data, "rating": rating}
    m = re.search(r"\b(?:final\s+)?rating\b[^A-Za-z]{0,12}(Buy|Overweight|Hold|Underweight|Sell)\b", text or "", re.I)
    if m:
        return {"rating": m.group(1).title(), "executive_summary": "", "investment_thesis": ""}
    raise ValueError("no rating found: the answer must end with the JSON block the prompt asks for "
                     f"(rating one of {', '.join(RATINGS)})")


def decision_markdown(d: dict) -> str:
    """The same shape the agents' portfolio manager writes (5_portfolio/decision.md)."""
    return (f"**Rating**: {d['rating']}\n\n**Executive Summary**: {d.get('executive_summary') or 'not provided'}\n\n"
            f"**Investment Thesis**: {d.get('investment_thesis') or 'not provided'}\n\n"
            f"**Price Target**: {d.get('price_target') if d.get('price_target') is not None else 'not provided'}\n\n"
            f"**Time Horizon**: {d.get('time_horizon') or 'not provided'}\n")
