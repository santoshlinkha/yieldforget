"""Recent market trends, and an explanation of them grounded in the news.

``market_overview`` is arithmetic on closing prices: returns over standard
windows for the indexes, sectors, bonds, gold and volatility; the S&P 500's
leaders, laggards and breadth; and the disruptive themes. Nothing is predicted.

``market_brief`` asks the quick LLM, once, why those moves happened. It is given
only the numbers above, numbered recent headlines and FRED macro data. Each trend
names the headlines it rests on by number, so the page can link them; a trend
that cites none is marked low confidence whatever the model said, since nobody's
reporting supports it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tradingagents.agents.structured import bind_structured, invoke_structured
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors.fred import get_macro_data
from tradingagents.dataflows.vendors.yahoo.news import get_market_headlines
from tradingagents.llm_clients import build_llm_kwargs, create_llm_client

from .universe import THEME_GROUPS

logger = logging.getLogger(__name__)

# (ticker, name, group, kind): "price" moves in percent, "level" in points (yields, VIX).
MARKET_ROWS = [
    ("SPY", "S&P 500", "Stocks", "price"),
    ("QQQ", "Nasdaq-100", "Stocks", "price"),
    ("DIA", "Dow Jones", "Stocks", "price"),
    ("IWM", "Small caps (Russell 2000)", "Stocks", "price"),
    ("VXUS", "International", "Stocks", "price"),
    ("VWO", "Emerging markets", "Stocks", "price"),
    ("XLK", "Technology", "Sectors", "price"),
    ("XLC", "Communication", "Sectors", "price"),
    ("XLY", "Consumer discretionary", "Sectors", "price"),
    ("XLF", "Financials", "Sectors", "price"),
    ("XLV", "Health care", "Sectors", "price"),
    ("XLI", "Industrials", "Sectors", "price"),
    ("XLE", "Energy", "Sectors", "price"),
    ("XLB", "Materials", "Sectors", "price"),
    ("XLP", "Consumer staples", "Sectors", "price"),
    ("XLU", "Utilities", "Sectors", "price"),
    ("XLRE", "Real estate", "Sectors", "price"),
    ("^TNX", "10-year Treasury yield", "Rates & more", "level"),
    ("TLT", "Long Treasuries (20+ yr)", "Rates & more", "price"),
    ("HYG", "High-yield bonds", "Rates & more", "price"),
    ("GLD", "Gold", "Rates & more", "price"),
    ("DBC", "Commodities", "Rates & more", "price"),
    ("BTC-USD", "Bitcoin", "Rates & more", "price"),
    ("^VIX", "VIX (expected volatility)", "Rates & more", "level"),
]
HEADLINE_TICKERS = ["SPY", "QQQ", "IWM", "^TNX", "^VIX", "GLD"]
PERIODS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "YTD": None, "1Y": 252}
NEWS_QUERIES = [
    "stock market today", "Wall Street stocks week", "Federal Reserve interest rates",
    "Treasury yields bonds", "Nasdaq tech stocks AI", "earnings season results",
    "oil prices energy", "inflation CPI jobs report", "dollar gold bitcoin",
]
MACRO = ["fed_funds", "2y_treasury", "10y_treasury", "cpi", "unemployment", "consumer_sentiment"]


def market_tickers() -> list[str]:
    return [t for t, *_ in MARKET_ROWS]


def _changes(series: pd.Series, kind: str) -> dict[str, float | None]:
    s = series.dropna()
    out = {}
    for label, days in PERIODS.items():
        if days is None:
            base = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
            base_value = base.iloc[-1] if len(base) else None
        else:
            base_value = s.iloc[-1 - days] if len(s) > days else None
        if base_value is None or not base_value:
            out[label] = None
        else:
            out[label] = float(s.iloc[-1] - base_value) if kind == "level" else float(s.iloc[-1] / base_value - 1)
    return out


def _trend(series: pd.Series) -> dict:
    s = series.dropna()
    year = s.iloc[-252:]
    return {
        "above_50": bool(len(s) >= 50 and s.iloc[-1] > s.iloc[-50:].mean()),
        "above_200": bool(len(s) >= 200 and s.iloc[-1] > s.iloc[-200:].mean()),
        "from_high": float(s.iloc[-1] / year.max() - 1) if len(year) else None,
        "spark": [round(float(v), 4) for v in s.iloc[-126:].iloc[::3]],
    }


def market_overview(closes: pd.DataFrame, sp500: dict[str, str]) -> dict:
    """Recent moves across the market, from closing prices only."""
    rows = []
    for ticker, name, group, kind in MARKET_ROWS:
        if ticker not in closes or closes[ticker].dropna().empty:
            continue
        s = closes[ticker].dropna()
        rows.append({"ticker": ticker, "name": name, "group": group, "kind": kind,
                     "last": float(s.iloc[-1]), "date": s.index[-1].strftime("%Y-%m-%d"),
                     "changes": _changes(s, kind), **_trend(s)})

    members = [t for t in sp500 if t in closes]
    stocks = closes[members].ffill()
    one_month = (stocks.iloc[-1] / stocks.iloc[-22] - 1).dropna()
    one_week = (stocks.iloc[-1] / stocks.iloc[-6] - 1).dropna()

    def ranked(r: pd.Series, n=6):
        top, bottom = r.nlargest(n), r.nsmallest(n)
        return ([{"ticker": t, "name": sp500[t], "change": float(v)} for t, v in top.items()],
                [{"ticker": t, "name": sp500[t], "change": float(v)} for t, v in bottom.items()])

    leaders_1m, laggards_1m = ranked(one_month)
    leaders_1w, laggards_1w = ranked(one_week)
    above_50 = float((stocks.iloc[-1] > stocks.iloc[-50:].mean()).mean()) if len(members) else None
    above_200 = float((stocks.iloc[-1] > stocks.iloc[-200:].mean()).mean()) if len(members) else None
    up_1m = float((one_month > 0).mean()) if len(one_month) else None

    themes = []
    for theme, tickers in THEME_GROUPS.items():
        moves = {t: (closes[t].dropna().iloc[-1] / closes[t].dropna().iloc[-22] - 1)
                 for t in tickers if t in closes and closes[t].dropna().size > 22}
        if moves:
            best = max(moves, key=moves.get)
            worst = min(moves, key=moves.get)
            themes.append({"theme": theme, "median_1m": float(np.median(list(moves.values()))),
                           "best": [best, float(moves[best])], "worst": [worst, float(moves[worst])],
                           "count": len(moves)})
    themes.sort(key=lambda x: -x["median_1m"])

    as_of = max((r["date"] for r in rows), default=date.today().isoformat())
    return {"as_of": as_of, "rows": rows, "themes": themes,
            "sp500": {"count": len(members), "above_50": above_50, "above_200": above_200, "up_1m": up_1m,
                      "leaders_1m": leaders_1m, "laggards_1m": laggards_1m,
                      "leaders_1w": leaders_1w, "laggards_1w": laggards_1w}}


class Trend(BaseModel):
    title: str = Field(description="Short headline for the trend, e.g. 'Tech led, small caps lagged'")
    direction: Literal["up", "down", "mixed"]
    what: str = Field(description="What moved and by how much, using the numbers given")
    why: str = Field(description="Why, in plain English, tied to the headlines or macro data given. "
                                 "If nothing given explains it, say so plainly.")
    sources: list[int] = Field(description="Numbers of the headlines (H1 = 1) that support the why; "
                                           "empty when none does")
    data: list[str] = Field(description="Figures copied from MARKET NUMBERS or MACRO DATA that the trend uses")
    confidence: Literal["high", "medium", "low"]


class MarketBrief(BaseModel):
    summary: str = Field(description="Two or three sentences a beginner can follow")
    trends: list[Trend] = Field(description="The 3 to 5 most important trends")
    watch: list[str] = Field(description="Upcoming events or open questions the headlines mention; empty if none")


def _numbers_block(overview: dict) -> str:
    def fmt(row, label):
        v = row["changes"].get(label)
        if v is None:
            return "n/a"
        return f"{v:+.2f} pts" if row["kind"] == "level" else f"{v:+.1%}"

    lines = ["Ticker | Name | Last | 1D | 1W | 1M | 3M | YTD | vs 52-week high"]
    for r in overview["rows"]:
        high = f"{r['from_high']:+.1%}" if r["from_high"] is not None else "n/a"
        lines.append(f"{r['ticker']} | {r['name']} | {r['last']:.2f} | "
                     + " | ".join(fmt(r, p) for p in ("1D", "1W", "1M", "3M", "YTD")) + f" | {high}")
    sp = overview["sp500"]
    if sp["count"]:
        lines.append(f"\nS&P 500 breadth: {sp['above_50']:.0%} of members above their 50-day average, "
                     f"{sp['above_200']:.0%} above their 200-day average, {sp['up_1m']:.0%} up over 1 month.")
        lines.append("1-month leaders: " + ", ".join(f"{x['ticker']} ({x['name']}) {x['change']:+.1%}" for x in sp["leaders_1m"]))
        lines.append("1-month laggards: " + ", ".join(f"{x['ticker']} ({x['name']}) {x['change']:+.1%}" for x in sp["laggards_1m"]))
    if overview["themes"]:
        lines.append("Themes, median 1-month move: " + ", ".join(f"{t['theme']} {t['median_1m']:+.1%}" for t in overview["themes"]))
    return "\n".join(lines)


def _checked(trend: Trend, headlines: list[dict]) -> dict:
    """The trend with its cited headlines attached; uncited or mis-cited means low confidence."""
    cited = [headlines[n - 1] for n in dict.fromkeys(trend.sources) if 1 <= n <= len(headlines)]
    out = {**trend.model_dump(), "sources": [{"title": h["title"], "publisher": h["publisher"], "link": h["link"]}
                                             for h in cited]}
    if not cited:
        out["confidence"] = "low"
    return out


def market_brief(overview: dict, config: dict, gateway=None, allow_paid: bool = False,
                 fallbacks: tuple[str, ...] = ()) -> dict:
    """One LLM call explaining the overview from headlines and macro data it is given.

    The call goes through the AI gateway: the same prompt to the same model is
    answered from the cache, and a new one is held to the budget. Another provider
    is tried after a failure only when it is listed in ``fallbacks`` (the spending
    policy), never by default.
    """
    as_of = overview["as_of"]
    with run_config(config):
        headlines = get_market_headlines(as_of, NEWS_QUERIES, look_back_days=7)
        macro = []
        for name in MACRO:
            try:
                macro.append(get_macro_data(name, as_of, look_back_days=400)[:900])
            except Exception as exc:  # a missing FRED key leaves the brief to the headlines
                logger.info("FRED %s unavailable: %s", name, exc)
                break
    numbers = _numbers_block(overview)
    news = "\n".join(f"H{i}. [{h['pub_date'][:10] if h['pub_date'] else 'undated'}] {h['title']} ({h['publisher']})"
                     + (f": {h['summary'][:240]}" if h["summary"] else "") for i, h in enumerate(headlines, 1))
    prompt = f"""You explain recent stock-market moves to a beginner investor. Today is {as_of}.

Use ONLY the three sources below. Do not use anything you remember about events, prices or
companies: your memory may be out of date, and the reader needs to check every claim.

Rules:
- Each trend's "what" uses the numbers from MARKET NUMBERS.
- Each trend's "why" must come from the headlines or macro data below. List the numbers of the
  headlines it rests on in "sources" (H3 -> 3) and the figures it uses in "data".
- If no headline explains a move, leave "sources" empty, say "The headlines here do not
  explain this", and set confidence to low. Never invent a cause.
- No predictions and no advice on what to buy or sell.
- Plain English, short sentences, no jargon without a quick definition.

MARKET NUMBERS (from closing prices; yields and VIX change in points, everything else in %):
{numbers}

HEADLINES FROM THE LAST 7 DAYS:
{news or "(none found)"}

MACRO DATA (FRED):
{chr(10).join(macro) if macro else "(not available)"}
"""
    if gateway is None:
        raise ValueError("the market brief is a paid AI call and needs the AI gateway")
    from tradingagents.ledger.ai_gateway import AIBlocked, AIInProgress

    errors = []
    for provider, model in _models(config, fallbacks):
        cfg = {**config, "llm_provider": provider}

        def work(callback, provider=provider, model=model, cfg=cfg):
            llm = create_llm_client(provider=provider, model=model,
                                    base_url=config.get("backend_url") if provider == config["llm_provider"] else None,
                                    callbacks=[callback], **build_llm_kwargs(cfg)).get_llm()
            structured = bind_structured(llm, MarketBrief, "Market brief")
            brief = None
            # A busy provider answers 503 for a moment; one retry before settling for free text.
            for wait in (0, 3):
                time.sleep(wait)
                if (brief := invoke_structured(structured, prompt, "Market brief")) is not None:
                    break
            if brief is None:
                return {"summary": llm.invoke(prompt).content, "trends": [], "watch": [], "structured": False}
            return {"summary": brief.summary, "trends": [t.model_dump() for t in brief.trends],
                    "watch": brief.watch, "structured": True}

        try:
            answer, _ = gateway.run(task="market_brief", provider=provider, model=model, schema_version="brief-v1",
                                    info_cutoff=as_of, inputs={"prompt": prompt}, work=work, est_usd=0.02,
                                    est_requests=3, allow_paid=allow_paid)
        except (AIBlocked, AIInProgress):
            raise
        except Exception as exc:  # out of credit or unreachable: the next provider, if the policy allows one
            errors.append(f"{provider}/{model}: {str(exc)[:160]}")
            logger.warning("market brief with %s/%s failed: %s", provider, model, exc)
            continue
        trends = [_checked(Trend(**t), headlines) for t in answer["trends"]]
        return {"as_of": as_of, "summary": answer["summary"], "trends": trends, "watch": answer["watch"],
                "headlines": headlines, "model": model, "structured": answer["structured"]}
    raise ValueError("No AI provider could write the explanation. " + " | ".join(errors))


def _models(config: dict, fallbacks: tuple[str, ...] = (), openrouter_models: list[str] | None = None) -> list[tuple[str, str]]:
    """The configured quick model, then the free OpenRouter models, then Gemini, each only when the spending
    policy lists that provider as a fallback and its key is set."""
    models = [(config["llm_provider"], config["quick_think_llm"])]
    if "openrouter" in fallbacks and os.environ.get("OPENROUTER_API_KEY"):
        if openrouter_models is None:
            from tradingagents.ledger.store import GLOBAL_DEFAULTS

            openrouter_models = GLOBAL_DEFAULTS["ai_openrouter_models"]
        models += [("openrouter", m) for m in openrouter_models if ("openrouter", m) != models[0]]
    if "google" in fallbacks and config["llm_provider"] != "google" and os.environ.get("GOOGLE_API_KEY"):
        models += [("google", "gemini-3.8-flash"), ("google", "gemini-3.5-flash")]   # the second when the first is busy
    return models


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
