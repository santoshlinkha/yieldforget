"""The AI's own first screen: it picks and ranks the stocks it would analyse, before the full TradingAgents team runs.

The TradingAgents framework (Tauric Research) looks at a stock through four analysts: market (price and technicals),
social sentiment, news, and fundamentals. A full run of the team on 500 stocks would cost hundreds of runs, so the
first pass gives the model one compact fact sheet of every stock the Screener ranked, arranged under those lenses,
and asks it to rank the ten it would most want to buy, with a short reason under each lens. The full team then
analyses the top of that ranking, in order.

The fact sheet holds facts only: prices and returns, analyst consensus and targets, quality and valuation. None of the
app's own model scores or verdicts are in it, so the AI's choice is its own. It is told to use nothing else (no memory
of later events), and any ticker not on the sheet is dropped. The call goes through the AI gateway: budgeted, logged
and cached, so the same sheet on the same day is answered from the cache.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

SCHEMA = "ai-screen-v1"
TOP = 10


class Pick(BaseModel):
    ticker: str
    technical: str = Field(description="Market analyst: what the price trend and volatility say, one short line")
    sentiment: str = Field(description="Sentiment analyst: what analysts' consensus and targets say, one short line")
    news: str = Field(description="News analyst: the setting or catalysts the facts suggest (no outside news), one line")
    fundamentals: str = Field(description="Fundamentals analyst: quality and valuation, one short line")
    why: str = Field(description="Why this stock ranks where it does, one sentence")


class Screening(BaseModel):
    picks: list[Pick] = Field(description=f"The {TOP} stocks you would most want to buy now, best first")
    note: str = Field(description="One sentence on what you looked for")


def fact_sheet(rows: list[dict], close: pd.DataFrame | None) -> tuple[str, list[str]]:
    """One line per stock under the four lenses, and the tickers on it."""
    lines, tickers = [], []
    ret = lambda s, n: float(s.iloc[-1] / s.iloc[-n - 1] - 1) if len(s) > n else np.nan  # noqa: E731
    for r in rows:
        t = r["ticker"]
        tech = ""
        if close is not None and t in close:
            s = close[t].dropna()
            if len(s) > 70:
                vol = float(np.log(s).diff().iloc[-63:].std() * np.sqrt(252))
                high = float(s.iloc[-252:].max())
                tech = (f"ret1m {ret(s, 21):+.1%} ret3m {ret(s, 63):+.1%} ret12m {ret(s, 252):+.1%} vol {vol:.0%} "
                        f"off-high {s.iloc[-1] / high - 1:+.1%}")
        f = lambda v, fmt: "na" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)  # noqa: E731
        lines.append(f"{t} | {r.get('sector') or ''} | price {f(r.get('price'), '.2f')} | TECH {tech or 'na'} | "
                     f"SENT consensus {f(r.get('analyst_mean'), '.2f')} (1 strong buy..5 sell) of {r.get('analysts') or 0} "
                     f"analysts, target upside {f(r.get('target_upside'), '+.1%')} | "
                     f"FUND quality {f(r.get('quality_score'), '.0f')}/100, moat {r.get('moat_passed', 'na')}/5, "
                     f"fwd P/E {f(r.get('forward_pe'), '.1f')}, implied growth {f(r.get('implied_growth'), '+.1%')}")
        tickers.append(t)
    return "\n".join(lines), tickers


def prompt_for(sheet: str, cutoff: str) -> str:
    return (
        "You are the TradingAgents research team (Tauric Research's framework): a market analyst (price and "
        "technicals), a sentiment analyst, a news analyst and a fundamentals analyst, doing a first screen together.\n"
        f"Below is a fact sheet of S&P 500 stocks as of the close of {cutoff}. Use only these facts: no memory of "
        "prices, news or events after that date, and no stock that is not on the sheet.\n"
        f"Rank the {TOP} stocks you would most want to buy now for the next month or so, best first. For each, give "
        "one short line from each analyst and one sentence on why it ranks there. Prefer a balance: strong but not "
        "overextended trends, supportive analysts, sound quality at a reasonable valuation. Avoid near-duplicates "
        "(two share classes of one company).\n\n"
        "FACT SHEET (ticker | sector | price | TECH | SENT | FUND):\n" + sheet)


def screen(rows: list[dict], close: pd.DataFrame | None, cutoff: str, config: dict, gateway,
           allow_paid: bool = True) -> dict:
    """``{"picks": [...ranked...], "note", "model", "how", "considered"}``; picks only from the sheet."""
    from tradingagents.agents.structured import bind_structured, invoke_structured
    from tradingagents.llm_clients import build_llm_kwargs, create_llm_client

    sheet, tickers = fact_sheet(rows, close)
    prompt = prompt_for(sheet, cutoff)
    provider, model = config["llm_provider"], config["quick_think_llm"]

    def work(callback):
        llm = create_llm_client(provider=provider, model=model, base_url=config.get("backend_url"),
                                callbacks=[callback], **build_llm_kwargs(config)).get_llm()
        structured = bind_structured(llm, Screening, "AI screen")
        for wait in (0, 3):
            time.sleep(wait)
            if (got := invoke_structured(structured, prompt, "AI screen")) is not None:
                return got.model_dump()
        raise ValueError("the model did not return a ranked screen")

    answer, how = gateway.run(task="ai_screen", provider=provider, model=model, schema_version=SCHEMA,
                              info_cutoff=cutoff, inputs={"prompt": prompt}, work=work, est_usd=0.02,
                              est_requests=2, allow_paid=allow_paid)
    known, seen, picks = set(tickers), set(), []
    for p in answer.get("picks") or []:
        t = str(p.get("ticker") or "").strip().upper()
        if t in known and t not in seen:                 # only stocks on the sheet, each once
            seen.add(t)
            picks.append({**p, "ticker": t, "rank": len(picks) + 1})
    if not picks:
        raise ValueError("the AI screen named no stock from the fact sheet")
    return {"picks": picks[:TOP], "note": answer.get("note"), "model": f"{provider}/{model}", "how": how,
            "considered": len(tickers), "cutoff": cutoff}
