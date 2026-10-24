"""Paid AI reading of a shortlist's news: structured events, facts kept apart from interpretation.

Only for a short list (``stock_rules.llm_shortlist`` stocks: the requested stocks with
the largest expected moves either way, and any held), only when asked, never on a
page load, a replay or a test. Every call goes through the AI gateway, which answers
a repeated request (same headlines, task, model, schema version and cutoff) from its
cache, makes concurrent identical requests wait for one, holds new ones to the
daily and monthly budgets, and logs what each cost.

The model gets numbered headlines with their sources and times and the app's own
numbers, and is told to use nothing else. Each event must cite headline numbers;
citations that do not exist are dropped. It extracts and explains: the numerical
forecasts stay the trained models'.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
SCHEMA = "stock-news-v1"


class Event(BaseModel):
    type: Literal["earnings", "guidance", "financing", "buyback_dividend", "litigation", "regulation", "management",
                  "mna", "analyst", "product_contract", "other"]
    sources: list[int] = Field(description="Numbers of the headlines (H1 = 1) this event comes from")
    fact: str = Field(description="What happened, as reported, in one sentence; no opinion")
    interpretation: str = Field(description="What it may mean for the stock, labelled as interpretation")
    direction: Literal["positive", "negative", "unclear"]
    materiality: Literal["high", "medium", "low"]


class Reading(BaseModel):
    events: list[Event]
    summary: str = Field(description="Two sentences a beginner can follow")


def prompt_for(ticker: str, name: str, cutoff: str, items: list[dict], numbers: str) -> str:
    lines = "\n".join(f"H{i}. [{(it.get('published_at') or '')[:16]}] {it['title']} ({it.get('publisher') or '?'})"
                      for i, it in enumerate(items, 1))
    return f"""You extract material events for {name} ({ticker}) from news, as of {cutoff}.

Use ONLY the numbered headlines and the numbers below; do not use anything you remember.
- One entry per distinct event; repeated coverage of the same event is one entry citing all its headlines.
- "fact" says what was reported, without opinion. "interpretation" is clearly your reading.
- Opinion pieces ("should you buy", lists, predictions) are not events unless they report one.
- Cite headline numbers for every event. If nothing material happened, return no events.

THE APP'S NUMBERS (from trained models; do not change them):
{numbers}

HEADLINES:
{lines or "(none)"}
"""


def read_news(ticker: str, name: str, cutoff: str, items: list[dict], numbers: str, config: dict, gateway,
              allow_paid: bool = False, fallbacks: tuple[str, ...] = ()) -> dict:
    from tradingagents.agents.structured import bind_structured, invoke_structured
    from tradingagents.ledger.ai_gateway import AIBlocked, AIInProgress
    from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
    from tradingagents.optimizer.market import _models

    if gateway is None:
        raise ValueError("reading news with AI is a paid call and needs the AI gateway")
    prompt = prompt_for(ticker, name, cutoff, items, numbers)
    errors = []
    for provider, model in _models(config, fallbacks):
        cfg = {**config, "llm_provider": provider}

        def work(callback, provider=provider, model=model, cfg=cfg):
            llm = create_llm_client(provider=provider, model=model,
                                    base_url=config.get("backend_url") if provider == config["llm_provider"] else None,
                                    callbacks=[callback], **build_llm_kwargs(cfg)).get_llm()
            structured = bind_structured(llm, Reading, "Stock news reading")
            for wait in (0, 3):
                time.sleep(wait)
                if (got := invoke_structured(structured, prompt, "Stock news reading")) is not None:
                    return got.model_dump()
            raise ValueError("the model did not return a structured reading")

        try:
            answer, _ = gateway.run(task="stock_news", provider=provider, model=model, schema_version=SCHEMA,
                                    info_cutoff=cutoff, inputs={"prompt": prompt}, work=work, est_usd=0.01,
                                    est_requests=2, allow_paid=allow_paid)
        except (AIBlocked, AIInProgress):
            raise
        except Exception as exc:  # out of credit or unreachable: the next provider, if the policy allows one
            errors.append(f"{provider}/{model}: {str(exc)[:160]}")
            continue
        return {**checked(answer, len(items)), "model": model, "ticker": ticker, "cutoff": cutoff}
    raise ValueError("No AI provider could read the news. " + " | ".join(errors))


def checked(answer: dict, n_items: int) -> dict:
    """Drop citations to headlines that do not exist, and events left with none."""
    events = []
    for e in answer.get("events", []):
        e["sources"] = [s for s in e.get("sources", []) if 1 <= s <= n_items]
        if e["sources"]:
            events.append(e)
    return {**answer, "events": events, "dropped": len(answer.get("events", [])) - len(events)}
