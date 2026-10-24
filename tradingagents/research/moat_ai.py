"""Where a moat comes from, read from the company's own 10-K by an AI model: one paid call per filing.

The numbers can show that a company earns high returns; they cannot say why. This
asks the configured quick model to rate the five usual sources of a moat (switching
costs, network effects, cost advantages, intangible assets, efficient scale) from the
business section of the latest 10-K, and whether the edge looks durable.

Grounding rules, checked afterwards rather than trusted: every rating above "none"
must quote the filing word for word, and a quote that is not in the filing is marked
unverified and its rating dropped to "unclear". The model is told not to use what it
remembers about the company, so the reader can check every claim against the filing.
The call goes through the AI gateway: the same filing and numbers are answered from
the cache, and a new call is held to the budget.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

KINDS = ("switching_costs", "network_effects", "cost_advantage", "intangible_assets", "efficient_scale")
LABELS = {"switching_costs": "Switching costs", "network_effects": "Network effects", "cost_advantage": "Cost advantage",
          "intangible_assets": "Intangible assets (brands, patents, licences)", "efficient_scale": "Efficient scale"}


class Source(BaseModel):
    kind: Literal["switching_costs", "network_effects", "cost_advantage", "intangible_assets", "efficient_scale"]
    strength: Literal["none", "weak", "moderate", "strong"]
    evidence: str = Field(description="A sentence or phrase copied word for word from FILING TEXT that supports the "
                                      "rating; empty when the strength is none")
    why: str = Field(description="One or two plain sentences on how this source protects (or does not protect) profits")


class Assessment(BaseModel):
    sources: list[Source] = Field(description="Exactly one entry for each of the five kinds")
    durable: Literal["yes", "no", "unclear"] = Field(description="Does the edge look likely to last 10+ years?")
    durable_why: str
    threats: list[str] = Field(description="Up to three threats to the edge the filing itself names; empty if none")
    summary: str = Field(description="Two or three sentences a beginner can follow")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def verify(answer: dict, filing_text: str) -> dict:
    """Mark each quote found or not in the filing; an unfound quote's rating becomes "unclear"."""
    body = _norm(filing_text)
    for s in answer["sources"]:
        quote = _norm(s.get("evidence") or "")
        s["verified"] = bool(quote) and quote in body
        if s["strength"] != "none" and not s["verified"]:
            s["strength"] = "unclear"
    return answer


def prompt_for(symbol: str, name: str, filing: dict, numbers: str) -> str:
    return f"""You assess the economic moat of {name} ({symbol}) for a beginner investor.

Use ONLY the filing text and numbers below. Do not use anything you remember about the company:
your memory may be out of date, and the reader must be able to check every claim.

For each of the five sources of a moat, give a strength (none, weak, moderate, strong):
- switching_costs: customers would lose time, money or data by moving to a rival.
- network_effects: the product gets more valuable as more people use it.
- cost_advantage: it can make or deliver more cheaply than rivals (scale, process, location).
- intangible_assets: brands people pay more for, patents, licences or regulatory approvals.
- efficient_scale: a market only big enough for a few players, so newcomers are not worth it.

Rules:
- Any strength above "none" needs "evidence": a sentence or phrase copied WORD FOR WORD from FILING TEXT.
  If the filing does not support it, the strength is "none" and the evidence is empty.
- "durable": yes only if the filing and the numbers both point to an edge lasting 10+ years.
- "threats": only threats the filing itself names (competition, regulation, technology shifts).
- Plain English, no advice on buying or selling.

NUMBERS (the app's own checks from Yahoo's annual statements):
{numbers}

FILING TEXT ({filing["form"]} for the period ending {filing["period"]}, filed {filing["filed"]}; business section):
{filing["text"]}
"""


def moat_sources(symbol: str, name: str, filing: dict, numbers: str, config: dict, gateway, allow_paid: bool = False,
                 fallbacks: tuple[str, ...] = ()) -> dict:
    """The five sources rated from the filing, verified against it; through the AI gateway."""
    from tradingagents.agents.structured import bind_structured, invoke_structured
    from tradingagents.ledger.ai_gateway import AIBlocked, AIInProgress
    from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
    from tradingagents.optimizer.market import _models

    if gateway is None:
        raise ValueError("the moat sources are a paid AI call and need the AI gateway")
    prompt = prompt_for(symbol, name, filing, numbers)
    errors = []
    for provider, model in _models(config, fallbacks):
        cfg = {**config, "llm_provider": provider}

        def work(callback, provider=provider, model=model, cfg=cfg):
            llm = create_llm_client(provider=provider, model=model,
                                    base_url=config.get("backend_url") if provider == config["llm_provider"] else None,
                                    callbacks=[callback], **build_llm_kwargs(cfg)).get_llm()
            structured = bind_structured(llm, Assessment, "Moat assessment")
            for wait in (0, 3):   # a busy provider answers 503 for a moment
                time.sleep(wait)
                if (got := invoke_structured(structured, prompt, "Moat assessment")) is not None:
                    return got.model_dump()
            raise ValueError("the model did not return a structured assessment")

        try:
            answer, _ = gateway.run(task="moat_sources", provider=provider, model=model, schema_version="moat-v1",
                                    info_cutoff=filing["filed"], inputs={"prompt": prompt}, work=work, est_usd=0.03,
                                    est_requests=2, allow_paid=allow_paid)
        except (AIBlocked, AIInProgress):
            raise
        except Exception as exc:  # out of credit or unreachable: the next provider, if the policy allows one
            errors.append(f"{provider}/{model}: {str(exc)[:160]}")
            logger.warning("moat sources with %s/%s failed: %s", provider, model, exc)
            continue
        answer = verify(answer, filing["text"])
        order = {k: i for i, k in enumerate(KINDS)}
        answer["sources"] = sorted(answer["sources"], key=lambda s: order.get(s["kind"], 9))
        for s in answer["sources"]:
            s["label"] = LABELS.get(s["kind"], s["kind"])
        return {**answer, "symbol": symbol, "model": model, "filing": {k: filing[k] for k in ("form", "filed", "period", "url")}}
    raise ValueError("No AI provider could assess the moat. " + " | ".join(errors))
