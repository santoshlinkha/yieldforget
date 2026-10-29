"""The TradingAgents team's own prompts (TauricResearch), as one prompt to paste into a chat app.

The text is not rewritten here: each agent node from ``tradingagents.agents`` is run against a stand-in
model that records what the node would send and answers with a placeholder naming the step. The chat
then plays every agent in the graph's order, writing each step's output where the next prompt quotes it:

  analysts (market, sentiment, news, fundamentals) -> bull -> bear -> research manager -> trader
  -> aggressive -> conservative -> neutral -> portfolio manager

Analysts call tools in the graph; in a chat there are none, so the same tool functions are called here
(through the configured vendors and the local store) and their output is pasted under each analyst's
prompt. The sentiment analyst fetches its own sources, so its prompt already holds them.

The app's own additions to these prompts (the entry-plan instructions and the reference close) are left
out, so the copied prompt is upstream's. The answer ends with the portfolio manager's decision and the
trader's proposal as one JSON block, which ``manual.parse_answer`` reads.
"""

from __future__ import annotations

import inspect
import json
from datetime import date, timedelta
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

PROMPT_FORMAT = "tradingagents-upstream-v2"   # v2: structured steps answered in the API's own schemas

# (step key, heading the chat writes, state field the next prompts read)
STEPS = [
    ("market", "Market Analyst report", "market_report"),
    ("sentiment", "Sentiment Analyst report", "sentiment_report"),
    ("news", "News Analyst report", "news_report"),
    ("fundamentals", "Fundamentals Analyst report", "fundamentals_report"),
    ("bull", "Bull Researcher argument", None),
    ("bear", "Bear Researcher argument", None),
    ("research_manager", "Research Manager investment plan", "investment_plan"),
    ("trader", "Trader proposal", "trader_investment_plan"),
    ("aggressive", "Aggressive Risk Analyst argument", None),
    ("conservative", "Conservative Risk Analyst argument", None),
    ("neutral", "Neutral Risk Analyst argument", None),
    ("portfolio_manager", "Portfolio Manager final decision", "final_trade_decision"),
]
# The steps the API calls with structured output: the model returns the schema's fields, which the app renders
# into the saved report (``schemas.render_*``). Everything else is the message's free text.
STRUCTURED_STEPS = ("sentiment", "research_manager", "trader", "portfolio_manager")
APP_FIELDS = {"entry_plan", "entry_plan_review"}   # the app's own additions, left out like its prompt text


def step_schema(step: str):
    from tradingagents.agents import schemas as S

    return {"sentiment": S.SentimentReport, "research_manager": S.ResearchPlan, "trader": S.TraderProposal,
            "portfolio_manager": S.PortfolioDecision}.get(step)


def schema_block(step: str) -> str:
    """The structured-output schema the API binds for ``step``: each field's name, type and description."""
    model = step_schema(step)
    js = model.model_json_schema()
    defs = js.get("$defs", {})
    required = set(js.get("required", []))
    lines = []
    for name, f in js["properties"].items():
        if name in APP_FIELDS:
            continue
        ref = (f.get("$ref") or "").split("/")[-1]
        options = f.get("anyOf") or [f]
        kinds = []
        for o in options:
            r = (o.get("$ref") or "").split("/")[-1]
            if r and "enum" in defs.get(r, {}):
                kinds.append("one of " + " | ".join(map(str, defs[r]["enum"])))
            elif "enum" in o:
                kinds.append("one of " + " | ".join(map(str, o["enum"])))
            else:
                kinds.append(o.get("type", "string"))
        if ref and "enum" in defs.get(ref, {}):
            kinds = ["one of " + " | ".join(map(str, defs[ref]["enum"]))]
        bounds = "".join(f", {k} {f[k]}" for k in ("minimum", "maximum") if k in f)
        desc = (f.get("description") or "").strip()
        lines.append(f'- "{name}" ({" or ".join(kinds)}{bounds}; {"required" if name in required else "optional, null when absent"}): {desc}')
    return "\n".join(lines)


INDICATORS = "close_50_sma,close_200_sma,close_10_ema,macd,macds,macdh,rsi,boll,boll_ub,boll_lb,atr,vwma"


def placeholder(step: str) -> str:
    n, heading = next((i + 1, h) for i, (k, h, _) in enumerate(STEPS) if k == step)
    return f"[the {heading} you wrote in step {n}]"


class _Recorder(BaseChatModel):
    """A chat model that answers nothing: it keeps each prompt and returns the step's placeholder."""

    step: str = ""
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "prompt-recorder"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("the recorder answers in text")   # every node takes its free-text path

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append((self.step, messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=placeholder(self.step)))])


def _as_text(messages: list[BaseMessage]) -> str:
    out = []
    for m in messages:
        role = {"system": "SYSTEM", "human": "USER", "ai": "ASSISTANT"}.get(m.type, m.type.upper())
        if m.type == "human" and m.content == m.content.strip() and len(m.content) < 16:
            continue   # the graph's opening message is only the ticker
        out.append(f"[{role}]\n{m.content}")
    return "\n\n".join(out)


def _strip_app_additions(text: str) -> str:
    from tradingagents.agents.entry_plan import PM_INSTRUCTION, RISK_INSTRUCTION, TRADER_INSTRUCTION

    for extra in (PM_INSTRUCTION, RISK_INSTRUCTION, TRADER_INSTRUCTION):
        if extra:
            text = text.replace(extra, "")
    return text


def _tool_data(step: str, ticker: str, day: str) -> list[tuple[str, str]]:
    """What the analyst's tools return for the usual calls, fetched now (as of ``day``)."""
    from tradingagents.agents import tools as T

    d = date.fromisoformat(day)
    back = lambda days: str(d - timedelta(days=days))  # noqa: E731
    calls = {
        "market": [("get_stock_data", T.get_stock_data, dict(symbol=ticker, start_date=back(365), end_date=day)),
                   ("get_indicators", T.get_indicators, dict(symbol=ticker, indicator=INDICATORS, curr_date=day)),
                   ("get_verified_market_snapshot", T.get_verified_market_snapshot, dict(symbol=ticker, curr_date=day))],
        "news": [("get_news", T.get_news, dict(ticker=ticker, start_date=back(7), end_date=day)),
                 ("get_global_news", T.get_global_news, dict(curr_date=day)),
                 *(("get_macro_indicators", T.get_macro_indicators, dict(indicator=i, curr_date=day))
                   for i in ("fed_funds_rate", "10y_treasury", "cpi", "unemployment"))],
        "fundamentals": [("get_fundamentals", T.get_fundamentals, dict(ticker=ticker, curr_date=day)),
                         ("get_income_statement", T.get_income_statement, dict(ticker=ticker, curr_date=day)),
                         ("get_balance_sheet", T.get_balance_sheet, dict(ticker=ticker, curr_date=day)),
                         ("get_cashflow", T.get_cashflow, dict(ticker=ticker, curr_date=day)),
                         ("get_insider_transactions", T.get_insider_transactions, dict(ticker=ticker))],
    }.get(step, [])
    out = []
    for name, fn, args in calls:
        if "trade_date" in inspect.signature(fn.func).parameters:
            args = {**args, "trade_date": day}
        try:
            result = fn.func(**args)
        except Exception as exc:   # one vendor failing leaves the others
            result = f"<{name} failed: {type(exc).__name__}: {exc}>"
        call = ", ".join(f"{k}={v!r}" for k, v in args.items() if k not in ("symbol", "ticker", "trade_date"))
        out.append((f"{name}({call})", str(result)))
    return out


def _nodes(llm):
    from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
    from tradingagents.agents.analysts.market_analyst import create_market_analyst
    from tradingagents.agents.analysts.news_analyst import create_news_analyst
    from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    from tradingagents.agents.managers.research_manager import create_research_manager
    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
    from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
    from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
    from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
    from tradingagents.agents.trader.trader import create_trader

    return {"market": create_market_analyst(llm), "sentiment": create_sentiment_analyst(llm),
            "news": create_news_analyst(llm), "fundamentals": create_fundamentals_analyst(llm),
            "bull": create_bull_researcher(llm), "bear": create_bear_researcher(llm),
            "research_manager": create_research_manager(llm), "trader": create_trader(llm),
            "aggressive": create_aggressive_debator(llm), "conservative": create_conservative_debator(llm),
            "neutral": create_neutral_debator(llm), "portfolio_manager": create_portfolio_manager(llm)}


def _merge(state: dict, update: dict) -> None:
    for k, v in (update or {}).items():
        if k != "messages":
            state[k] = v


def record_prompts(ticker: str, day: str, past_context: str = "", with_data: bool = True) -> list[dict]:
    """Each agent's prompt in graph order: ``{"step", "heading", "prompt", "data": [(call, output)]}``."""
    from tradingagents.agents.context import build_instrument_context, resolve_instrument_identity
    from tradingagents.graph.propagation import Propagator

    try:
        context = build_instrument_context(ticker, "stock", resolve_instrument_identity(ticker), day)
    except Exception:   # identity is a nicety: the agents fall back to the ticker alone
        context = ""
    state = Propagator().create_initial_state(ticker, day, past_context=past_context, instrument_context=context)
    llm = _Recorder()
    llm.seen = []
    nodes = _nodes(llm)
    out = []
    for step, heading, field in STEPS:
        llm.step = step
        before = len(llm.seen)
        try:
            update = nodes[step]({**state, "messages": list(state["messages"])})
        except Exception as exc:   # e.g. the sentiment feeds are down: say so, and go on
            out.append({"step": step, "heading": heading,
                        "prompt": f"<this agent's prompt could not be built: {type(exc).__name__}: {exc}>",
                        "data": []})
            if field:
                state[field] = placeholder(step)
            continue
        prompt = "\n\n".join(_as_text(m) for _, m in llm.seen[before:])
        _merge(state, update)
        if field:
            state[field] = placeholder(step)
        out.append({"step": step, "heading": heading, "prompt": _strip_app_additions(prompt),
                    "data": _tool_data(step, ticker, day) if with_data else []})
    return out


ANSWER_FORMAT = {
    "rating": "Buy | Overweight | Hold | Underweight | Sell (the portfolio manager's)",
    "executive_summary": "two to four sentences",
    "investment_thesis": "the evidence that decided it",
    "price_target": "number or null (the sell price where the profit is booked)",
    "stop_loss": "number or null (the exit price if it goes wrong)",
    "time_horizon": "e.g. 3-6 months, or null",
    "trader": {"action": "Buy | Hold | Sell", "entry_price": "number or null", "price_target": "number or null",
               "stop_loss": "number or null", "position_sizing": "text or null"},
}


DEEP_STEPS = ("research_manager", "portfolio_manager")   # the framework's deep-thinking tier; the rest are quick


def offered_tools(step: str) -> list[tuple[str, str]]:
    """(name, first line of its description) for each tool the analyst is offered in the graph."""
    import importlib

    module = {"market": "market_analyst", "news": "news_analyst", "fundamentals": "fundamentals_analyst"}.get(step)
    if not module:
        return []
    tools = importlib.import_module(f"tradingagents.agents.analysts.{module}").TOOLS
    return [(t.name, (t.description or "").strip().splitlines()[0] if t.description else "") for t in tools]


def build(ticker: str, day: str, past_context: str = "", with_data: bool = True) -> str:
    """The whole prompt as the API calls the framework makes: one request per agent, in graph order, with the
    messages it sends, the tools it offers, the tool calls an analyst makes and what they return, and for the
    structured steps the output schema the API binds. The chat answers every request in turn, each in the form the
    API returns it, so the pasted answer is saved as the same report files an API run writes."""
    steps = record_prompts(ticker, day, past_context, with_data)
    n = len(steps)
    structured = [i for i, s in enumerate(steps, 1) if s["step"] in STRUCTURED_STEPS]
    head = (f"Simulate the TradingAgents framework (TauricResearch) for {ticker} as of {day}. Below are the {n} API "
            f"requests it sends to its models, in order, exactly as sent: [SYSTEM] and [USER] messages, the tools "
            f"offered, and for the analysts the tool calls they make ([ASSISTANT -> tool call]) with what each tool "
            f"returned ([TOOL]). Answer each request as that agent's model would answer the API, in turn. Use only "
            f"the tool results given; do not browse. Where a later request quotes [the ... you wrote in step N], use "
            f"your answer to request N.\n\n"
            f"Answer in exactly the form the API returns:\n"
            f"- Requests {', '.join(map(str, structured))} are made with STRUCTURED OUTPUT: the API binds the "
            f"[OUTPUT SCHEMA] shown under each one. Answer each of them with one JSON object in a ```json fence holding "
            f"exactly that schema's fields (enum values spelled as listed, numbers as plain numbers, null where a "
            f"field is optional and you have no value), and nothing else.\n"
            f"- Every other request is answered in Markdown, as the message content the agent returns: the report or "
            f"argument itself, with no preamble or sign-off.\n"
            f"- Put each answer under its own heading line '## Response N: <the request's title>', in order, and "
            f"write nothing after Response {n}.")
    parts = [head]
    for i, s in enumerate(steps, 1):
        tier = "deep" if s["step"] in DEEP_STEPS else "quick"
        tools = offered_tools(s["step"])
        kind = "structured output" if s["step"] in STRUCTURED_STEPS else "free text"
        block = (f"\n\n{'=' * 70}\nAPI request {i} of {n}: {s['heading']}  (model tier: {tier}"
                 f"{'; tools offered: ' + ', '.join(name for name, _ in tools) if tools else '; no tools'}; {kind})"
                 f"\n{'=' * 70}")
        if tools:
            block += "\n[TOOLS]\n" + "\n".join(f"- {name}: {desc}" for name, desc in tools)
        block += f"\n\n{s['prompt']}"
        for call, result in s["data"]:
            block += f"\n\n[ASSISTANT -> tool call] {call}\n[TOOL {call.split('(')[0]}]\n{result}"
        if s["step"] in STRUCTURED_STEPS:
            block += f"\n\n[OUTPUT SCHEMA] (structured output: answer with one JSON object with these fields)\n" \
                     f"{schema_block(s['step'])}"
            block += f"\n\n[ASSISTANT] <- your Response {i}: the {s['heading']}, as one ```json object"
        else:
            block += f"\n\n[ASSISTANT] <- your Response {i}: the {s['heading']}"
        parts.append(block)
    parts.append(f"\n\n{'=' * 70}\nNow write Responses 1 to {n}, each under '## Response N: <title>'.")
    return "".join(parts)


def trader_line(d: dict[str, Any]) -> str:
    """The trader's proposal from an answer, one Markdown line; empty without one."""
    t = d.get("trader") if isinstance(d.get("trader"), dict) else {}
    if not t:
        return ""
    bits = [f"{k.replace('_', ' ')} {t[k]}" for k in ("action", "entry_price", "stop_loss", "position_sizing")
            if t.get(k) not in (None, "", "null")]
    return f"**Trader Proposal**: {'; '.join(bits)}\n" if bits else ""
