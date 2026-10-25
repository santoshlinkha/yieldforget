"""Write docs/AGENT_API_IO.txt: what an agent run sends to the AI model and what it expects back.

Built from a recorded run (the request log next to its reports), so every prompt and reply is real:

    python docs/build_agent_io.py MSFT 2026-10-23

Within one agent, each request resends the whole conversation; the file prints an agent's first
request in full and, for its later requests, only the messages added since the one before.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from tradingagents.agents import schemas
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.request_log import read

OUT = Path(__file__).with_name("AGENT_API_IO.txt")
RULE, THIN = "=" * 100, "-" * 100

# Who speaks in what order, what each is given, and what it must return.
STAGES = [
    ("Analysts (market / technical, news & economy, fundamentals, sentiment)",
     "Their instructions, the ticker and date, and the tools they may call.",
     "Either tool calls (requests for data: prices, indicators, news, statements) or, once they have "
     "the data, a long Markdown report ending in a summary table. Free text."),
    ("Bull researcher, bear researcher",
     "Their role, every analyst report, and the debate so far.",
     "A written argument for (bull) or against (bear) buying. Free text."),
    ("Research manager",
     "The whole bull/bear debate and past lessons.",
     "An investment plan in the ResearchPlan format below (JSON when the model supports it)."),
    ("Trader",
     "The investment plan and the analyst reports.",
     "A trade proposal in the TraderProposal format below."),
    ("Aggressive, conservative and neutral risk analysts",
     "The trader's proposal, the reports and the risk debate so far.",
     "A written argument from their risk stance. Free text."),
    ("Portfolio manager",
     "The risk debate, the trader's proposal and past lessons.",
     "The final decision in the PortfolioDecision format below. Its `rating` is what the app shows: "
     "Buy / Overweight -> BUY, Hold -> HOLD, Underweight / Sell -> SELL (AVOID or WATCH when not held)."),
]
SCHEMAS = [("ResearchPlan (research manager)", schemas.ResearchPlan),
           ("TraderProposal (trader)", schemas.TraderProposal),
           ("PortfolioDecision (portfolio manager: the final answer)", schemas.PortfolioDecision),
           ("SentimentReport (sentiment analyst, when structured)", schemas.SentimentReport)]


def _fmt(m: dict) -> str:
    head = f"[{m.get('role', '?').upper()}{' / ' + m['name'] if m.get('name') else ''}]"
    body = m.get("content") or ""
    if body == "[]":   # an empty reply in older logs: the turn was only tool calls
        body = ""
    calls = "".join(f"\n  -> tool call: {c['name']}({json.dumps(c.get('args'), default=str)})"
                    for c in m.get("tool_calls") or [])
    return f"{head}\n{body}{calls}".rstrip()


def _schema(model) -> str:
    props = model.model_json_schema().get("properties", {})
    lines = []
    for name, p in props.items():
        kind = p.get("type") or " | ".join(x.get("type", "?") for x in p.get("anyOf", [])) or "enum"
        if "$ref" in p or ("allOf" in p):
            kind = "one of the allowed values"
        desc = " ".join((p.get("description") or "").split())
        lines.append(f"  {name} ({kind}): " + textwrap.shorten(desc, 220, placeholder="..."))
    return "\n".join(lines)


def build(ticker: str, day: str) -> Path:
    folder = Path(DEFAULT_CONFIG["results_dir"]) / ticker.upper() / day / "reports"
    calls = read(folder)
    if not calls:
        raise SystemExit(f"no recorded requests in {folder}; run the agents for {ticker} on {day} first")
    tin = sum(c.get("input_tokens") or 0 for c in calls)
    tout = sum(c.get("output_tokens") or 0 for c in calls)
    models = sorted({c.get("model") or "?" for c in calls})
    ratings = [schemas.PortfolioRating(r).value for r in schemas.PortfolioRating]
    out = [RULE, f"WHAT THE AGENTS SEND TO THE AI MODEL, AND WHAT THEY EXPECT BACK",
           f"Recorded run: {ticker.upper()} as of {day} · {len(calls)} requests · models: {', '.join(models)}",
           f"Tokens: {tin:,} sent, {tout:,} received", RULE, "",
           "HOW TO READ THIS",
           "Each request to the API is a list of messages, plus the tools the agent may call:",
           "  [SYSTEM]  the agent's standing instructions (role, date, ticker, rules)",
           "  [HUMAN]   the task, or for the debaters and managers, everything they are given to read",
           "  [AI]      the model's own earlier turn (often a tool call)",
           "  [TOOL]    data a tool returned (prices, indicators, news, statements)",
           "The reply is either tool calls (the agent asks for data and gets another turn) or the agent's text.",
           "No API key or account data is ever part of a request.", "",
           "THE ORDER OF THE AGENTS", THIN]
    for i, (who, given, back) in enumerate(STAGES, 1):
        out += [f"{i}. {who}", f"   Sent:     {given}", f"   Expected: {back}"]
    out += ["", "EXPECTED OUTPUT FORMATS", THIN, f"Ratings: {', '.join(ratings)}", ""]
    for title, model in SCHEMAS:
        out += [title, _schema(model), ""]
    out += ["", "EVERY REQUEST OF THE RECORDED RUN", RULE]
    previous: dict[str, list] = {}
    for c in calls:
        agent = c.get("agent") or "agent"
        sent = c.get("sent") or []
        before = previous.get(agent, [])
        same = before and sent[:len(before)] == before
        out += ["", RULE, f"REQUEST #{c['n']} · {agent} · {c.get('model')} · {c.get('input_tokens')} tokens in / "
                f"{c.get('output_tokens')} out · {c.get('seconds')} s", RULE]
        if c.get("tools_offered"):
            out.append("Tools offered: " + ", ".join(c["tools_offered"]))
        if same:
            out += [f"EXPECTED INPUT: the same {len(before)} messages as this agent's previous request, then its reply and the new data:", THIN]
            out += [_fmt(m) + "\n" for m in sent[len(before):]]
        else:
            out += ["EXPECTED INPUT (sent to the API):", THIN] + [_fmt(m) + "\n" for m in sent]
        out += ["EXPECTED OUTPUT (came back):", THIN,
                f"ERROR: {c['error']}" if c.get("error") else _fmt(c.get("received") or {})]
        previous[agent] = sent
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return OUT


if __name__ == "__main__":
    path = build(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("MSFT", "2026-10-23")))
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
