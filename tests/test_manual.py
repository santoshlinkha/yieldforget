"""The manual route: the answer's rating is read back from its JSON block."""

import pytest

from tradingagents.agents.manual import decision_markdown, parse_answer

pytestmark = pytest.mark.unit


def test_the_last_json_block_with_a_rating_wins():
    text = 'Draft: ```json\n{"rating": "Sell"}\n```\n... final:\n```json\n{"rating": "overweight", "executive_summary": "Scale in.", "investment_thesis": "Trend intact.", "price_target": 420, "time_horizon": "3-6 months"}\n```'
    d = parse_answer(text)
    assert d["rating"] == "Overweight" and d["price_target"] == 420
    assert "**Rating**: Overweight" in decision_markdown(d) and "**Price Target**: 420" in decision_markdown(d)


def test_a_plain_rating_line_is_a_fallback_and_nonsense_is_refused():
    assert parse_answer("Portfolio manager. Final rating: Hold, because the trend is flat.")["rating"] == "Hold"
    with pytest.raises(ValueError, match="no portfolio manager decision"):
        parse_answer('```json\n{"rating": "Strong maybe"}\n```')


def offline(monkeypatch):
    """The upstream prompt without the network: canned tool data, sentiment feeds and identity."""
    from tradingagents.agents import upstream_prompt
    from tradingagents.agents.analysts import sentiment_analyst

    monkeypatch.setattr(upstream_prompt, "_tool_data", lambda step, t, d: [(f"get_{step}_data()", f"{step.upper()}-DATA")])
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages", lambda *a, **k: "STOCKTWITS-POSTS")
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts", lambda *a, **k: "REDDIT-POSTS")
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a, **k: "NEWS-HEADLINES")
    monkeypatch.setattr("tradingagents.agents.context.resolve_instrument_identity", lambda t: {})


def test_the_prompt_is_the_agents_own_in_graph_order(monkeypatch):
    from tradingagents.agents import manual, upstream_prompt
    from tradingagents.agents.entry_plan import RISK_INSTRUCTION, TRADER_INSTRUCTION

    offline(monkeypatch)
    p = manual.build_prompt("TPR", "2026-10-23")
    order = [p.index(f"API request {i} of 12: {h}") for i, (_, h, _) in enumerate(upstream_prompt.STEPS, 1)]
    assert order == sorted(order)
    for text in ("You are a Bull Analyst advocating", "As the Research Manager", "As the Aggressive Risk Analyst",
                 "As the Portfolio Manager", "MARKET-DATA", "FUNDAMENTALS-DATA", "STOCKTWITS-POSTS"):
        assert text in p
    # Later agents read what the chat wrote for the earlier steps.
    assert upstream_prompt.placeholder("market") in p and upstream_prompt.placeholder("bull") in p
    for extra in (TRADER_INSTRUCTION, RISK_INSTRUCTION):   # the app's additions stay out
        assert extra.strip() not in p
    for leak in ("verdict", "model rank", "already holds"):
        assert leak not in p.lower()
    # The structured steps carry the schema the API binds; the answer is given in that form.
    assert p.count("[OUTPUT SCHEMA] (structured output") == 4 and '"rating" (one of Buy | Overweight | Hold | Underweight | Sell' in p
    assert '"overall_band"' in p and '"strategic_actions"' in p and '"stop_loss"' in p
    assert '"entry_plan"' not in p                     # the app's own fields stay out, like its prompt text


def test_the_trader_proposal_is_kept_with_the_decision():
    d = parse_answer('```json\n{"rating": "Buy", "trader": {"action": "Buy", "entry_price": 101.5, "stop_loss": 95}}\n```')
    assert "**Trader Proposal**: action Buy; entry price 101.5; stop loss 95" in decision_markdown(d)


def test_the_prompt_reads_as_the_frameworks_api_calls(monkeypatch):
    from tradingagents.agents import manual

    offline(monkeypatch)
    p = manual.build_prompt("TPR", "2026-10-23")
    assert "API request 1 of 12: Market Analyst report  (model tier: quick; tools offered: get_stock_data" in p
    assert "API request 7 of 12: Research Manager investment plan  (model tier: deep; no tools; structured output)" in p
    assert "API request 5 of 12: Bull Researcher argument  (model tier: quick; no tools; free text)" in p
    assert "[ASSISTANT -> tool call] get_market_data()\n[TOOL get_market_data]\nMARKET-DATA" in p
    assert "[TOOLS]\n- get_stock_data:" in p and "Now write Responses 1 to 12" in p



ANSWER = """Some preamble the chat app adds.

## Response 1: Market Analyst report
MARKET REPORT TEXT

## Response 2: Sentiment Analyst report
```json
{"overall_band": "Mildly Bullish", "overall_score": 6.1, "confidence": "medium", "narrative": "SENTIMENT NARRATIVE"}
```

## Response 3: News Analyst report
NEWS REPORT

## Response 4: Fundamentals Analyst report
FUNDAMENTALS REPORT

## Response 5: Bull Researcher argument
BULL CASE

## Response 6: Bear Researcher argument
BEAR CASE

## Response 7: Research Manager investment plan
```json
{"recommendation": "Overweight", "rationale": "WHY", "strategic_actions": "HOW"}
```

## Response 8: Trader proposal
```json
{"action": "Buy", "reasoning": "TRADE CASE", "entry_price": 570.5, "stop_loss": 520, "position_sizing": "3% of portfolio"}
```

## Response 9: Aggressive Risk Analyst argument
AGGRESSIVE

## Response 10: Conservative Risk Analyst argument
CONSERVATIVE

## Response 11: Neutral Risk Analyst argument
NEUTRAL

## Response 12: Portfolio Manager final decision
```json
{"rating": "Overweight", "executive_summary": "SUMMARY", "investment_thesis": "THESIS", "price_target": 640, "time_horizon": "3-6 months"}
```
"""


def test_a_per_step_answer_becomes_the_api_runs_final_state():
    from tradingagents.agents import manual
    from tradingagents.agents.schemas import PortfolioDecision, TraderProposal, render_pm_decision, render_trader_proposal

    state, meta = manual.final_state(ANSWER, "AMD", "2026-10-28")
    assert meta["notes"] == {} and len(meta["steps"]) == 12
    assert state["market_report"] == "MARKET REPORT TEXT"
    assert state["sentiment_report"].startswith("**Overall Sentiment:** **Mildly Bullish** (Score: 6.1/10)")
    assert state["investment_debate_state"]["bull_history"] == "\nBull Analyst: BULL CASE"
    assert state["investment_plan"].startswith("**Recommendation**: Overweight")
    assert state["trader_investment_plan"] == render_trader_proposal(TraderProposal(
        action="Buy", reasoning="TRADE CASE", entry_price=570.5, stop_loss=520, position_sizing="3% of portfolio"))
    assert state["final_trade_decision"] == render_pm_decision(PortfolioDecision(
        rating="Overweight", executive_summary="SUMMARY", investment_thesis="THESIS", price_target=640,
        time_horizon="3-6 months"))
    assert state["risk_debate_state"]["neutral_history"] == "\nNeutral Analyst: NEUTRAL"
    d = manual.decision_from_steps(meta)
    assert d["rating"] == "Overweight" and d["trader"] == {"action": "Buy", "entry_price": 570.5, "stop_loss": 520.0,
                                                          "position_sizing": "3% of portfolio"}


def test_a_structured_answer_outside_its_schema_is_kept_and_flagged():
    from tradingagents.agents import manual

    bad = ANSWER.replace('"overall_band": "Mildly Bullish"', '"overall_band": "Sunny"')
    state, meta = manual.final_state(bad, "AMD", "2026-10-28")
    assert "sentiment" in meta["notes"] and '"Sunny"' in state["sentiment_report"]
    assert manual.final_state("no sections here", "AMD", "2026-10-28") is None


def test_chat_app_formatting_is_tolerated():
    import re

    from tradingagents.agents import manual

    # Bold headings with "of 12" and a dash, an upper-case fence label, bold enum values, a "$" price.
    gemini = re.sub(r"^## Response (\d+): (.*)$", r"**Response \1 of 12 — \2**", ANSWER, flags=re.M)
    gemini = gemini.replace("```json", "```JSON").replace('"rating": "Overweight"', '"rating": "**overweight**"')
    gemini = gemini.replace('"price_target": 640', '"price_target": "$640"')
    state, meta = manual.final_state(gemini, "AMD", "2026-10-28")
    assert len(meta["steps"]) == 12 and meta["notes"] == {}
    d = manual.decision_from_steps(meta)
    assert d["rating"] == "Overweight" and d["price_target"] == 640.0 and d["trader"]["action"] == "Buy"
    # No headings at all: the structured answers are still found by their fields.
    bare = re.sub(r"^## Response .*$", "", ANSWER, flags=re.M)
    state, meta = manual.final_state(bare, "AMD", "2026-10-28")
    assert manual.decision_from_steps(meta)["rating"] == "Overweight"
    assert state["trader_investment_plan"].startswith("**Action**: Buy")


def test_the_error_says_what_was_found():
    from tradingagents.agents import manual

    try:
        manual.parse_answer("## Response 1: Market\ntext\n\n## Response 2: Sentiment\nmore")
    except ValueError as exc:
        assert "Found responses for: market, sentiment" in str(exc)
    else:
        raise AssertionError("no error")
