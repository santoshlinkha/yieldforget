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
    with pytest.raises(ValueError, match="no rating"):
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
    assert '"trader"' in p and '"rating"' in p


def test_the_trader_proposal_is_kept_with_the_decision():
    d = parse_answer('```json\n{"rating": "Buy", "trader": {"action": "Buy", "entry_price": 101.5, "stop_loss": 95}}\n```')
    assert "**Trader Proposal**: action Buy; entry price 101.5; stop loss 95" in decision_markdown(d)


def test_the_prompt_reads_as_the_frameworks_api_calls(monkeypatch):
    from tradingagents.agents import manual

    offline(monkeypatch)
    p = manual.build_prompt("TPR", "2026-10-23")
    assert "API request 1 of 12: Market Analyst report  (model tier: quick; tools offered: get_stock_data" in p
    assert "API request 7 of 12: Research Manager investment plan  (model tier: deep; no tools)" in p
    assert "[ASSISTANT -> tool call] get_market_data()\n[TOOL get_market_data]\nMARKET-DATA" in p
    assert "[TOOLS]\n- get_stock_data:" in p and "Now write Responses 1 to 12" in p
