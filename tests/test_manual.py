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


def test_the_prompt_carries_raw_data_only(monkeypatch):
    from tradingagents.agents import manual

    monkeypatch.setattr("tradingagents.dataflows.router.route_to_vendor", lambda *a, **k: "data row 1.23")
    monkeypatch.setattr("tradingagents.dataflows.vendors.yahoo.snapshot.build_verified_market_snapshot",
                        lambda *a, **k: "snapshot row")
    p = manual.build_prompt("TPR", "2026-10-23", "Tapestry, Inc.")
    assert "PRICES" in p and "snapshot row" in p and '"rating"' in p
    for leak in ("verdict", "model rank", "already holds", "does not hold", "APP'S OWN"):
        assert leak.lower() not in p.lower()
