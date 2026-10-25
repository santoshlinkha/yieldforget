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
