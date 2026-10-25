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


def test_the_prompt_carries_data_only(monkeypatch):
    import pandas as pd

    from tradingagents.agents import manual, packet

    idx = pd.bdate_range(end="2026-10-23", periods=260)
    close = pd.Series(range(100, 360), index=idx, dtype=float)
    inputs = {"ticker": "TPR", "cutoff": "2026-10-23", "ohlcv": None, "facts": None, "macro": {}, "news": [],
              "bars": pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Volume": 1e6}),
              "profile": {"quoteType": "EQUITY", "longName": "Tapestry, Inc."}, "profile_fetched": None}
    monkeypatch.setattr(packet, "load_inputs", lambda *a, **k: inputs)
    p = manual.build_prompt("TPR", "2026-10-23", cache_dir=None)
    assert "EVIDENCE PACKET" in p and '"PX1"' in p and '"rating"' in p and manual.PROMPT_VERSION in p
    for leak in ("verdict", "model rank", "already holds", "does not hold", "APP'S OWN"):
        assert leak.lower() not in p.lower()
