"""Compact tool output: fewer tokens, same facts."""

import pytest

from tradingagents.agents.tools import compact_indicator, compact_prices

pytestmark = pytest.mark.unit

PRICES = """# Stock data for AAA from 2026-01-02 to 2026-01-07
# Total records: 3

Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
2026-01-05,100.123456,101.987654,99.5,100.75,1000,0.0,0.0
2026-01-06,100.75,102.0,100.0,101.5,1200,0.0,0.0
2026-01-07,101.5,103.333333,101.0,103.0,900,0.0,0.0"""


def test_prices_lose_decimals_and_empty_columns_and_old_rows():
    out = compact_prices(PRICES, max_rows=2)
    assert "Dividends" not in out and "Stock Splits" not in out
    assert "2026-01-05" not in out and "2026-01-07,101.5,103.33,101.0,103.0,900" in out
    assert "1 earlier left out" in out and out.startswith("# Stock data for AAA")


def test_a_paid_dividend_keeps_its_column():
    paid = PRICES.replace("2026-01-06,100.75,102.0,100.0,101.5,1200,0.0", "2026-01-06,100.75,102.0,100.0,101.5,1200,0.83")
    assert "Dividends" in compact_prices(paid, max_rows=10)


def test_indicators_lose_placeholder_days_and_long_decimals():
    text = "## rsi values\n\n2026-01-07: 52.631111\n2026-01-04: N/A: Not a trading day (weekend or holiday)\n2026-01-02: 48.5"
    assert compact_indicator(text) == "## rsi values\n\n2026-01-07: 52.63\n2026-01-02: 48.5"
