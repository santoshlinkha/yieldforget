"""Fixes from the 2026-10-25 audit validation: shadow idempotence, deadlines past the loaded prices."""

import pandas as pd
import pytest

from tradingagents.stocks.positions import POLICIES, Book

pytestmark = pytest.mark.unit


def test_a_lot_bought_on_the_latest_session_keeps_a_real_deadline():
    sessions = pd.bdate_range("2026-10-01", "2026-10-23")
    book = Book(sessions, cash=1000.0)
    lot = book.buy(sessions[-1], "AAA", 200.0, 100.0, POLICIES["barrier_20"], 0.02, 0.004)
    assert lot.deadline > sessions[-1] + pd.Timedelta(days=20)        # 20 sessions on the exchange calendar
    assert book.exits_due(sessions[-1], pd.Series({"AAA": 100.0})) == []
    mid = book.buy(sessions[5], "BBB", 200.0, 100.0, POLICIES["barrier_20"], 0.02, 0.004)
    assert mid.deadline > sessions[-1]                                  # not clamped to the last row
