"""The position engine follows the execution contract: next-close entry, close-confirmed exits at the
next close, exact deadlines, no same-day exits, T+1 proceeds, dividends as cash, no invented fills."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks.positions import Book, Policy, benchmark, trades

pytestmark = pytest.mark.unit

DAYS = pd.bdate_range("2025-01-01", periods=40)


def _frames(path, lows=None, divs=None):
    px = pd.DataFrame({"S": path}, index=DAYS[: len(path)]).reindex(DAYS)
    low = pd.DataFrame({"S": lows if lows is not None else path}, index=DAYS[: len(path)]).reindex(DAYS)
    div = pd.DataFrame({"S": np.zeros(len(DAYS))}, index=DAYS)
    for day, amount in (divs or {}).items():
        div.loc[DAYS[day], "S"] = amount
    return px, low, px.copy(), div


def _one(path, policy, sigma=0.01, lows=None, divs=None, t=0, cost=0.0):
    px, low, close, div = _frames(path, lows, divs)
    d = pd.DataFrame({"date": [DAYS[t]], "ticker": ["S"], "sigma": [sigma]})
    return trades(px, low, close, div, d, policy, np.array([cost])).iloc[0]


def test_entry_is_the_next_close_and_a_fixed_exit_lands_on_the_deadline():
    path = np.arange(100.0, 140.0)
    r = _one(path, Policy("f", "fixed", 5))
    assert r["entry_date"] == DAYS[1] and r["exit_date"] == DAYS[6]        # t+1 and t+1+5
    assert r["reason"] == "deadline" and r["held_sessions"] == 5
    assert r["gross"] == pytest.approx(106 / 101 - 1)


def test_a_stop_is_confirmed_at_the_close_and_filled_at_the_next_close_through_a_gap():
    # Entry 100 at t+1; the close of t+3 is 95 (past a -3% stop); the next close gaps to 80.
    path = np.r_[100.0, 100.0, 99.0, 95.0, 80.0, np.full(35, 80.0)]
    p = Policy("b", "barrier", 10, profit_k=10, risk_k=1)       # risk level expm1(-sqrt(10)*0.01) = about -3.1%
    r = _one(path, p, sigma=0.01)
    assert r["reason"] == "stop" and r["exit_date"] == DAYS[4]
    assert r["exit_px"] == 80.0 and r["gross"] == pytest.approx(-0.2)    # no fill at the stop: the gap is lost
    assert r["gross"] < r["risk_level"]


def test_a_target_never_fills_at_an_intraday_high():
    path = np.r_[100.0, 100.0, 104.0, 103.0, np.full(36, 103.0)]
    p = Policy("b", "barrier", 10, profit_k=1, risk_k=10)        # target about +3.2%
    r = _one(path, p, sigma=0.01)
    assert r["reason"] == "target" and r["exit_px"] == 103.0          # the next close, not the 104 that triggered it


def test_a_trigger_on_the_last_sessions_exits_on_the_deadline_not_after():
    path = np.r_[100.0, np.full(5, 100.0), 90.0, np.full(33, 90.0)]   # the close of entry+5 falls through the stop
    r = _one(path, Policy("b", "barrier", 5, 10, 1), sigma=0.01)
    assert r["exit_date"] == DAYS[6] and r["held_sessions"] == 5


def test_dividends_are_cash_and_reconcile_with_the_total_return_series():
    path = np.full(40, 50.0)
    r = _one(path, Policy("f", "fixed", 5), divs={3: 1.0})
    assert r["dividends"] == 1.0 and r["gross"] == pytest.approx(1.0 / 50)
    adjusted = pd.Series(np.r_[np.full(3, 50.0 * (1 - 1 / 50)), np.full(37, 50.0)], index=DAYS)   # back-adjusted
    total = adjusted.iloc[6] / adjusted.iloc[1] - 1
    assert r["gross"] == pytest.approx(total, abs=1e-3)


def test_the_adverse_excursion_starts_after_entry_and_is_never_positive():
    path = np.r_[100.0, 90.0, np.full(38, 95.0)]
    lows = np.r_[100.0, 85.0, 92.0, np.full(37, 95.0)]
    r = _one(path, Policy("f", "fixed", 5), lows=lows)
    assert r["mae"] == pytest.approx(92 / 90 - 1 if 92 / 90 - 1 < 0 else 0.0)    # the entry day's own low is excluded
    up = _one(np.arange(100.0, 140.0), Policy("f", "fixed", 5))
    assert up["mae"] == 0.0


def test_a_path_that_runs_out_is_incomplete_not_guessed():
    path = np.r_[100.0, 100.0, 101.0, np.full(37, np.nan)]          # delisted two sessions after entry
    r = _one(path, Policy("f", "fixed", 5))
    assert r["reason"] == "incomplete" and np.isnan(r["net"])


def test_costs_come_off_net_only():
    r = _one(np.arange(100.0, 140.0), Policy("f", "fixed", 5), cost=0.004)
    assert r["net"] == pytest.approx(r["gross"] - 0.004)


def test_benchmark_uses_the_same_entry_and_exit_closes():
    path = np.arange(100.0, 140.0)
    r = _one(path, Policy("f", "fixed", 5))
    spy = pd.Series(np.arange(200.0, 240.0), index=DAYS)
    assert benchmark(spy, pd.DataFrame([r]))[0] == pytest.approx(206 / 201 - 1)


def test_the_book_refuses_same_day_sales_and_settles_next_session():
    book = Book(DAYS, cash=1000.0)
    lot = book.buy(DAYS[1], "S", 500.0, 100.0, Policy("f", "fixed", 5), 0.01, 0.0)
    with pytest.raises(ValueError, match="day trade"):
        book.sell(DAYS[1], lot, 101.0, "test")
    assert book.buy(DAYS[1], "T", 600.0, 10.0, Policy("f", "fixed", 5), 0.01, 0.0) is None   # only settled cash
    book.sell(DAYS[2], lot, 110.0, "test")
    assert book.cash == 500.0 and book.settling == {DAYS[3]: 550.0}
    book.settle(DAYS[3])
    assert book.cash == 1050.0
