"""The money settings and the trading budget: persisted, validated; they never move money or rewrite history."""

import json
from datetime import date

import pandas as pd
import pytest

from tradingagents.ledger import capital, income
from tradingagents.ledger.store import Store

pytestmark = pytest.mark.unit

DAY = date(2026, 10, 1)


def test_defaults_persist_across_restarts_and_validate(tmp_path):
    store = Store(tmp_path)
    assert capital.load(store) == capital.Capital(0.0, 29.0, 0.03, 0.02)
    capital.update(store, {"daily_sgov_usd": "30.005", "match_rate": 0.01})
    again = Store(tmp_path)
    assert capital.load(again) == capital.Capital(0.0, 30.0, 0.01, 0.02)
    for bad in ({"daily_sgov_usd": -1}, {"match_rate": 0.5}, {"daily_sgov_usd": "lots"}, {"nope": 1}):
        with pytest.raises(ValueError):
            capital.update(again, bad)


def test_every_change_is_in_the_history_and_snapshots_are_frozen(tmp_path):
    store = Store(tmp_path)
    before = capital.snapshot(store)
    record = json.dumps(before)                    # what a decision or experiment stores
    after = capital.update(store, {"daily_sgov_usd": 50})
    assert after["version"] > before["version"] and after["hash"] != before["hash"]
    assert json.loads(record)["daily_sgov_usd"] == 29.0      # the old record still says 29
    assert capital.history(store)[0]["key"] == "daily_sgov_usd"


def test_a_setting_change_moves_no_money(tmp_path):
    store = Store(tmp_path)
    store.ledger.contribute(500.0, DAY, settle_date=DAY)
    capital.update(store, {"daily_sgov_usd": 100})
    assert store.ledger.state(None, DAY).cash.total == 500.0


def test_the_budget_is_what_was_set_less_what_went_into_stocks_since(tmp_path):
    from datetime import date as d

    store = Store(tmp_path)
    assert income.budget(store, tmp_path, d.today())["available"] == 0.0           # nothing set yet
    capital.update(store, {"trading_budget_usd": 100})
    today = d.today()
    store.ledger.contribute(100.0, today, settle_date=today)
    store.ledger.fill("trading", "buy", "AAA", 2.0, 10.0, today, settle_date=today)
    store.ledger.fill("trading", "sell", "AAA", 1.0, 12.0, today, settle_date=today)
    b = income.budget(store, tmp_path, str(today))                                  # a date string works too
    assert b["set"] == 100.0 and b["traded"] == {"spent": 20.0, "returned": 12.0} and b["available"] == 92.0


def test_the_earlier_interest_budget_is_interest_and_match_less_what_went_into_stocks(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.ledger.record("opening_position", "trading", DAY, ticker="SGOV", quantity=100.0, source="manual")
    income.set_start(store, DAY)
    # SGOV paid $0.30 a share on 10-02 and 11-02; 100 shares were held the day before each.
    monkeypatch.setattr(income, "_dividends", lambda cache, t: [(date(2026, 9, 1), 0.31), (date(2026, 10, 2), 0.30),
                                                                 (date(2026, 11, 2), 0.30)])
    as_of = date(2026, 11, 3)
    b = income.interest_budget(store, tmp_path, as_of)
    assert b["interest"]["basis"] == "estimated" and b["interest"]["earned"] == 60.0       # the 09-01 one is before the start
    sessions = b["match"]["sessions"]
    assert b["match"]["earned"] == round(29.0 * sessions * 0.03, 2)
    store.ledger.contribute(100.0, DAY, settle_date=DAY)
    store.ledger.fill("trading", "buy", "AAA", 2.0, 10.0, date(2026, 10, 5), settle_date=date(2026, 10, 6))
    store.ledger.fill("trading", "sell", "AAA", 1.0, 12.0, date(2026, 10, 20), settle_date=date(2026, 10, 21))
    b = income.interest_budget(store, tmp_path, as_of)
    assert b["match"]["basis"] == "recorded" and b["match"]["earned"] == 3.0               # 3% of the recorded deposit
    assert b["traded"] == {"spent": 20.0, "returned": 12.0}
    assert b["available"] == round(60.0 + 3.0 - 20.0 + 12.0, 2)
    # A recorded SGOV dividend replaces the estimate.
    store.ledger.record("dividend", "trading", date(2026, 10, 2), ticker="SGOV", cash=29.5, source="manual")
    assert income.interest_budget(store, tmp_path, as_of)["interest"] == {
        "earned": 29.5, "basis": "recorded", "items": [{"date": "2026-10-02", "amount": 29.5, "basis": "recorded"}]}


def test_no_start_date_means_no_budget(tmp_path):
    assert income.budget(Store(tmp_path), tmp_path, DAY)["available"] == 0.0


def _closes(sgov=100.0, bill_now=4.0, bill_then=4.0, days=300):
    idx = pd.bdate_range(end="2026-10-30", periods=days)
    bill = pd.Series(bill_then, index=idx)
    bill.iloc[-60:] = bill_now
    return pd.DataFrame({"SGOV": sgov, "^IRX": bill}, index=idx)


@pytest.mark.parametrize("last_div, bill_now, status", [(0.33, 4.0, "ok"), (0.33, 3.2, "falling"), (0.15, 3.2, "dried")])
def test_interest_health_flags_falling_and_dried_up_yields(tmp_path, monkeypatch, last_div, bill_now, status):
    store = Store(tmp_path)
    store.ledger.record("opening_position", "trading", DAY, ticker="SGOV", quantity=100.0, source="manual")
    divs = [(date(2026, m, 1), 0.33) for m in range(1, 10)] + [(date(2026, 10, 1), last_div)]
    monkeypatch.setattr(income, "_dividends", lambda cache, t: divs)
    monkeypatch.setattr(income, "alternatives", lambda cache, as_of, st: {"pick": "SHY", "status": st})
    h = income.health(store, tmp_path, date(2026, 10, 30), closes=_closes(bill_now=bill_now))
    assert h["status"] == status
    assert round(h["latest_yield"], 4) == round(last_div * 12 / 100, 4)
    assert (h["alternatives"] is None) == (status == "ok")
    assert h["monthly_interest_now"] == round(100 * last_div, 2)


def test_retiring_investing_keeps_the_account_and_backs_up_first(tmp_path):
    store = Store(tmp_path)
    con = store.con
    con.execute("INSERT INTO sleeves VALUES ('rh-roth-ira', 'investing', '2026-10-01T00:00:00Z')")
    con.commit()
    con.execute("INSERT INTO events (event_uid, account_id, sleeve, kind, trade_date, settle_date, ticker, quantity,"
                " cash, transfer_id, source, recorded_at) VALUES ('a', 'rh-roth-ira', 'investing', 'opening_position',"
                " '2026-10-01', '2026-10-01', 'SGOV', 40, 0, NULL, 'migration', 'x'), ('b', 'rh-roth-ira', 'investing',"
                " 'transfer', '2026-10-02', '2026-10-02', 'SGOV', -10, 0, 't1', 'manual', 'x'), ('c', 'rh-roth-ira',"
                " 'trading', 'transfer', '2026-10-02', '2026-10-02', 'SGOV', 10, 0, 't1', 'manual', 'x')")
    con.commit()
    before = store.ledger.state(None, DAY.replace(day=5)).positions
    out = store.retire_investing(DAY.replace(day=5), backup_dir=tmp_path / "bk")
    assert (tmp_path / "bk").exists() and out["deleted_events"] >= 2
    assert store.ledger.state(None, DAY.replace(day=5)).positions == before == {"SGOV": 40.0}
    assert not con.execute("SELECT 1 FROM events WHERE sleeve = 'investing'").fetchone()
    assert store.get("global", "income_start") == "2026-10-05"
    assert store.retire_investing(DAY.replace(day=6)) is None                       # once
    with pytest.raises(Exception, match="append-only"):                             # the guards are back
        con.execute("DELETE FROM events")
