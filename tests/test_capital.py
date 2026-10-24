"""The three capital amounts: persisted, migrated, validated; they never move money or rewrite history."""

import json
from datetime import date

import pytest

from tradingagents.ledger import capital
from tradingagents.ledger.store import Store

pytestmark = pytest.mark.unit

DAY = date(2026, 10, 1)


def test_defaults_persist_across_restarts_and_validate(tmp_path):
    store = Store(tmp_path)
    assert capital.load(store) == capital.Capital(1000.0, 4800.0, 27.61)
    capital.update(store, {"trading_capital_usd": 1500, "daily_investment_usd": "30.005"})
    again = Store(tmp_path)                                   # a restart
    assert capital.load(again) == capital.Capital(1500.0, 4800.0, 30.0)
    for bad in ({"trading_capital_usd": -1}, {"daily_investment_usd": "lots"}, {"investing_capital_usd": float("nan")},
                {"weekly_usd": 5}):
        with pytest.raises(ValueError):
            capital.update(again, bad)
    with pytest.raises(ValueError):
        again.set("capital", "trading_capital_usd", -5)       # any writer is checked, not only the API


def test_earlier_saved_values_are_kept_not_overwritten(tmp_path):
    store = Store(tmp_path)
    # As saved by an earlier version, before the capital settings existed.
    store.con.execute("INSERT INTO settings VALUES ('trading', 'initial_allocation', '1250.0', 'x')")
    store.con.execute("INSERT INTO settings VALUES ('investing', 'daily_contribution', '12.5', 'x')")
    cap = capital.load(store)
    assert cap.trading_capital_usd == 1250.0 and cap.daily_investment_usd == 12.5 and cap.investing_capital_usd == 4800.0
    assert store.get("trading", "initial_allocation") == 1250.0          # the old key reads the new setting
    store.set("investing", "daily_contribution", 20.0)                    # and writes it
    assert capital.load(store).daily_investment_usd == 20.0


def test_every_change_is_in_the_history_and_snapshots_are_frozen(tmp_path):
    store = Store(tmp_path)
    before = capital.snapshot(store)
    record = json.dumps(before)                    # what a recommendation or experiment stores
    after = capital.update(store, {"trading_capital_usd": 2000})
    assert after["version"] > before["version"] and after["hash"] != before["hash"]
    assert json.loads(record)["trading_capital_usd"] == 1000.0      # the old record still says 1,000
    assert capital.history(store)[0]["key"] == "trading_capital_usd"


def test_changing_a_target_moves_no_money_and_keeps_trading_pnl(tmp_path):
    store = Store(tmp_path)
    store.ledger.contribute(5000.0, DAY, sleeve="investing", settle_date=DAY)
    store.ledger.transfer("investing", "trading", DAY, cash=1000.0)
    store.ledger.fill("trading", "buy", "AAA", 10.0, 50.0, DAY, settle_date=DAY)
    prices = {"AAA": 70.0}                                   # Trading is up $200
    before = {s: store.ledger.state(s, DAY, prices).equity for s in ("trading", "investing")}
    capital.update(store, {"trading_capital_usd": 1500})
    after = {s: store.ledger.state(s, DAY, prices).equity for s in ("trading", "investing")}
    assert before == after                                   # a setting change is not a transfer
    p = capital.reallocation(store, DAY, prices)
    assert p["sleeves"]["trading"]["funded"] == 1000.0 and p["sleeves"]["trading"]["equity"] == 1200.0
    assert p["transfers"] == [{"from": "investing", "to": "trading", "cash": 500.0}]   # target - funded, not - equity


def test_a_proposal_never_creates_money(tmp_path):
    store = Store(tmp_path)
    store.ledger.contribute(300.0, DAY, sleeve="investing", settle_date=DAY)
    p = capital.reallocation(store, DAY, {})
    assert p["transfers"] == [{"from": "investing", "to": "trading", "cash": 300.0}]   # only what is settled
    assert any("needs a deposit" in n for n in p["notes"])


def test_trading_capital_is_not_replenished_each_day(tmp_path):
    store = Store(tmp_path)
    store.ledger.contribute(5000.0, DAY, sleeve="investing", settle_date=DAY)
    store.ledger.transfer("investing", "trading", DAY, cash=1000.0)
    store.ledger.fill("trading", "buy", "AAA", 10.0, 50.0, DAY, settle_date=DAY)
    store.ledger.fill("trading", "sell", "AAA", 10.0, 30.0, date(2026, 10, 2), settle_date=date(2026, 10, 5))
    later = date(2026, 10, 9)
    p = capital.reallocation(store, later, {"AAA": 30.0})
    assert p["transfers"] == []                    # Trading lost $200; nothing refills it
    assert p["sleeves"]["trading"]["funded"] == 1000.0


def test_the_daily_plan_is_bounded_by_settled_cash_and_skips_holidays(tmp_path):
    store = Store(tmp_path)
    store.ledger.contribute(10.0, DAY, sleeve="investing", settle_date=DAY)
    plan = capital.daily_plan(store, DAY)
    assert plan["planned"] == 27.61 and plan["buy"] == 10.0 and plan["shortfall"] == 17.61
    assert any("not a deposit" in n for n in plan["notes"])
    holiday = capital.daily_plan(store, date(2026, 11, 26))          # Thanksgiving
    assert holiday["session"] == "2026-11-27"
    reserved = capital.daily_plan(store, DAY, reserved_today=8.0)
    assert reserved["buy"] == 2.0
