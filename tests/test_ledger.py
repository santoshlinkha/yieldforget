"""Durable records: sleeves, settlement, transfers, immutability, recommendations, CSV import, AI budgets.

Fixtures are labelled synthetic: made-up tickers and amounts, never your account's history.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import pytest
import pytz

from tradingagents.ledger import ai_gateway, imports, price_store, recommendations
from tradingagents.ledger.accounting import Ledger, LedgerError
from tradingagents.ledger.db import connect
from tradingagents.ledger.sessions import is_session, next_session, previous_session
from tradingagents.ledger.store import Store

pytestmark = pytest.mark.unit

MON, TUE, WED = "2026-10-19", "2026-10-20", "2026-10-21"


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(connect(tmp_path / "t.sqlite3"))
    led.ensure_account()
    return led


def test_sessions_follow_the_exchange_calendar():
    assert not is_session("2026-11-26") and not is_session("2026-07-03")   # Thanksgiving; July 4th observed
    assert is_session("2026-11-27") and not is_session("2026-10-24")
    assert next_session("2026-11-25").isoformat() == "2026-11-27"
    assert previous_session("2026-01-02").isoformat() == "2025-12-31"


def test_sale_proceeds_settle_next_session_and_buys_need_settled_cash(ledger):
    ledger.contribute(100.0, MON)
    ledger.fill("investing", "buy", "AAA", 1, 80.0, MON)
    with pytest.raises(LedgerError, match="settled"):
        ledger.fill("investing", "buy", "AAA", 1, 80.0, MON)
    ledger.fill("investing", "sell", "AAA", 1, 90.0, MON)
    s = ledger.state("investing", MON)
    assert s.cash.settled == pytest.approx(20.0) and s.cash.unsettled == pytest.approx(90.0)
    assert ledger.state("investing", TUE).cash.settled == pytest.approx(110.0)


def test_transfers_move_value_between_sleeves_without_changing_the_account(ledger):
    ledger.contribute(500.0, MON)
    ledger.fill("investing", "buy", "CASHLIKE", 4, 100.0, MON)
    before = ledger.state(None, MON, {"CASHLIKE": 100.0})
    ledger.transfer("investing", "trading", MON, ticker="CASHLIKE", quantity=2, price=100.0)
    ledger.transfer("investing", "trading", MON, cash=50.0)
    after = ledger.state(None, MON, {"CASHLIKE": 100.0})
    assert after.equity == pytest.approx(before.equity) and after.positions == before.positions
    trading = ledger.state("trading", MON, {"CASHLIKE": 100.0})
    assert trading.equity == pytest.approx(250.0)
    # For the sleeve the transfer is a capital flow; for the account it is nothing.
    assert ledger.flows("trading", "2026-10-01", MON) == pytest.approx(250.0)
    assert ledger.flows(None, "2026-10-01", MON) == pytest.approx(500.0)
    with pytest.raises(LedgerError):
        ledger.transfer("investing", "trading", MON, ticker="CASHLIKE", quantity=5, price=100.0)


def test_unsettled_cash_cannot_fund_the_trading_sleeve(ledger):
    ledger.contribute(10.0, MON)
    ledger.fill("investing", "buy", "AAA", 1, 10.0, MON)
    ledger.fill("investing", "sell", "AAA", 1, 1000.0, MON)
    with pytest.raises(LedgerError, match="unsettled"):
        ledger.transfer("investing", "trading", MON, cash=1000.0)
    ledger.transfer("investing", "trading", TUE, cash=1000.0)


def test_investment_pnl_excludes_contributions(ledger):
    ledger.contribute(1000.0, MON)
    ledger.fill("investing", "buy", "AAA", 10, 100.0, MON)
    ledger.contribute(27.61, TUE)
    end = ledger.state("investing", TUE, {"AAA": 110.0})
    start_equity = 0.0
    pnl = end.equity - start_equity - ledger.flows("investing", "2026-10-01", TUE)
    assert pnl == pytest.approx(100.0)


def test_events_are_deduplicated_and_survive_a_restart(tmp_path):
    path = tmp_path / "t.sqlite3"
    led = Ledger(connect(path))
    led.ensure_account()
    first, new = led.record("contribution", "investing", MON, cash=50.0, uid="broker-123")
    again, new_again = led.record("contribution", "investing", MON, cash=50.0, uid="broker-123")
    assert new and not new_again and first == again
    led.con.close()
    reopened = Ledger(connect(path))
    assert reopened.state("investing", MON).cash.settled == pytest.approx(50.0)


def test_recorded_events_cannot_be_edited_or_deleted_only_voided(ledger):
    event, _ = ledger.record("contribution", "investing", MON, cash=50.0)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.con.execute("UPDATE events SET cash = 5 WHERE id = ?", (event,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.con.execute("DELETE FROM events WHERE id = ?", (event,))
    ledger.void(event, "entered twice")
    assert ledger.state("investing", MON).cash.settled == 0
    assert len(ledger.events("investing", include_void=True)) == 1


def test_reservations_reduce_what_can_be_spent(ledger):
    ledger.contribute(100.0, MON)
    rid = ledger.reserve("investing", 60.0, "collateral", as_of=MON)
    assert ledger.state("investing", MON).cash.available == pytest.approx(40.0)
    with pytest.raises(LedgerError):
        ledger.fill("investing", "buy", "AAA", 1, 50.0, MON)
    ledger.release(rid)
    ledger.fill("investing", "buy", "AAA", 1, 50.0, MON)


def test_reconcile_reports_differences(ledger):
    ledger.contribute(100.0, MON)
    ledger.fill("investing", "buy", "AAA", 1, 50.0, MON)
    assert ledger.reconcile({"AAA": 1.0}, 50.0, MON) == []
    problems = ledger.reconcile({"AAA": 2.0}, 40.0, MON)
    assert any("AAA" in p for p in problems) and any("cash" in p for p in problems)


def test_legacy_files_are_kept_and_become_opening_positions(tmp_path):
    (tmp_path / "my_portfolio.json").write_text(json.dumps({"positions": [{"ticker": "aaa", "quantity": 2.5}]}))
    (tmp_path / "my_risk.json").write_text(json.dumps({"level": 5}))
    (tmp_path / "my_buys.json").write_text(json.dumps({"daily": {"AAA": 20}, "weekly": {"BBB": 5}}))
    store = Store(tmp_path)
    assert store.ledger.state("investing", "2100-01-01").positions == {"AAA": 2.5}
    assert store.ledger.state("trading", "2100-01-01").positions == {}
    assert store.get("investing", "risk_level") == 5 and store.get("trading", "risk_level") == 2
    assert store.get("investing", "daily_contribution") == 21.0
    assert json.loads(store.legacy("my_risk.json")) == {"level": 5}
    Store(tmp_path)   # opening again imports nothing twice
    assert len(store.ledger.events("investing")) == 1


def test_trading_assignment_is_only_a_preview_of_what_exists(tmp_path):
    (tmp_path / "my_portfolio.json").write_text(json.dumps({"positions": [{"ticker": "SGOV", "quantity": 5}]}))
    store = Store(tmp_path)
    preview = store.trading_assignment_preview(1000.0, {"SGOV": 100.0}, "2100-01-01")
    assert preview["steps"] == [{"ticker": "SGOV", "quantity": 5, "price": 100.0, "value": 500.0}]
    assert preview["short_by"] == 500.0
    assert store.ledger.state("trading", "2100-01-01").positions == {}


def test_sleeve_settings_are_separate_and_keep_history(tmp_path):
    store = Store(tmp_path)
    store.set("trading", "risk_level", 7)
    assert store.get("investing", "risk_level") == 2 and store.get("trading", "risk_level") == 7
    store.set("trading", "risk_level", 8)
    assert [r[0] for r in store.con.execute("SELECT value_json FROM settings_history WHERE scope='trading'")] == ["7", "8"]
    with pytest.raises(KeyError):
        store.set("trading", "leverage", 10)


def test_work_claims_stop_a_second_owner(tmp_path):
    store = Store(tmp_path)
    assert store.claim("tracker:2026-10-20", "proc-a")
    assert not store.claim("tracker:2026-10-20", "proc-b")
    store.release("tracker:2026-10-20", "proc-a")
    assert store.claim("tracker:2026-10-20", "proc-b")


def test_price_store_rebuilds_each_snapshot_exactly(ledger):
    days = pd.bdate_range("2026-01-01", periods=5)
    first = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=days)
    at1 = price_store.save(ledger.con, first.iloc[:3])
    at2 = price_store.save(ledger.con, first)                       # appends two days
    adjusted = first * 0.99                                        # a dividend rewrites history
    at3 = price_store.save(ledger.con, adjusted)
    assert price_store.frame_as_of(ledger.con, ["AAA"], at1)["AAA"].tolist() == [1.0, 2.0, 3.0]
    assert price_store.frame_as_of(ledger.con, ["AAA"], at2)["AAA"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert price_store.frame_as_of(ledger.con, ["AAA"], at3)["AAA"].tolist() == pytest.approx([0.99, 1.98, 2.97, 3.96, 4.95])
    kinds = [r[0] for r in ledger.con.execute("SELECT kind FROM price_chunks ORDER BY id")]
    assert kinds == ["base", "append", "base"]


@dataclass
class _Plan:
    rows: list
    cash: object
    total: float = 100.0
    current_stress: float = 0.1
    target_stress: float = 0.05
    expected_current: float = 0.05
    expected_target: float = 0.06
    stress_worst_case: tuple = (0.1, 0.05)
    stress_scenarios: tuple | None = None
    worst_scenario: tuple = (None, None)
    current_history_dd: tuple = (0.1, "x")
    target_history_dd: tuple = (0.05, "x")
    goal: object = None


@dataclass
class _Row:
    ticker: str
    target: float
    value: float = 0.0
    held: bool = True
    reason: str = "synthetic"


@dataclass
class _Suggestion:
    orders: list = field(default_factory=list)
    tomorrow: list = field(default_factory=list)
    settled_cash: float = 0.0


def _save(con, when):
    closes = pd.DataFrame({"AAA": [1.0, 2.0], "CASH": [1.0, 1.0]}, index=pd.bdate_range("2026-10-15", periods=2))
    plan = _Plan([_Row("AAA", 0.4, 40.0)], _Row("CASH", 0.6, 60.0))
    return recommendations.save(con, account_id="rh-roth-ira", sleeve="investing", plan=plan, suggestion=_Suggestion(),
                                settings={"loss_limit": 0.025}, closes=closes, universe=["AAA"],
                                info_cutoff="2026-10-16", market_data_at="2026-10-16T21:00:00Z",
                                published=NY.localize(when))


NY = pytz.timezone("America/New_York")


def test_recommendations_are_immutable_and_superseded_not_edited(ledger):
    first = _save(ledger.con, datetime(2026, 10, 19, 9, 0))
    second = _save(ledger.con, datetime(2026, 10, 19, 10, 0))
    assert first["session"] == second["session"] == MON and second["supersedes"] == first["id"]
    assert recommendations.status(ledger.con, first["id"]) == "superseded"
    assert recommendations.official(ledger.con, "rh-roth-ira", "investing", MON)["id"] == second["id"]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.con.execute("UPDATE recommendations SET targets_json = '{}' WHERE id = ?", (first["id"],))
    rec = recommendations.load(ledger.con, first["id"])
    assert rec["artifact"]["plan"]["rows"][0]["ticker"] == "AAA" and json.loads(rec["targets_json"])["AAA"] == 0.4
    recommendations.decide(ledger.con, second["id"], "accepted")
    assert ledger.state(None, WED).positions == {}   # accepting buys nothing


def test_a_recommendation_cannot_fill_before_its_data_or_its_publication():
    # Saved after the cutoff on Monday from Friday's close: for Tuesday.
    after = NY.localize(datetime(2026, 10, 19, 17, 0))
    assert recommendations.session_for(after) == TUE
    assert recommendations.earliest_execution("2026-10-16", after) == TUE
    # Saved Monday morning from Monday's own close is impossible to act on Monday.
    morning = NY.localize(datetime(2026, 10, 19, 9, 0))
    assert recommendations.earliest_execution(MON, morning) == TUE
    assert recommendations.earliest_execution("2026-10-16", morning) == MON


CSV = """"Activity Date","Process Date","Settle Date","Instrument","Description","Trans Code","Quantity","Price","Amount"
"10/19/2026","10/19/2026","10/19/2026","","ACH Deposit","ACH","","","$200.00"
"10/19/2026","10/19/2026","10/20/2026","AAA","Synthetic fund","Buy","2","$50.00","($100.00)"
"10/20/2026","10/20/2026","10/21/2026","AAA","Synthetic fund","Sell","1","$55.00","$54.98"
"10/20/2026","10/20/2026","10/20/2026","AAA","Cash Div","CDIV","","","$0.40"
"10/20/2026","10/20/2026","10/20/2026","","Something else","XYZ","","","$1.00"
"""


def test_csv_import_previews_dedupes_and_reconciles(ledger):
    rows = imports.preview(ledger, CSV)
    assert [r.action for r in rows] == ["contribution", "fill", "fill", "dividend", "skip"]
    assert rows[-1].reason and "XYZ" in rows[-1].reason
    result = imports.commit(ledger, rows)
    assert result["written"] == 4
    s = ledger.state("investing", WED)
    assert s.positions == {"AAA": 1.0} and s.cash.total == pytest.approx(200 - 100 + 54.98 + 0.40)
    again = imports.preview(ledger, CSV)
    assert all(r.duplicate for r in again if r.action != "skip")
    assert imports.commit(ledger, again)["written"] == 0


def _gateway(ledger, **budget):
    policy = ai_gateway.Policy(budget=ai_gateway.Budget(**budget),
                               prices={"openai/test-model": ai_gateway.Price(input=1.0, output=4.0)})
    return ai_gateway.Gateway(ledger.con, policy)


def _job(gw, inputs, calls, allow=True, est=0.01):
    def work(cb):
        calls.append(1)
        gw.record(cb.key, "brief", "openai", "test-model", input_tokens=1000, output_tokens=500)
        return {"answer": inputs["x"]}
    return gw.run(task="brief", provider="openai", model="test-model", schema_version="v1", info_cutoff="2026-10-16",
                  inputs=inputs, work=work, est_usd=est, est_requests=1, allow_paid=allow)


def test_ai_results_are_cached_by_what_matters(ledger, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_NO_PAID_AI", raising=False)
    gw, calls = _gateway(ledger), []
    assert _job(gw, {"x": 1}, calls) == ({"answer": 1}, "paid")
    assert _job(gw, {"x": 1}, calls) == ({"answer": 1}, "cache")
    assert _job(gw, {"x": 2}, calls)[1] == "paid"
    assert len(calls) == 2
    cost = ledger.con.execute("SELECT SUM(cost_usd) FROM ai_usage").fetchone()[0]
    assert cost == pytest.approx(2 * (1000 * 1.0 + 500 * 4.0) / 1e6)


def test_paid_ai_needs_an_explicit_request_a_price_and_budget(ledger, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_NO_PAID_AI", raising=False)
    gw, calls = _gateway(ledger, daily_usd=0.004), []
    with pytest.raises(ai_gateway.AIBlocked, match="not requested"):
        _job(gw, {"x": 1}, calls, allow=False)
    _job(gw, {"x": 1}, calls, est=0.003)           # spends 0.003
    with pytest.raises(ai_gateway.AIBlocked, match="budget"):
        _job(gw, {"x": 2}, calls, est=0.003)
    unpriced = ai_gateway.Gateway(ledger.con, ai_gateway.Policy())
    with pytest.raises(ai_gateway.AIBlocked, match="no price"):
        _job(unpriced, {"x": 3}, calls)
    monkeypatch.setenv("TRADINGAGENTS_NO_PAID_AI", "1")
    with pytest.raises(ai_gateway.AIBlocked, match="switched off"):
        _job(gw, {"x": 4}, calls)
    assert len(calls) == 1


def test_a_running_job_is_not_paid_for_twice_and_reservations_count(ledger, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_NO_PAID_AI", raising=False)
    gw = _gateway(ledger, daily_usd=0.05)
    other = ai_gateway.Gateway(ledger.con, gw.policy, owner="other-process")

    def work(cb):
        with pytest.raises(ai_gateway.AIInProgress):
            _job(other, {"x": 1}, [])
        with pytest.raises(ai_gateway.AIBlocked, match="budget"):   # 0.04 reserved by this run
            _job(other, {"x": 9}, [], est=0.02)
        return {"answer": 1}

    gw.run(task="brief", provider="openai", model="test-model", schema_version="v1", info_cutoff="2026-10-16",
           inputs={"x": 1}, work=work, est_usd=0.04, est_requests=1, allow_paid=True)
    assert gw.reserved() == (0.0, 0)


def test_funding_trading_twice_proposes_only_what_is_missing(tmp_path):
    (tmp_path / "my_portfolio.json").write_text(json.dumps({"positions": [{"ticker": "SGOV", "quantity": 30}]}))
    store = Store(tmp_path)
    first = store.trading_assignment_preview(1000.0, {"SGOV": 100.0}, "2100-01-01")
    step = first["steps"][0]
    store.ledger.transfer("investing", "trading", "2100-01-01", ticker="SGOV", quantity=step["quantity"], price=100.0)
    again = store.trading_assignment_preview(1000.0, {"SGOV": 100.0}, "2100-01-01")
    assert again["steps"] == [] and again["trading_already"] == 1000.0 and again["amount"] == 0.0
