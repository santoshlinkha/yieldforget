"""The books under concurrent writers, retried requests and cash edits.

Fixtures are synthetic: made-up tickers (SYNTH, AAA) and amounts in temporary databases.
"""

import threading
import time

import pytest

from tradingagents.ledger.accounting import Ledger, LedgerError
from tradingagents.ledger.db import connect
from tradingagents.ledger.store import Store

pytestmark = pytest.mark.unit

MON = "2026-10-19"


def _ledger(path):
    led = Ledger(connect(path))
    led.ensure_account()
    return led


def _race(path, monkeypatch, action, n=2):
    """Run ``action(ledger)`` on ``n`` threads, each with its own connection, with every
    state read slowed so that, without one transaction around read and write, all of them
    would see the same balance before any wrote."""
    real_state = Ledger.state

    def slow_state(self, *a, **k):
        out = real_state(self, *a, **k)
        time.sleep(0.15)
        return out

    monkeypatch.setattr(Ledger, "state", slow_state)
    results, start = [], threading.Barrier(n)

    def run():
        led = Ledger(connect(path, migrate=False))
        start.wait()
        try:
            action(led)
            results.append("ok")
        except LedgerError as exc:
            results.append(exc)

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    monkeypatch.setattr(Ledger, "state", real_state)
    return results


@pytest.mark.parametrize("action", [
    lambda led: led.fill("trading", "buy", "SYNTH", 1, 80.0, MON),
    lambda led: led.withdraw(80.0, MON),
    lambda led: led.reserve("trading", 80.0, as_of=MON),
    lambda led: led.fill("trading", "sell", "AAA", 3, 10.0, MON),
], ids=["buy", "withdraw", "reserve", "sell"])
def test_conflicting_writers_on_separate_connections_cannot_both_succeed(tmp_path, monkeypatch, action):
    path = tmp_path / "t.sqlite3"
    led = _ledger(path)
    led.contribute(100.0, MON)
    led.record("opening_position", "trading", MON, ticker="AAA", quantity=4.0)
    results = _race(path, monkeypatch, action)
    assert results.count("ok") == 1, results
    state = led.state("trading", MON)
    assert state.cash.available >= -1e-9 and state.positions.get("AAA", 0.0) >= -1e-9


def test_a_reservation_consumed_by_its_fill_pays_for_it_in_one_step(tmp_path):
    led = _ledger(tmp_path / "t.sqlite3")
    led.contribute(100.0, MON)
    rid = led.reserve("trading", 80.0, as_of=MON)
    with pytest.raises(LedgerError):
        led.fill("trading", "buy", "SYNTH", 1, 80.0, MON)            # the reserved cash is not spendable
    led.fill("trading", "buy", "SYNTH", 1, 80.0, MON, consumes=rid)
    s = led.state("trading", MON)
    assert s.cash.reserved == 0 and s.cash.settled == pytest.approx(20.0) and s.positions["SYNTH"] == 1


def test_two_identical_manual_buys_are_two_buys_but_a_retry_is_one(tmp_path):
    led = _ledger(tmp_path / "t.sqlite3")
    led.contribute(1000.0, MON)
    first, _ = led.fill("trading", "buy", "SYNTH", 1, 80.0, MON, op_id="click-one")
    second, _ = led.fill("trading", "buy", "SYNTH", 1, 80.0, MON, op_id="click-two")
    assert first != second
    s = led.state("trading", MON)
    assert s.positions["SYNTH"] == 2 and s.cash.settled == pytest.approx(840.0)

    retry, warnings = led.fill("trading", "buy", "SYNTH", 1, 80.0, MON, op_id="click-two")
    assert retry == second and warnings == []
    assert led.state("trading", MON).positions["SYNTH"] == 2
    with pytest.raises(LedgerError, match="different"):
        led.fill("trading", "buy", "SYNTH", 5, 80.0, MON, op_id="click-two")   # an id reused for another trade


def test_a_retry_is_answered_even_when_the_first_attempt_spent_the_cash(tmp_path):
    led = _ledger(tmp_path / "t.sqlite3")
    led.contribute(100.0, MON)
    first, _ = led.fill("trading", "buy", "SYNTH", 1, 80.0, MON, op_id="only-once")
    assert led.fill("trading", "buy", "SYNTH", 1, 80.0, MON, op_id="only-once")[0] == first   # no "not enough cash"
    a = led.contribute(50.0, MON, op_id="deposit-1")
    assert led.contribute(50.0, MON, op_id="deposit-1") == a
    assert led.contribute(50.0, MON, op_id="deposit-2") != a
    assert led.state("trading", MON).cash.settled == pytest.approx(120.0)


def test_imports_without_an_operation_id_keep_content_and_broker_dedupe(tmp_path):
    led = _ledger(tmp_path / "t.sqlite3")
    a, new = led.record("contribution", "trading", MON, cash=50.0, source="csv", source_ref="row-1")
    b, again = led.record("contribution", "trading", MON, cash=50.0, source="csv", source_ref="row-1")
    assert new and not again and a == b
    _, new_ref = led.record("contribution", "trading", MON, cash=50.0, uid="broker-77")
    _, dup_ref = led.record("contribution", "trading", MON, cash=50.0, uid="broker-77")
    assert new_ref and not dup_ref
    assert led.state("trading", MON).cash.settled == pytest.approx(100.0)


def test_cash_edits_are_saved_and_say_whether_they_are_flows_or_corrections(tmp_path):
    store = Store(tmp_path)
    led = store.ledger
    led.contribute(10.0, MON)
    out = store.adjust_cash(1234.0, MON, "correction", note="opening cash was mistyped")
    assert out["settled"] == pytest.approx(1234.0) and out["change"] == pytest.approx(1224.0)
    assert led.flows("trading", "2026-10-01", MON) == pytest.approx(10.0)         # a correction is not new money
    out = store.adjust_cash(1300.0, MON, "flow")
    assert out["settled"] == pytest.approx(1300.0)
    assert led.flows("trading", "2026-10-01", MON) == pytest.approx(76.0)         # a deposit is
    out = store.adjust_cash(1000.0, MON, "flow")
    assert led.flows("trading", "2026-10-01", MON) == pytest.approx(-224.0)       # and a withdrawal
    assert store.adjust_cash(1000.0, MON, "flow")["event"] is None               # no change, no event
    with pytest.raises(ValueError):
        store.adjust_cash(5.0, MON, "guess")
    reopened = Store(tmp_path)
    assert reopened.ledger.state("trading", MON).cash.settled == pytest.approx(1000.0)
