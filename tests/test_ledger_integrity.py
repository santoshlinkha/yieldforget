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


def _ny(y, m, d, h, mi=0):
    from datetime import datetime

    import pytz
    return pytz.timezone("America/New_York").localize(datetime(y, m, d, h, mi))


def test_half_day_sessions_move_the_order_cutoff_and_the_final_bar():
    from tradingagents.dataflows.local_store import last_final_session
    from tradingagents.ledger.recommendations import earliest_execution, session_for
    from tradingagents.ledger.sessions import early_closes, session_close

    assert {d.isoformat() for d in early_closes(2026)} == {"2026-11-27", "2026-12-24"}   # July 3, 2026 is a holiday
    assert {d.isoformat() for d in early_closes(2028)} >= {"2028-07-03", "2028-11-24"}
    assert session_close("2026-11-27").hour == 13 and session_close("2026-11-25").hour == 16
    # The audit's case: published at 2 pm on the half day after Thanksgiving, with data through the 25th.
    assert session_for(_ny(2026, 11, 27, 14)) == "2026-11-30"
    assert earliest_execution("2026-11-25", _ny(2026, 11, 27, 14)) == "2026-11-30"
    assert session_for(_ny(2026, 11, 27, 12, 44)) == "2026-11-27"
    assert session_for(_ny(2026, 11, 27, 12, 45)) == "2026-11-30"            # exactly at the cutoff: too late
    assert session_for(_ny(2026, 11, 25, 15, 44)) == "2026-11-25"
    assert session_for(_ny(2026, 11, 25, 15, 45)) == "2026-11-27"
    assert session_for(_ny(2026, 11, 2, 15, 44)) == "2026-11-02"             # the Monday after the DST change
    assert last_final_session(_ny(2026, 11, 27, 13, 31)).isoformat() == "2026-11-27"
    assert last_final_session(_ny(2026, 11, 27, 13, 29)).isoformat() == "2026-11-25"
    assert last_final_session(_ny(2026, 11, 25, 16, 29)).isoformat() == "2026-11-24"


def test_a_filing_or_report_after_a_half_day_close_is_known_the_next_session():
    import numpy as np
    import pandas as pd

    from tradingagents.stocks import consensus, news_archive

    sessions = pd.DatetimeIndex(["2026-11-25", "2026-11-27", "2026-11-30"])
    accepted = pd.Series(["2026-11-27T19:00:00Z",      # 2 pm New York on the half day: after its 1 pm close
                          "2026-11-27T17:30:00Z",      # 12:30 pm: before it
                          "2026-11-25T20:30:00Z"])     # 3:30 pm on a regular day: before its 4 pm close
    flags = news_archive.after_close(accepted).tolist()
    assert flags == [1, 0, 0]
    known = news_archive.known_session(pd.Series(["2026-11-27", "2026-11-27", "2026-11-25"]), pd.Series(flags), sessions)
    assert [d.date().isoformat() for d in known] == ["2026-11-30", "2026-11-27", "2026-11-25"]
    assert news_archive._ny("2026-11-27T19:00:00Z") == ("2026-11-27", 1)
    times = pd.DatetimeIndex(pd.to_datetime(accepted, utc=True))
    assert np.array_equal(consensus._known_after_close(times, sessions), [2, 1, 0])


def test_a_fred_failure_never_logs_or_raises_the_key(tmp_path, monkeypatch, caplog):
    import logging

    import requests

    from tradingagents.stocks import macro

    secret = "SYNTHETIC0123456789abcdef"
    monkeypatch.setenv("FRED_API_KEY", secret)
    monkeypatch.setattr(macro.time, "sleep", lambda s: None)

    def refuse(url, params=None, timeout=None):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}?api_key={params['api_key']}")

    monkeypatch.setattr(requests, "get", refuse)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(requests.ConnectionError) as err:
        macro._fetch("DFF", False, tmp_path)
    assert secret not in str(err.value) and secret not in caplog.text
    assert err.value.__cause__ is None and err.value.__context__ is None or secret not in repr(err.value.__context__)
