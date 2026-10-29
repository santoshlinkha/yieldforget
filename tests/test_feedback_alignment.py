"""Live feedback learns the trade the model was trained on, from what was known at the time, once per observation.

Synthetic prices for made-up tickers in a temporary database; no model is fitted.
"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.ledger.store import Store
from tradingagents.stocks import feedback
from tradingagents.stocks.positions import POLICIES

pytestmark = pytest.mark.unit

DAYS = pd.bdate_range("2026-03-02", periods=80)


def _store(tmp_path):
    store = Store(tmp_path)
    store.con.execute("PRAGMA foreign_keys=OFF")      # the decisions themselves are not needed here
    return store


def _record(con, day, tickers, procedure="proc-A", run="", p_win=0.6, expected=0.01, horizon=20):
    scores = {t: {"rank": i + 1, "of": len(tickers), "score_pct": 1 - i / max(len(tickers) - 1, 1),
                  "expected_net": expected, "p_win": p_win, "p_win_raw": p_win, "expected_net_raw": expected,
                  "p_risk": 0.1, "cost_rt": 0.004, "sigma": 0.01} for i, t in enumerate(tickers)}
    return feedback.record(con, {"id": f"{procedure}-{day.date()}{run}", "info_cutoff": str(day.date()),
                                 "horizon": horizon, "scores": scores, "entries": [], "lots": [],
                                 "model": {"procedure_id": procedure}})


def _engine(close):
    stocks = [c for c in close.columns if c != "SPY"]
    return {"px": close[stocks], "div": close[stocks] * 0.0, "low": close[stocks], "close": close[stocks],
            "policy": POLICIES["barrier_20"]}


def _bars(monkeypatch, close):
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close})


def test_a_barrier_exit_is_scored_as_the_policy_not_as_a_hold_to_the_deadline(tmp_path, monkeypatch):
    # The audit's case: entry 100; the next two closes 110; then 90 to the deadline; sigma 0.01; cost 0.004.
    path = [100.0, 100.0, 110.0, 110.0] + [90.0] * 40
    close = pd.DataFrame({"SYN": path, "SPY": 100.0}, index=DAYS[:len(path)])
    store = _store(tmp_path)
    _record(store.con, close.index[0], ["SYN"])
    _bars(monkeypatch, close)
    assert feedback.resolve(store.con, tmp_path, engine=_engine(close)) == 1
    row = store.con.execute("SELECT * FROM model_outcomes").fetchone()
    assert row["policy_net"] == pytest.approx(0.096) and row["policy_reason"] == "target"
    assert row["policy_exit_on"] == str(close.index[3].date())                  # sold at the close after the trigger
    assert row["realized_return"] == pytest.approx(-0.10) and row["won"] == 0  # the fixed-horizon diagnostic
    m = feedback.observations(store.con, close.index[-1].date(), "proc-A")
    assert (m["policy_net"] > 0).all()                                          # calibration learns a win


def _rising(n_tickers=12):
    rng = np.random.default_rng(1)
    cols = {f"S{i:02d}": 100 * np.cumprod(1 + rng.normal(0.002 - 0.0003 * i, 0.01, len(DAYS)))
            for i in range(n_tickers)}
    return pd.DataFrame({**cols, "SPY": 100.0}, index=DAYS)


def _resolved_store(tmp_path, monkeypatch, close, days, **kw):
    store = _store(tmp_path)
    for d in days:
        _record(store.con, d, [c for c in close.columns if c != "SPY"], **kw)
    _bars(monkeypatch, close)
    feedback.resolve(store.con, tmp_path, engine=_engine(close))
    return store


def test_outcomes_after_the_cutoff_cannot_change_an_earlier_calibration(tmp_path, monkeypatch):
    close = _rising()
    cutoff = DAYS[45].date()
    early = _resolved_store(tmp_path / "a", monkeypatch, close, DAYS[:10])
    before = feedback.calibration(early.con, cutoff, "proc-A", min_rows=10, min_dates=2)
    # Predictions made after the cutoff: every one of their exits comes later than it.
    both = _resolved_store(tmp_path / "b", monkeypatch, close, list(DAYS[:10]) + list(DAYS[46:56]))
    after = feedback.calibration(both.con, cutoff, "proc-A", min_rows=10, min_dates=2)
    assert before["active"] and before["rows"] == after["rows"]
    assert before["p_win_map"] == after["p_win_map"] and before["net_map"] == after["net_map"]
    later = feedback.calibration(both.con, DAYS[-1].date(), "proc-A", min_rows=10, min_dates=2)
    assert later["rows"] > before["rows"]


def test_another_procedures_record_cannot_change_this_ones_calibration(tmp_path, monkeypatch):
    close = _rising()
    store = _resolved_store(tmp_path, monkeypatch, close, DAYS[:10])
    mine = feedback.calibration(store.con, DAYS[-1].date(), "proc-A", min_rows=10, min_dates=2)
    for d in DAYS[:10]:
        _record(store.con, d, [c for c in close.columns if c != "SPY"], procedure="proc-B", p_win=0.1,
                expected=-0.05)
    feedback.resolve(store.con, tmp_path, engine=_engine(close))
    again = feedback.calibration(store.con, DAYS[-1].date(), "proc-A", min_rows=10, min_dates=2)
    assert mine["p_win_map"] == again["p_win_map"] and mine["rows"] == again["rows"]


def test_running_a_day_again_does_not_count_its_outcomes_twice(tmp_path, monkeypatch):
    close = _rising()
    once = _resolved_store(tmp_path / "a", monkeypatch, close, DAYS[:10])
    twice = _store(tmp_path / "b")
    for d in DAYS[:10]:
        for run in ("", "-again", "-third"):
            _record(twice.con, d, [c for c in close.columns if c != "SPY"], run=run)
    feedback.resolve(twice.con, tmp_path, engine=_engine(close))
    a = feedback.calibration(once.con, DAYS[-1].date(), "proc-A", min_rows=10, min_dates=2)
    b = feedback.calibration(twice.con, DAYS[-1].date(), "proc-A", min_rows=10, min_dates=2)
    assert a["rows"] == b["rows"] == 10 * 12 and a["p_win_map"] == b["p_win_map"]


def test_a_replay_resolves_only_what_its_prices_reach(tmp_path, monkeypatch):
    close = _rising()
    store = _store(tmp_path)
    _record(store.con, DAYS[0], ["S00"])
    _bars(monkeypatch, close)                                   # the cache has the whole history...
    assert feedback.resolve(store.con, tmp_path, engine=_engine(close.iloc[:15])) == 0   # ...the replay does not
    assert feedback.resolve(store.con, tmp_path, engine=_engine(close)) == 1
    assert store.con.execute("SELECT policy_net FROM model_outcomes").fetchone()[0] is not None


def test_calibration_is_fitted_on_raw_numbers_not_on_already_calibrated_ones(tmp_path, monkeypatch):
    close = _rising()
    store = _store(tmp_path)
    for d in DAYS[:10]:
        scores = {t: {"rank": 1, "of": 12, "score_pct": 0.5, "expected_net": 0.5, "p_win": 0.99,   # calibrated
                      "p_win_raw": 0.6, "expected_net_raw": 0.01, "cost_rt": 0.004, "sigma": 0.01}
                  for t in close.columns if t != "SPY"}
        feedback.record(store.con, {"id": f"d{d.date()}", "info_cutoff": str(d.date()), "horizon": 20,
                                    "scores": scores, "entries": [], "lots": [],
                                    "model": {"procedure_id": "proc-A", "live_calibration": {"version": "live-x"}}})
    _bars(monkeypatch, close)
    feedback.resolve(store.con, tmp_path, engine=_engine(close))
    m = feedback.observations(store.con, DAYS[-1].date(), "proc-A")
    assert set(m["p_win_x"]) == {0.6} and set(m["expected_net_x"]) == {0.01}
    assert set(m["calibration_version"]) == {"live-x"}
