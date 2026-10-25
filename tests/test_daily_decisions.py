"""The daily decision: actions, dates, sizing limits, evidence modes, frozen records and the shadow account."""

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_stocks import _stock_data  # noqa: E402

from tradingagents.research.features import MACRO_FEATURES  # noqa: E402
from tradingagents.stocks import daily  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def env(tmp_path, monkeypatch):
    from tradingagents.ledger.store import Store
    from tradingagents.stocks import data as D
    from tradingagents.stocks import macro as M

    data = _stock_data(tmp_path, n=16, years=8, requested=())
    monkeypatch.setattr(D, "load", lambda *a, **k: data)
    monkeypatch.setattr(M, "features", lambda dates, cache_dir: pd.DataFrame(index=dates, columns=MACRO_FEATURES, dtype=float))
    monkeypatch.setattr("tradingagents.optimizer.data.latest_prices",
                        lambda cache_dir, tickers: {t: float(data.close[t].dropna().iloc[-1]) for t in tickers if t in data.close})
    store = Store(tmp_path / "book")
    sessions = data.close.index
    cutoff = sessions[-1]
    funded = sessions[-30].date()
    store.ledger.contribute(1000.0, funded, sleeve="trading", settle_date=funded)
    return store, data, tmp_path, cutoff, sessions


def _buy(store, data, ticker, day, dollars):
    price = float(data.close[ticker].loc[day])
    store.ledger.fill("trading", "buy", ticker, dollars / price, price, day.date(), settle_date=day.date())


def test_a_decision_has_dates_modes_sizes_and_is_frozen(env):
    from tradingagents.ledger import capital
    from tradingagents.ledger.sessions import next_session

    store, data, cache, cutoff, sessions = env
    _buy(store, data, "S03", sessions[-25], 200.0)                 # past its 20-session deadline by the cutoff
    _buy(store, data, "S04", sessions[-3], 400.0)                  # 40% of the sleeve: above the 25% cap
    rec = daily.run(store, cache, cutoff.date(), train=True)
    assert rec["info_cutoff"] == str(cutoff.date()) and rec["execution_date"] == str(next_session(cutoff.date()))
    lots = {x["ticker"]: x for x in rec["lots"]}
    assert lots["S03"]["action"] == "EXIT" and "deadline" in lots["S03"]["reasons"][0]
    assert lots["S04"]["action"] in ("REDUCE", "EXIT")
    # No study backs this policy, so nothing is validated: buys are paper only.
    assert rec["mode"] == "experimental" and all(x["action"] == "WAIT" for x in rec["entries"])
    paper = [x for x in rec["entries"] if x.get("paper_action") == "PAPER ENTER"]
    staying = sum(1 for x in rec["lots"] if x["action"] != "EXIT")      # names still held after the next close
    assert len(paper) + staying <= 5
    for x in paper:
        assert x["dollars"] <= 0.25 * rec["account"]["sleeve_value"] + 1e-6
    assert sum(x["dollars"] for x in paper) <= rec["account"]["settled_cash"] - 0 + 1e-6
    assert rec["settings"]["capital"]["trading_capital_usd"] == 1000.0
    stored = daily.load(store.con, rec["id"])
    capital.update(store, {"trading_capital_usd": 3000})
    assert daily.load(store.con, rec["id"]) == stored                    # later settings never change a record
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.con.execute("UPDATE stock_decisions SET mode = 'validated'")


def test_models_are_reused_when_only_the_dollar_amounts_change(env):
    from tradingagents.ledger import capital

    store, data, cache, cutoff, _ = env
    first = daily.run(store, cache, cutoff.date(), train=True)
    capital.update(store, {"trading_capital_usd": 1500, "daily_investment_usd": 10})
    second = daily.run(store, cache, cutoff.date(), train=False)          # no training allowed, and none needed
    assert first["model"]["bundle"]["reused"] is False and second["model"]["bundle"]["reused"] is True
    assert second["model"]["bundle"]["sha256"] == first["model"]["bundle"]["sha256"]
    assert second["settings"]["capital"]["trading_capital_usd"] == 1500.0


def test_no_lot_is_sold_on_its_entry_session(env):
    store, data, cache, cutoff, sessions = env
    _buy(store, data, "S05", cutoff, 900.0)
    rec = daily.run(store, cache, cutoff.date(), train=True)
    lot = next(x for x in rec["lots"] if x["ticker"] == "S05")
    assert lot["action"] == "HOLD" and "same-day" in lot["reasons"][0]


def test_missing_risk_blocks_entry():
    today = pd.DataFrame({"expected_net": [0.05], "net_sd": [0.1], "net_q10": [-0.1], "expected_excess": [0.02],
                          "p_win": [0.7], "p_risk": [np.nan], "sigma": [0.02], "cost_rt": [0.003], "score_pct": [0.99],
                          "in_index": [True]}, index=["AAA"])
    r = {**daily.DEFAULT_RULES, "max_names": 5, "max_position_share": 0.25, "min_order": 1.0, "max_large_loss_prob": 0.15}
    lines = daily.entries(today, set(), set(), r, 1000.0, 1000.0, {"total": 6000.0, "by_ticker": {}},
                          pd.Series({"AAA": 10.0}), validated=True, cutoff=None, execute="2026-10-26")
    assert lines[0]["action"] == "WAIT" and "risk estimate missing" in lines[0]["blocked"][0]


def test_whole_ira_concentration_blocks_entry():
    today = pd.DataFrame({"expected_net": [0.05], "net_sd": [0.1], "net_q10": [-0.1], "expected_excess": [0.02],
                          "p_win": [0.7], "p_risk": [0.05], "sigma": [0.02], "cost_rt": [0.003], "score_pct": [0.99],
                          "in_index": [True]}, index=["AAA"])
    r = {**daily.DEFAULT_RULES, "max_names": 5, "max_position_share": 0.25, "min_order": 1.0, "max_large_loss_prob": 0.15}
    lines = daily.entries(today, set(), set(), r, 1000.0, 1000.0, {"total": 6000.0, "by_ticker": {"AAA": 500.0}},
                          pd.Series({"AAA": 10.0}), validated=True, cutoff=None, execute="2026-10-26")
    assert lines[0]["action"] == "WAIT" and any("IRA concentration" in b for b in lines[0]["blocked"])
    ok = daily.entries(today, set(), set(), r, 1000.0, 1000.0, {"total": 6000.0, "by_ticker": {}},
                       pd.Series({"AAA": 10.0}), validated=True, cutoff=None, execute="2026-10-26")
    # Equal dollars across the open name slots (1,000 / 5), within the 25% cap: never all the cash on one name.
    assert ok[0]["action"] == "ENTER" and ok[0]["dollars"] == 200.0


def test_lots_are_rebuilt_first_in_first_out(tmp_path):
    from tradingagents.ledger.store import Store

    store = Store(tmp_path)
    d1, d2, d3 = date(2026, 10, 1), date(2026, 10, 5), date(2026, 10, 9)
    store.ledger.contribute(1000.0, d1, sleeve="trading", settle_date=d1)
    store.ledger.fill("trading", "buy", "AAA", 2.0, 10.0, d1, settle_date=d1)
    store.ledger.fill("trading", "buy", "AAA", 3.0, 12.0, d2, settle_date=d2)
    store.ledger.fill("trading", "sell", "AAA", 2.5, 13.0, d3, settle_date=d3)
    lots = daily.lots_from_ledger(store, "trading", d3)
    assert lots == [{"ticker": "AAA", "shares": 2.5, "entry_date": "2026-10-05", "entry_px": 12.0}]


def test_the_shadow_account_freezes_its_capital_and_advances_once(env):
    from tradingagents.ledger import capital

    store, data, cache, cutoff, sessions = env
    daily.run(store, cache, sessions[-10].date(), train=True)
    first = json.loads(store.con.execute("SELECT state_json FROM shadow_snapshots ORDER BY id LIMIT 1").fetchone()[0])
    capital.update(store, {"trading_capital_usd": 9000})
    daily.run(store, cache, cutoff.date(), train=True)
    last = json.loads(store.con.execute("SELECT state_json FROM shadow_snapshots ORDER BY id DESC LIMIT 1").fetchone()[0])
    assert first["capital"] == last["capital"] == 1000.0               # frozen at inception
    days = [e[0] for e in last["equity"]]
    assert len(days) == len(set(days))                                 # each session replayed once


def test_tied_forecasts_are_ordered_by_score_then_ticker_never_by_input_order():
    base = pd.DataFrame({"expected_net": [0.007] * 4, "net_sd": [0.08] * 4, "net_q10": [-0.1] * 4,
                         "expected_excess": [0.0] * 4, "p_win": [0.54] * 4, "p_risk": [0.07] * 4, "sigma": [0.02] * 4,
                         "cost_rt": [0.003] * 4, "score_pct": [0.95, 0.99, 0.95, 0.97], "in_index": [True] * 4},
                        index=["ZZZ", "MMM", "AAA", "QQQ"])
    r = {**daily.DEFAULT_RULES, "max_names": 5, "max_position_share": 0.25, "min_order": 1.0, "max_large_loss_prob": 0.15}
    ira = {"total": 6000.0, "by_ticker": {}}
    prices = pd.Series(10.0, index=base.index)
    a = daily.entries(base, set(), set(), r, 1000.0, 1000.0, ira, prices, True, None, "2026-10-26")
    b = daily.entries(base.sample(frac=1, random_state=3), set(), set(), r, 1000.0, 1000.0, ira, prices, True, None,
                      "2026-10-26")
    assert [x["ticker"] for x in a] == [x["ticker"] for x in b] == ["MMM", "QQQ", "AAA", "ZZZ"]
    assert all(x["tied_with"] == 3 for x in a)


def test_a_cash_fund_in_the_sleeve_is_cash_not_a_lot_to_review(env, monkeypatch):
    store, data, cache, cutoff, sessions = env
    day = sessions[-5]
    store.ledger.fill("trading", "buy", "SGOV", 5.0, 100.0, day.date(), settle_date=day.date())
    monkeypatch.setattr("tradingagents.optimizer.data.latest_prices", lambda cache_dir, tickers: {t: 100.5 for t in tickers})
    rec = daily.run(store, cache, cutoff.date(), train=True)
    assert all(x["ticker"] != "SGOV" for x in rec["lots"])
    assert rec["cash_fund"]["holdings"] == {"SGOV": 5.0} and rec["account"]["cash_fund_value"] == pytest.approx(502.5)
    assert np.isfinite(rec["account"]["sleeve_value"])
