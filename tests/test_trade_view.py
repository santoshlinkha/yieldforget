"""The trade view's parts: the best buy and its prices, sector rotation with causes, the model's live record."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import feedback, trade

pytestmark = pytest.mark.unit


def _decision(**kw):
    entries = [
        {"ticker": "HELD", "action": "WAIT", "paper_action": "PAPER ENTER", "price": 50.0, "sigma": 0.02},
        {"ticker": "AAA", "action": "WAIT", "paper_action": "PAPER ENTER", "price": 100.0, "sigma": 0.02,
         "expected_net": 0.01, "p_win": 0.6, "p_risk": 0.05},
        {"ticker": "BAD", "action": "WAIT", "price": 10.0, "sigma": 0.02, "blocked": ["expected net below"]},
        {"ticker": "BBB", "action": "WAIT", "paper_action": "PAPER ENTER", "price": 20.0, "sigma": 0.03},
    ]
    lots = [{"ticker": "HELD", "shares": 1.0, "action": "EXIT", "entry_px": 40.0, "price": 50.0, "profit_level": 0.2,
             "risk_level": -0.15, "reasons": ["close-confirmed profit level"], "execute": "2026-10-28"}]
    return {"id": "d1", "info_cutoff": "2026-10-26", "execution_date": "2026-10-27", "mode": "experimental",
            "horizon": 20, "entries": entries, "lots": lots, **kw}


def test_the_best_buy_passes_the_gates_is_not_held_and_carries_its_exits():
    p = trade.plan(_decision(), 30.0, {"AAA": {"price": 97.0, "chance": 0.5, "stop": 89.0, "target": 106.0}})
    best = p["best"]
    # $30 across 5 open name slots (HELD exits at the next close): $6, not the whole budget.
    assert best["ticker"] == "AAA" and best["last_close"] == 100.0 and best["shares_at_reference"] == pytest.approx(0.06)
    assert best["entry"]["basis"] == "next_close" and best["reference_date"] == "2026-10-26"
    assert best["buy_limit"] == 100.0 and best["shares"] == pytest.approx(0.06) and best["dollars"] == 6.0
    assert best["size"]["budget_usd"] == 30.0 and best["size"]["slots_left"] == 5
    s = 0.02 * math.sqrt(20)
    assert best["stop"] == round(100 * math.exp(-s), 2) and best["target"] == round(100 * math.exp(s), 2)
    assert best["patient"]["price"] == 97.0 and best["patient"]["status"] == "untested policy"
    assert p["execution"]["order"].startswith("a buy of the whole size at the next session's official close")
    assert p["blocked_top"] == []
    assert [r["ticker"] for r in p["runners_up"]] == ["BBB"]            # BAD failed the model's gates
    sell = p["sells"][0]
    assert sell["action"] == "EXIT" and sell["sell_target"] == 48.0 and sell["sell_stop"] == 34.0
    assert any("Experimental" in n for n in p["notes"])


def test_the_trade_view_keeps_the_saved_capped_size_instead_of_the_whole_budget():
    # The audit's case: the decision sized AAA at $25 (0.25 shares at $100); $1,000 is available now.
    entries = [{"ticker": "AAA", "action": "ENTER", "paper_action": "PAPER ENTER", "price": 100.0, "sigma": 0.02,
                "dollars": 25.0, "shares": 0.25}]
    d = _decision(entries=entries, lots=[], account={"sleeve_value": 100.0},
                  settings={"rules": {"max_position_share": 0.25, "max_names": 5, "min_order": 1.0}})
    best = trade.plan(d, 1000.0)["best"]
    assert best["dollars"] == 25.0 and best["shares"] == pytest.approx(0.25)
    assert best["size"]["basis"] == "the saved decision's size" and best["size"]["budget_usd"] == 1000.0
    smaller = trade.plan(d, 10.0)["best"]                                 # cash fell since the decision
    assert smaller["dollars"] == 2.0 and "reduced" in smaller["size"]["basis"]    # $10 over 5 slots


def test_paper_picks_are_sized_by_the_decisions_cap_and_share_the_budget():
    entries = [{"ticker": t, "action": "WAIT", "paper_action": "PAPER ENTER", "price": 10.0, "sigma": 0.02}
               for t in ("AAA", "BBB", "CCC")]
    d = _decision(entries=entries, lots=[], account={"sleeve_value": 200.0},
                  settings={"rules": {"max_position_share": 0.25, "max_names": 2, "min_order": 1.0}})
    p = trade.plan(d, 300.0)
    sizes = [p["best"]["dollars"]] + [r["dollars"] for r in p["runners_up"]]
    assert sizes == [50.0, 50.0, 0.0]                                     # 25% of $200 each; two name slots
    assert "slot" in p["runners_up"][-1]["size"]["basis"]


def test_no_decision_and_an_empty_budget_say_so():
    assert not trade.plan(None, 10.0)["ready"]
    p = trade.plan(_decision(), 0.0)
    assert p["best"]["shares_at_reference"] == 0.0


def test_when_nothing_passes_the_top_of_the_ranking_is_shown_with_what_blocks_it():
    entries = [{"ticker": "AAA", "action": "WAIT", "price": 10.0, "sigma": 0.02, "score_pct": 0.99,
                "blocked": ["risk gate cannot be assessed"]}]
    p = trade.plan(_decision(entries=entries), 30.0)
    assert p["best"] is None and p["blocked_top"][0]["blocked"] == ["risk gate cannot be assessed"]


def _sector_prices(rotating_in="XLU", rotating_out="XLK", n=400):
    from tradingagents.research import rotation as R

    rng = np.random.default_rng(3)
    idx = pd.bdate_range(end="2026-10-26", periods=n)
    tnx = 4 + np.cumsum(rng.normal(0, 0.03, n))
    tnx[-25:] -= np.linspace(0, 0.5, 25)                               # rates fall over the last month
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    out = {"SPY": spy, "^TNX": tnx, "^IRX": np.full(n, 4.0), "^VIX": 15 + rng.normal(0, 1, n)}
    for t in ("USO", "UUP", "HYG", "IEF"):
        out[t] = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    dtnx = np.diff(tnx, prepend=tnx[0])
    for t in R.SECTORS:
        beta = -0.03 if t == rotating_in else 0.0                      # gains when rates fall
        own = rng.normal(0, 0.004, n) + beta * dtnx
        if t == rotating_out:
            own[-80:-15] += 0.002
            own[-15:] -= 0.004                                         # a leader starting to slip
        out[t] = spy * np.cumprod(1 + own)
    return pd.DataFrame(out, index=idx)


def test_rotation_flags_turns_and_names_the_rate_move_as_the_cause():
    from tradingagents.research.rotation import rotation

    r = rotation(None, None, close=_sector_prices())
    stages = {s["ticker"]: s["stage"] for s in r["sectors"]}
    assert stages["XLK"] in ("Weakening", "Lagging")
    xlu = next(s for s in r["sectors"] if s["ticker"] == "XLU")
    assert "rates fell" in xlu["cause"] and xlu["vs_spy_1m"] > 0
    assert all(a["direction"] in ("in", "out") and a["cause"] for a in r["alerts"])
    early = [a["stage"] == "early" for a in r["alerts"]]
    assert early == sorted(early, reverse=True)                        # early alerts come first


def test_predictions_are_saved_scored_once_and_calibrate_only_with_enough_data(tmp_path, monkeypatch):
    from tradingagents.ledger.store import Store

    store = Store(tmp_path)
    con = store.con
    days = pd.bdate_range("2026-01-02", periods=60)
    con.execute("PRAGMA foreign_keys=OFF")                              # decisions themselves are not needed here
    for d in days[:12]:
        scores = {f"S{i:02d}": {"rank": i + 1, "of": 30, "score_pct": 1 - i / 29, "expected_net": 0.02 - i * 0.001,
                                "p_win": 0.7 - i * 0.01, "p_risk": 0.1, "cost_rt": 0.001, "sigma": 0.01}
                  for i in range(30)}
        feedback.record(con, {"id": f"d{d.date()}", "info_cutoff": str(d.date()), "horizon": 5, "scores": scores,
                              "entries": [], "lots": [], "model": {"procedure_id": "proc-A"}})
    assert feedback.record(con, {"id": f"d{days[0].date()}", "info_cutoff": str(days[0].date()), "horizon": 5,
                                 "scores": {"S00": {}}, "entries": [], "lots": []}) == 1
    assert con.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == 360        # saved once
    # Higher-ranked stocks really did better: S00 rises fastest.
    close = pd.DataFrame({f"S{i:02d}": 100 * np.cumprod(np.full(60, 1 + 0.003 - i * 0.0002)) for i in range(30)},
                         index=days)
    close["SPY"] = 100.0
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close})
    from tradingagents.stocks.positions import Policy

    stocks = [c for c in close.columns if c != "SPY"]
    engine = {"px": close[stocks], "div": close[stocks] * 0.0, "low": close[stocks], "close": close[stocks],
              "policy": Policy("barrier_5", "barrier", 5)}
    assert feedback.resolve(con, tmp_path, engine=engine) == 360
    assert feedback.resolve(con, tmp_path, engine=engine) == 0                                  # never twice
    card = feedback.scorecard(con)
    assert card["matured"] == 360 and card["ic_mean"] > 0.9 and card["top_fifth_edge"] > 0
    end = days[-1].date()
    assert feedback.calibration(con, end, "proc-A")["active"]
    assert not feedback.calibration(con, end, "proc-A", min_rows=1000)["active"]
    assert not feedback.calibration(con, end, "another-procedure")["active"]
    today = pd.DataFrame({"p_win": [0.7, 0.4], "expected_net": [0.02, -0.01]}, index=["X", "Y"])
    out = feedback.apply(today, feedback.calibration(con, end, "proc-A"))
    assert list(out["p_win_model"]) == [0.7, 0.4] and out["p_win"].between(0, 1).all()
    path = feedback.export(con, tmp_path / "hist")
    assert path.exists() and len(pd.read_csv(path)) == 360
    assert date.fromisoformat(pd.read_csv(path)["matured_on"].dropna().iloc[0])


def test_a_pick_is_judged_only_after_its_20_sessions_and_against_its_cost():
    from tradingagents.stocks import track_record as T

    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame({"AAA": np.linspace(100, 110, 30), "SPY": 100.0, "BBB": np.linspace(100, 90, 30)}, index=idx)
    done = T._follow(close, "AAA", str(idx[0].date()), 20)
    assert done["entry_date"] == str(idx[1].date()) and done["sessions"] == 20 and done["status"] == "done"
    assert done["return"] == pytest.approx(close["AAA"].iloc[21] / close["AAA"].iloc[1] - 1) and done["spy"] == 0.0
    still = T._follow(close, "AAA", str(idx[20].date()), 20)
    assert still["status"] == "open" and still["sessions"] == 8
    assert T._follow(close, "AAA", str(idx[-1].date()), 20)["status"] == "not bought yet"


def test_the_second_models_buys_are_recorded_and_a_limit_fills_only_through_its_price(tmp_path, monkeypatch):
    import json

    from tradingagents.ledger.store import Store
    from tradingagents.stocks import track_record as T

    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0, "SPY": 100.0}, index=idx)
    low = close.copy()
    low.loc[idx[3], "BBB"] = 40.0                                   # deep through the limit on the 3rd session
    sigma = pd.DataFrame(0.02, index=idx, columns=close.columns)
    p = {"ticker": "BBB", "data_to": str(idx[0].date()), "limit_k": 0.5, "limit_days": 5, "fill_buffer": 0.001,
         "horizon": 20}
    got = T._follow_limit(close, low, sigma, p)
    assert got["entry_date"] == str(idx[3].date()) and got["entry"] == pytest.approx(50 * np.exp(-0.5 * 0.02 * np.sqrt(5)))
    assert T._follow_limit(close, close, sigma, {**p, "ticker": "AAA"})["status"] == "no fill"   # never traded below

    store = Store(tmp_path)
    opts = lambda c, ex: [{"action": "cash", "expected_excess": 0.0}, {"action": c, "expected_excess": ex}]  # noqa: E731
    payload = {"entries": [], "lots": [], "horizon": 20, "scores": {},
               "policy": {"available": True, "policies": {"limit_dip": {"limit_k": 0.5, "limit_days": 5, "fill_buffer": 0.001}},
                          "stocks": {"AAA": {"action": "insufficient evidence", "model_choice": "next_close",
                                             "options": opts("next_close", 0.01), "validated": False},
                                     "BBB": {"action": "insufficient evidence", "model_choice": "limit_dip",
                                             "options": opts("limit_dip", 0.02), "validated": False},
                                     "CCC": {"action": "insufficient evidence", "model_choice": "cash", "options": []}}}}
    monkeypatch.setattr(T, "_picks", lambda con, per_day=5: [])
    rows = [{"id": "d1", "info_cutoff": str(idx[0].date()), "execution_date": str(idx[1].date()),
             "payload_json": json.dumps(payload)}]

    class Con:                       # the decision as stock_decisions holds it; hidden rows in the real table
        def execute(self, sql, *a):
            if "track_record_hidden" in sql:
                return store.con.execute(sql, *a)
            return type("Rows", (), {"fetchall": lambda self: rows})()

    picks = T._policy_picks(Con())
    assert [(r["ticker"], r["choice"]) for r in picks] == [("BBB", "limit_dip"), ("AAA", "next_close")]   # cash left out
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close, "Low": low})
    out = T.build(Con(), tmp_path)
    assert {r["ticker"] for r in out["policy"]} == {"AAA", "BBB"} and out["hidden"] == 0
    T.hide(store.con, out["policy"][0]["key"])
    again = T.build(Con(), tmp_path)
    assert len(again["policy"]) == 1 and again["hidden"] == 1
    assert T.unhide(store.con, out["policy"][0]["key"]) == 1 and T.build(Con(), tmp_path)["hidden"] == 0   # Undo
    T.hide(store.con, out["policy"][1]["key"])
    assert T.restore_all(store.con) == 1 and len(T.build(Con(), tmp_path)["policy"]) == 2


def test_the_track_record_leads_with_net_returns_with_the_cost_the_verdict_uses(tmp_path, monkeypatch):
    from tradingagents.stocks import track_record as T

    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame({"AAA": np.r_[100.0, 100.0, np.full(28, 101.0)], "SPY": 100.0}, index=idx)
    picks = [{"key": f"p|{d}|AAA", "ticker": "AAA", "data_to": str(idx[0].date()), "horizon": 20, "place": 1,
              "cost_rt": 0.02, "role": "best"} for d in (1, 2)]
    monkeypatch.setattr(T, "_picks", lambda con: [dict(p) for p in picks])
    monkeypatch.setattr(T, "_policy_picks", lambda con: [])
    monkeypatch.setattr(T, "hidden", lambda con: set())
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close, "Low": close})
    s = T.build(None, tmp_path)["summary"]["all_picks"]
    # +1% raw, 2% cost: the headline is the net -1%, and "beat cost" agrees with it.
    assert s["avg_raw_return"] == pytest.approx(0.01) and s["avg_net_return"] == pytest.approx(-0.01)
    assert s["correct"] == 0 and s["dates"] == 1 and "avg_return" not in s


def test_one_dollar_on_every_recommendation_adds_up_net_of_costs(tmp_path, monkeypatch):
    from tradingagents.stocks import track_record as T

    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame({"AAA": np.r_[100.0, 100.0, np.full(28, 110.0)], "BBB": np.r_[50.0, 50.0, np.full(28, 45.0)],
                          "SPY": 100.0}, index=idx)
    picks = [{"key": f"p|{t}", "ticker": t, "data_to": str(idx[0].date()), "horizon": 20, "place": i + 1,
              "cost_rt": 0.01, "role": "best"} for i, t in enumerate(("AAA", "BBB"))]
    monkeypatch.setattr(T, "_picks", lambda con: [dict(p) for p in picks])
    monkeypatch.setattr(T, "_policy_picks", lambda con: [])
    monkeypatch.setattr(T, "hidden", lambda con: set())
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close, "Low": close})
    one = T.build(None, tmp_path)["one_dollar"]
    f = one["all_picks"]["finished"]
    # $1 each: AAA +10% -1% cost = +$0.09, BBB -10% -1% = -$0.11: $2 in, $1.98 now.
    assert f["trades"] == 2 and f["invested"] == 2.0 and f["profit"] == pytest.approx(-0.02)
    assert f["value"] == pytest.approx(1.98) and one["best_buy"]["finished"]["profit"] == pytest.approx(0.09)
    assert one["all_picks"]["running"][-1] == {"date": str(idx[0].date()), "invested": 2.0, "profit": -0.02}
    # The weighted split: $1 a day across the day's picks (equal here: too little history for the risk model).
    w = one["weighted"]["finished"]
    assert w["trades"] == 2 and w["invested"] == pytest.approx(1.0) and w["profit"] == pytest.approx(-0.01)


def test_the_cumulative_value_curve_follows_each_dollar_from_entry_to_exit(tmp_path, monkeypatch):
    from tradingagents.stocks import track_record as T

    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame({"AAA": np.r_[100.0, 100.0, np.full(28, 110.0)], "SPY": 100.0}, index=idx)
    picks = [{"key": "p|AAA", "ticker": "AAA", "data_to": str(idx[0].date()), "horizon": 20, "place": 1,
              "cost_rt": 0.02, "role": "best"}]
    monkeypatch.setattr(T, "_picks", lambda con: [dict(p) for p in picks])
    monkeypatch.setattr(T, "_policy_picks", lambda con: [])
    monkeypatch.setattr(T, "hidden", lambda con: set())
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close, "Low": close})
    curve = T.build(None, tmp_path)["one_dollar"]["best_buy"]["curve"]
    assert curve[0] == {"date": str(idx[1].date()), "invested": 1.0, "value": pytest.approx(0.99)}  # bought: half the cost
    assert curve[1]["value"] == pytest.approx(1.09)                       # held at 110, entry half of the cost paid
    sold = next(c for c in curve if c["date"] == str(idx[21].date()))
    assert sold["value"] == pytest.approx(1.08) and curve[-1]["value"] == pytest.approx(1.08)  # +10% less 2%
    assert all(c["invested"] == 1.0 for c in curve)


def test_an_ai_pick_is_bought_at_buy_at_and_sold_at_its_target_or_stop_whichever_comes_first():
    from tradingagents.stocks import track_record as T

    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame({"AAA": 100.0, "SPY": 100.0}, index=idx)
    low, high = close.copy(), close.copy()
    low.loc[idx[2], "AAA"] = 95.0                      # reaches Buy at 96 on the 2nd session after the pick
    high.loc[idx[5], "AAA"] = 110.0                    # then the Sell target 108
    p = {"ticker": "AAA", "data_to": str(idx[0].date()), "buy_at": 96.0, "target": 108.0, "stop": 90.0,
         "expires": str(idx[4].date()), "horizon": 20}
    got = T._follow_plan(close, low, high, p)
    assert got["entry_date"] == str(idx[2].date()) and got["entry"] == 96.0
    assert got["exit"] == "target" and got["last"] == 108.0 and got["last_date"] == str(idx[5].date())
    assert got["return"] == pytest.approx(108 / 96 - 1) and got["status"] == "done"
    low.loc[idx[5], "AAA"] = 89.0                      # the stop too that day: counted as the stop (the worse case)
    stopped = T._follow_plan(close, low, high, p)
    assert stopped["exit"] == "stop" and stopped["last"] == 90.0
    flat = T._follow_plan(close, close, close, p)      # never down to Buy at before the plan expired
    assert flat["status"] == "no fill"
    nobuy = T._follow_plan(close, close, close, {**p, "buy_at": None, "target": None, "stop": None})
    assert nobuy["entry_date"] == str(idx[1].date()) and nobuy["exit"] == "day 20" and nobuy["sessions"] == 20


def test_a_sell_call_is_counted_once_and_is_right_when_the_stock_then_fell(tmp_path, monkeypatch):
    import json

    from tradingagents.stocks import track_record as T

    idx = pd.bdate_range("2026-01-01", periods=30)
    lot = lambda action: {"ticker": "AAA", "action": action, "entry_px": 120.0}  # noqa: E731
    rows = [{"id": f"d{i}", "info_cutoff": str(idx[i].date()), "payload_json": json.dumps({"lots": [lot(a)]})}
            for i, a in enumerate(("HOLD", "REDUCE", "REDUCE", "HOLD"))]

    class Con:
        def execute(self, sql, *a):
            return type("Rows", (), {"fetchall": lambda self: rows})()

    calls = T._sell_calls(Con())
    assert [(c["data_to"], c["action"]) for c in calls] == [(str(idx[1].date()), "REDUCE")]   # the repeat is one call
    close = pd.DataFrame({"AAA": np.r_[100.0, 100.0, 100.0, np.full(27, 90.0)], "SPY": 100.0}, index=idx)
    monkeypatch.setattr(T, "_picks", lambda con: [])
    monkeypatch.setattr(T, "_policy_picks", lambda con: [])
    monkeypatch.setattr(T, "_ai_picks", lambda con: [])
    monkeypatch.setattr(T, "_sell_calls", lambda con: [dict(c) for c in calls])
    monkeypatch.setattr(T, "hidden", lambda con: set())
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close, "Low": close})
    out = T.build(None, tmp_path)
    s = out["sells"][0]
    assert s["entry_date"] == str(idx[2].date()) and s["return"] == pytest.approx(-0.10) and s["verdict"] == "correct"
    assert out["summary"]["sells"]["correct"] == 1


def test_the_track_record_is_a_paper_trade_from_one_start_date():
    from tradingagents.stocks.track_record import paper_start

    picks = [{"data_to": "2026-09-01"}, {"data_to": "2026-10-01"}]
    agents = [{"data_to": "2026-10-20"}]
    assert paper_start([picks, [], agents]) == "2026-10-20"        # the first day every group made calls
    assert paper_start([picks, agents], "2026-09-15") == "2026-09-15"
    assert paper_start([[], []]) is None
