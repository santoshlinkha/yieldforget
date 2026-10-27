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
    assert best["ticker"] == "AAA" and best["last_close"] == 100.0 and best["shares_at_reference"] == pytest.approx(0.3)
    assert best["entry"]["basis"] == "next_close" and best["reference_date"] == "2026-10-26"
    assert best["buy_limit"] == 100.0 and best["shares"] == pytest.approx(0.3)
    s = 0.02 * math.sqrt(20)
    assert best["stop"] == round(100 * math.exp(-s), 2) and best["target"] == round(100 * math.exp(s), 2)
    assert best["patient"]["price"] == 97.0 and best["patient"]["status"] == "untested policy"
    assert p["execution"]["order"].startswith("a buy of the whole size at the next session's official close")
    assert p["blocked_top"] == []
    assert [r["ticker"] for r in p["runners_up"]] == ["BBB"]            # BAD failed the model's gates
    sell = p["sells"][0]
    assert sell["action"] == "EXIT" and sell["sell_target"] == 48.0 and sell["sell_stop"] == 34.0
    assert any("Experimental" in n for n in p["notes"])


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
                                "p_win": 0.7 - i * 0.01, "p_risk": 0.1, "cost_rt": 0.001} for i in range(30)}
        feedback.record(con, {"id": f"d{d.date()}", "info_cutoff": str(d.date()), "horizon": 5, "scores": scores,
                              "entries": [], "lots": []})
    assert feedback.record(con, {"id": f"d{days[0].date()}", "info_cutoff": str(days[0].date()), "horizon": 5,
                                 "scores": {"S00": {}}, "entries": [], "lots": []}) == 1
    assert con.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == 360        # saved once
    # Higher-ranked stocks really did better: S00 rises fastest.
    close = pd.DataFrame({f"S{i:02d}": 100 * np.cumprod(np.full(60, 1 + 0.003 - i * 0.0002)) for i in range(30)},
                         index=days)
    close["SPY"] = 100.0
    monkeypatch.setattr("tradingagents.research.prices.bars", lambda tickers, cache: {"Close": close})
    assert feedback.resolve(con, tmp_path) == 360
    assert feedback.resolve(con, tmp_path) == 0                                                 # never twice
    card = feedback.scorecard(con)
    assert card["matured"] == 360 and card["ic_mean"] > 0.9 and card["top_fifth_edge"] > 0
    assert feedback.calibration(con)["active"]
    assert not feedback.calibration(con, min_rows=1000)["active"]
    today = pd.DataFrame({"p_win": [0.7, 0.4], "expected_net": [0.02, -0.01]}, index=["X", "Y"])
    out = feedback.apply(today, feedback.calibration(con))
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
    assert T.restore_all(store.con) == 1 and len(T.build(Con(), tmp_path)["policy"]) == 2
