"""Regression tests for defects confirmed in the stock module's foundations (2026-10-24 review)."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import fundamentals as FU
from tradingagents.stocks import membership as MB
from tradingagents.stocks import panel as P
from tradingagents.stocks.targets import Thresholds, labels

pytestmark = pytest.mark.unit


def test_ranks_use_only_members_and_ignore_ticker_order():
    dates = pd.DatetimeIndex(["2020-01-02"])
    frame = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 100.0]], index=dates, columns=["A", "B", "C", "D", "OUT"])
    member = pd.DataFrame([[True, True, True, True, False]], index=dates, columns=frame.columns)
    r = P.rank_against(frame, member)
    assert list(r.iloc[0, :4]) == pytest.approx([-0.25, 0.0, 0.25, 0.5])   # percentile ranks; the outsider moved nobody
    assert r.iloc[0, 4] == pytest.approx(0.5)                                # above every member, still in range
    shuffled = P.rank_against(frame[["OUT", "D", "B", "A", "C"]], member[["OUT", "D", "B", "A", "C"]])
    pd.testing.assert_frame_equal(shuffled[frame.columns], r)
    ties = P.rank_against(pd.DataFrame([[5.0, 5.0, 1.0]], index=dates, columns=list("XYZ")),
                          pd.DataFrame(True, index=dates, columns=list("XYZ")))
    assert ties.iloc[0, 0] == ties.iloc[0, 1]


def test_asking_about_a_stock_changes_no_member_feature_and_it_never_trains(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_stocks import _stock_data

    # S15 is asked about but is not a member: it must be forecast, never trained on, and change no member's ranks.
    with_new = _stock_data(tmp_path, requested=("S00", "S15"))
    with_new.info["S15"] = {**with_new.info["S15"], "sp500": False, "added": None}
    without = _stock_data(tmp_path, requested=("S00",))
    without.info.pop("S15")
    a = P.build(with_new, 20, groups=("technical", "market"))
    b = P.build(without, 20, groups=("technical", "market"))
    feats = P.features_for(("technical", "market"))
    members = b.index[b["in_index"]]
    pd.testing.assert_frame_equal(a.loc[members, feats], b.loc[members, feats])
    s15 = a.xs("S15", level="ticker")
    assert len(s15) > 50 and not s15["in_index"].any() and s15["requested"].all()


def test_membership_intervals_include_removed_stocks():
    changes = pd.DataFrame({"date": pd.to_datetime(["2010-01-04", "2012-06-01", "2015-03-02"]),
                            "tickers": ["AAA,BBB", "AAA,CCC", "AAA,BBB"]})
    spans = MB.intervals(changes)
    assert spans["BBB"] == [(pd.Timestamp("2010-01-04"), pd.Timestamp("2012-06-01")), (pd.Timestamp("2015-03-02"), None)]
    m = MB.mask(spans, pd.DatetimeIndex(["2011-01-03", "2013-01-02"]), ["BBB", "CCC"])
    assert m.loc["2011-01-03", "BBB"] and not m.loc["2013-01-02", "BBB"] and m.loc["2013-01-02", "CCC"]
    assert MB.members_since(spans, "2013-01-01") == ["AAA", "BBB", "CCC"]   # CCC left in 2015: a member after 2013
    assert MB.members_since(spans, "2016-01-01") == ["AAA", "BBB"]


def test_missing_capex_leaves_free_cash_flow_unknown():
    facts = {"revenue": [["2020-01-01", "2020-12-31", "2021-02-01", "10-K", 100.0]],
             "ocf": [["2020-01-01", "2020-12-31", "2021-02-01", "10-K", 30.0]]}
    snap = FU.snapshot(facts, "2021-03-01")
    assert np.isnan(snap["fcf_y"]) and np.isnan(snap["fcf_margin"])


def test_valuation_puts_shares_and_price_on_the_same_split_basis():
    fund = pd.DataFrame({"shares_i": [100.0], "net_income_y": [40.0], "revenue_y": [400.0], "fcf_y": [20.0],
                         "equity_i": [200.0], "shares_end_ord": [float(date(2020, 6, 30).toordinal())]},
                        index=pd.DatetimeIndex(["2020-12-01"]))
    # A 4-for-1 split on 2020-08-31: the December price is post-split, so 100 shares are 400.
    v = FU.valuation(fund, pd.Series([2.5], index=fund.index), splits=[["2020-08-31", 4.0]])
    assert v.loc["2020-12-01", "earnings_yield"] == pytest.approx(40.0 / (400 * 2.5))


def test_the_deep_drawdown_barrier_is_expm1_of_the_negative_move():
    days = pd.bdate_range("2024-01-01", periods=140)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"S": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 140)))}, index=days)
    flat = pd.Series(1.0, index=days)
    t = days[119]
    out = labels(close, close, close, flat, flat, pd.DatetimeIndex([t]), 10, Thresholds()).loc[(t, "S")]
    line = np.expm1(-1.5 * out["sigma_h"])
    assert out["deep_drawdown"] == float(out["drawdown"] <= line)
    assert Thresholds().contract == "v2"


def test_cash_returns_missing_before_the_cash_fund_existed_leave_no_label():
    days = pd.bdate_range("2024-01-01", periods=140)
    close = pd.DataFrame({"S": np.linspace(100, 120, 140)}, index=days)
    cash = pd.Series(np.nan, index=days)
    t = days[119]
    out = labels(close, close, close, cash, pd.Series(1.0, index=days), pd.DatetimeIndex([t]), 10)
    assert np.isnan(out.loc[(t, "S"), "label"]) and not np.isnan(out.loc[(t, "S"), "label_rel"])


def test_position_adverse_excursion_starts_after_the_entry_close_and_includes_zero():
    days = pd.bdate_range("2024-01-01", periods=140)
    path = np.r_[np.full(120, 100.0), 80.0, 110.0, 120.0, np.full(17, 125.0)]   # t = day 119; t+1 = 80 (entry)
    close = pd.DataFrame({"S": path}, index=days)
    flat = pd.Series(1.0, index=days)
    t = days[119]
    out = labels(close, close, close, flat, flat, pd.DatetimeIndex([t]), 5).loc[(t, "S")]
    assert out["mae_position"] == 0.0           # never below the 80 entry afterwards: zero, not the drop to 80
    assert out["drawdown"] == pytest.approx(-0.2)   # the descriptive close-t event still sees it


def test_daily_economic_series_wait_for_publication():
    from tradingagents.stocks import macro

    assert macro.LAG_BDAYS == {"DFF": 1, "T10Y2Y": 2, "BAA10Y": 2}


def test_missing_risk_blocks_approval(tmp_path, monkeypatch):
    from tradingagents.ledger.db import now_utc
    from tradingagents.ledger.store import Store
    from tradingagents.stocks import ideas

    store = Store(tmp_path)
    day = date(2026, 10, 1)
    store.ledger.contribute(1000.0, day, sleeve="trading", settle_date=day)
    close = pd.DataFrame({t: np.linspace(40, 47.3, 260) for t in ["AAA", "SPY"]}, index=pd.bdate_range(end="2026-10-23", periods=260))
    monkeypatch.setattr(ideas, "_prices", lambda cache_dir, names: close)
    store.con.execute("INSERT INTO stock_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      ("r1", now_utc(), "2026-10-23", "2026-10-26", "{}", "{}", 1, "[20]", "x", "x", "thr", "{}", "{}",
                       "{}", "{}", "{}", 1.0))
    payload = {"forecast": {"net": 0.05, "mean": 0.053, "cost_rt": 0.003, "p_large_loss": None},
               "evidence_status": {"return": "backed"}}
    store.con.execute("INSERT INTO stock_forecasts VALUES (?, ?, ?, ?, ?, ?)", ("r1", "AAA", 20, "ok", "[]", json.dumps(payload)))
    out = ideas.make(store.con, store, tmp_path, "r1", 20, as_of=date(2026, 10, 24))
    assert out["lines"][0]["action"].startswith("watch (risk estimate missing") and not out["lines"][0].get("order")
