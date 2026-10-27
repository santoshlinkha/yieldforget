"""The SEC 8-K archive and the rebuilt analyst consensus use only what was public by each close."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import consensus as C
from tradingagents.stocks import news_archive as N

pytestmark = pytest.mark.unit

SESSIONS = pd.bdate_range("2026-01-05", periods=40)


def test_a_filing_after_the_close_is_known_only_from_the_next_session():
    assert N._ny("2026-08-04T20:08:24.000Z") == ("2026-08-04", 1)          # 16:08 New York (EDT): after the close
    assert N._ny("2026-01-06T14:30:00.000Z") == ("2026-01-06", 0)          # 09:30 New York (EST): before
    known = N.known_session(pd.Series(["2026-01-06", "2026-01-06", "2026-01-10"]), pd.Series([0, 1, 0]), SESSIONS)
    assert list(known) == [SESSIONS[1], SESSIONS[2], SESSIONS[5]]          # Saturday's filing: Monday


def test_news_features_count_only_filings_known_by_each_close(tmp_path):
    con = N.connect(tmp_path)
    rows = [("a1", "1", "AAA", "8-K", "2026-01-06", "2026-01-06T21:00:00Z", "2026-01-06", 1, "2.02,9.01", "", ""),
            ("a2", "1", "AAA", "8-K", "2026-01-12", "2026-01-12T15:00:00Z", "2026-01-12", 0, "5.02", "", "")]
    con.executemany("INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    f = N.features(tmp_path, SESSIONS[[1, 2, 5, 30]], ["AAA"], SESSIONS)
    by = f.xs("AAA", level="ticker")
    assert by["n8k_20"].tolist() == [0, 1, 2, 0]           # the 2.02 counts from the 7th, not the evening of the 6th
    assert by["since_results"].tolist()[1:3] == [0, 3] and np.isnan(by["since_results"].iloc[0])
    assert by["exec_change_60"].iloc[2] == 1


def _actions(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows], name="GradeDate")
    return pd.DataFrame([r[1:] for r in rows], index=idx,
                        columns=["Firm", "ToGrade", "FromGrade", "Action", "priceTargetAction", "currentPriceTarget",
                                 "priorPriceTarget"])


def test_consensus_uses_each_firms_latest_view_and_targets_in_todays_shares():
    s = pd.bdate_range("2024-01-01", periods=300)
    close = pd.Series(100.0, index=s)                       # split-adjusted (today's shares)
    acts = _actions([
        ("2024-01-02 15:00", "A", "Buy", "Hold", "up", "Raises", 1100.0, 900.0),     # before a 10:1 split: $110 now
        ("2024-01-03 15:00", "B", "Hold", "Hold", "main", "Maintains", 120.0, 120.0),  # already in new shares
        ("2024-01-04 15:00", "C", "Sell", "Hold", "down", "Lowers", 80.0, 100.0),
        ("2024-03-01 15:00", "C", "Buy", "Sell", "up", "Raises", 130.0, 80.0),        # C changes its mind later
    ])
    splits = [["2024-06-03", 10.0]]
    pos = np.array([s.get_loc(pd.Timestamp("2024-01-03")), s.get_loc(pd.Timestamp("2024-01-08")),
                    s.get_loc(pd.Timestamp("2024-03-05"))])
    f = C.ticker_features(acts, None, pos, s, close, splits)
    assert f["an_firms"].tolist() == [1, 3, 3]              # an action counts from the session after its day
    assert f["an_score"].tolist() == [1.0, 0.0, pytest.approx(2 / 3)]
    assert f["an_net_90"].iloc[1] == 0 and f["an_net_90"].iloc[2] == 1
    assert f["an_pt_upside"].iloc[1] == pytest.approx(110 / 100 - 1)    # median of 110, 120, 80
    assert f["an_pt_upside"].iloc[2] == pytest.approx(120 / 100 - 1)    # median of 110, 120, 130


def test_earnings_after_the_close_count_from_the_next_session():
    s = pd.bdate_range("2024-01-01", periods=600)
    idx = pd.DatetimeIndex(["2024-01-25 16:05", "2024-04-25 16:05", "2024-07-25 16:05", "2024-10-24 16:05",
                            "2025-01-23 16:05"]).tz_localize("America/New_York")
    reports = pd.DataFrame({"EPS Estimate": [1.0, 1.0, 1.0, 1.0, 1.0], "Reported EPS": [1.1, 0.9, 1.2, 1.3, 1.5],
                            "Surprise(%)": [10.0, -10.0, 20.0, 30.0, 50.0]}, index=idx)
    pos = np.array([s.get_loc(pd.Timestamp("2024-01-25")), s.get_loc(pd.Timestamp("2024-01-26")),
                    s.get_loc(pd.Timestamp("2025-01-24"))])
    f = C.ticker_features(None, reports, pos, s, None)
    assert np.isnan(f["eps_surprise_last"].iloc[0]) and f["eps_surprise_last"].iloc[1] == 10.0
    assert f["eps_beats_4"].iloc[2] == 3 and f["eps_growth_yoy"].iloc[2] == pytest.approx(1.5 / 1.1 - 1)
