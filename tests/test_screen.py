"""The quality and Wall Street screen: scores, consensus, ranking, and reading without fetching."""

import pandas as pd
import pytest

from tradingagents.stocks import screen

pytestmark = pytest.mark.unit


def test_consensus_runs_from_strong_buy_to_strong_sell():
    a = {"recommendations": [{"period": "0m", "strongBuy": 6, "buy": 18, "hold": 14, "sell": 3, "strongSell": 3}]}
    mean, n, label = screen._consensus(a)
    assert n == 44 and mean == pytest.approx((6 + 36 + 42 + 12 + 15) / 44) and label == "Hold"   # 2.52: just past Buy
    assert screen._consensus({"recommendations": []}) == (None, 0, None)


def test_a_row_combines_the_checklist_and_wall_street():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_quality import _base

    fin = _base()
    closes = pd.Series([40.0, 50.0], index=pd.to_datetime(["2026-10-22", "2026-10-23"]))
    analysts = {"recommendations": [{"period": "0m", "strongBuy": 10, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}],
                "price_targets": {"mean": 60.0}}
    row = screen.score_row("AAA", {"longName": "Aaa", "sector": "Technology", "marketCap": 1e9}, fin, closes, [], analysts)
    assert 0 <= row["quality_score"] <= 100 and row["quality_tests"] > 5
    assert row["analyst_mean"] == 1.0 and row["analyst_label"] == "Strong buy" and row["target_upside"] == pytest.approx(0.2)


def test_ranking_is_stable_under_ties_and_ranks_missing_measures_on_the_rest():
    rows = [{"ticker": t, "quality_score": q, "analyst_mean": a, "target_upside": u}
            for t, q, a, u in (("BBB", 80, 2.0, 0.1), ("AAA", 80, 2.0, 0.1), ("CCC", 50, 3.0, None), ("DDD", 90, 1.5, 0.2))]
    out = screen.rank(rows)
    assert [r["ticker"] for r in out] == ["DDD", "AAA", "BBB", "CCC"]       # the tie breaks by ticker
    assert out[-1]["note"] == "ranked on 2 of 3 measures"
    assert [r["quality_rank"] for r in out] == [1, 2, 2, 4]                  # equal quality scores share a rank
    assert out[0]["combined_score"] == 100.0
    assert [r["ticker"] for r in screen.rank(list(reversed(rows)))] == ["DDD", "AAA", "BBB", "CCC"]


def test_reading_the_screen_fetches_nothing(tmp_path, monkeypatch):
    def no_fetch(*a, **k):
        raise AssertionError("reading must not fetch")

    monkeypatch.setattr("tradingagents.dataflows.vendors.yahoo.history.profile", no_fetch)
    assert screen.latest(tmp_path) is None
    (tmp_path / "screen").mkdir()
    (tmp_path / "screen" / "quality_2026-10-23.json").write_text('{"rows": [], "stocks": 0}')
    assert screen.latest(tmp_path)["stocks"] == 0
    (tmp_path / "screen" / "quality_2026-10-24.json").write_text(
        '{"rows": [{"ticker": "AAA", "quality_score": 50, "analyst_mean": 2.0, "target_upside": 0.1, "rank": 9}], "stocks": 1}')
    row = screen.latest(tmp_path)["rows"][0]
    assert row["rank"] == 1 and row["quality_rank"] == 1                     # older files gain the new columns
