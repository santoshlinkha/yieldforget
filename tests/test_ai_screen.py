"""The AI's own first screen: a fact sheet of facts only, and a ranking kept to stocks on the sheet. No model calls."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks import ai_screen as A

pytestmark = pytest.mark.unit


def _rows():
    return [{"ticker": t, "sector": "Tech", "price": 100.0, "analyst_mean": 1.9, "analysts": 30, "target_upside": 0.1,
             "quality_score": 80.0, "moat_passed": 4, "forward_pe": 20.0, "implied_growth": 0.1, "rank": i + 1}
            for i, t in enumerate(["AAA", "BBB", "CCC"])]


def test_the_fact_sheet_holds_facts_under_the_four_lenses_and_no_model_scores():
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = pd.DataFrame({t: np.linspace(50, 100, 300) for t in ("AAA", "BBB", "CCC")}, index=idx)
    sheet, tickers = A.fact_sheet(_rows(), close)
    assert tickers == ["AAA", "BBB", "CCC"] and sheet.count("\n") == 2
    assert all(k in sheet for k in ("TECH ret1m", "SENT consensus", "FUND quality"))
    assert "rank" not in sheet.lower().replace("ranked", "")              # no screen or model rank given to the AI


def test_the_ranking_keeps_only_stocks_on_the_sheet_once_each_in_order():
    class Gateway:
        def run(self, **kw):
            assert kw["task"] == "ai_screen" and "FACT SHEET" in kw["inputs"]["prompt"]
            pick = lambda t: {"ticker": t, "technical": "", "sentiment": "", "news": "", "fundamentals": "", "why": ""}  # noqa: E731
            return {"picks": [pick("ccc"), pick("ZZZ"), pick("AAA"), pick("CCC")], "note": "n"}, "paid"

    out = A.screen(_rows(), None, "2026-10-28", {"llm_provider": "openrouter", "quick_think_llm": "m"}, Gateway())
    assert [p["ticker"] for p in out["picks"]] == ["CCC", "AAA"] and [p["rank"] for p in out["picks"]] == [1, 2]
    assert out["considered"] == 3 and out["model"] == "openrouter/m"
