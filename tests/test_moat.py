"""Quantitative moat checks on hand-made statements in the shape ``history.financials`` returns."""

import pytest

from tradingagents.research.moat import moat

pytestmark = pytest.mark.unit

PERIODS = ["2025-09-30", "2024-09-30", "2023-09-30", "2022-09-30"]


def _statements(revenue, gross, operating, equity, debt, cash, fcf):
    def table(rows):
        return {"annual": {"periods": PERIODS, "rows": rows}}
    return {"income": table({"Total Revenue": revenue, "Gross Profit": gross, "Operating Income": operating}),
            "balance": table({"Stockholders Equity": equity, "Total Debt": debt, "Cash And Cash Equivalents": cash}),
            "cashflow": table({"Free Cash Flow": fcf})}


def test_a_wide_margin_high_return_grower_shows_strong_indicators():
    s = _statements(revenue=[400, 380, 360, 340], gross=[184, 175, 162, 150], operating=[120, 114, 108, 100],
                    equity=[200, 190, 180, 170], debt=[100, 100, 100, 100], cash=[50, 50, 50, 50],
                    fcf=[100, 95, 90, 80])
    out = moat(s)
    assert out["verdict"] == "strong indicators"
    assert out["passed"] == out["scored"] == 5
    assert out["years"] == "2022 to 2025"


def test_a_thin_margin_shrinking_business_shows_few_indicators():
    s = _statements(revenue=[300, 320, 340, 360], gross=[45, 50, 60, 70], operating=[6, 10, 15, 20],
                    equity=[500, 500, 500, 500], debt=[200, 200, 200, 200], cash=[20, 20, 20, 20],
                    fcf=[-5, 10, 12, 15])
    out = moat(s)
    assert out["verdict"] == "few indicators"
    assert out["passed"] == 0


def test_missing_lines_are_not_counted_either_way():
    # A bank: no gross profit and no operating income, only revenue and cash flow.
    s = _statements(revenue=[100, 90, 80, 70], gross=[], operating=[], equity=[50] * 4, debt=[500] * 4,
                    cash=[10] * 4, fcf=[20, 18, 15, 12])
    out = moat(s)
    by = {c["name"]: c["passed"] for c in out["checks"]}
    assert by["Pricing power"] is None and by["Returns on capital"] is None and by["Profitability"] is None
    assert out["scored"] == 2 and out["verdict"] == "not enough data"


def test_empty_statements_do_not_crash():
    out = moat({})
    assert out["verdict"] == "not enough data" and out["years"] is None
