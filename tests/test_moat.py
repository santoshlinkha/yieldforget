"""Quantitative moat checks on hand-made statements in the shape ``history.financials`` returns."""

import pytest

from tradingagents.research.moat import moat

pytestmark = pytest.mark.unit

PERIODS = ["2025-09-30", "2024-09-30", "2023-09-30", "2022-09-30"]


def _table(rows, periods=PERIODS):
    return {"annual": {"periods": periods, "rows": rows}}


def _statements(revenue, gross, operating, equity, debt, cash, fcf, pretax=(), tax=()):
    return {"income": _table({"Total Revenue": revenue, "Gross Profit": gross, "Operating Income": operating,
                              "Pretax Income": list(pretax), "Tax Provision": list(tax)}),
            "balance": _table({"Stockholders Equity": equity, "Total Debt": debt, "Cash And Cash Equivalents": cash}),
            "cashflow": _table({"Free Cash Flow": fcf})}


def _strong(**over):
    rows = dict(revenue=[400, 380, 360, 340], gross=[184, 175, 162, 150], operating=[120, 114, 108, 100],
                equity=[200, 190, 180, 170], debt=[100, 100, 100, 100], cash=[50, 50, 50, 50],
                fcf=[100, 95, 90, 80])
    return _statements(**{**rows, **over})


def test_a_wide_margin_high_return_grower_shows_strong_indicators():
    out = moat(_strong())
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
    s = _statements(revenue=[100, 90, 80, 70], gross=[], operating=[], equity=[50] * 4, debt=[500] * 4,
                    cash=[10] * 4, fcf=[20, 18, 15, 12])
    out = moat(s)
    by = {c["name"]: c["passed"] for c in out["checks"]}
    assert by["Pricing power"] is None and by["Returns on capital"] is None and by["Profitability"] is None
    assert out["scored"] == 2 and out["verdict"] == "not enough data"


def test_empty_statements_do_not_crash():
    out = moat({})
    assert out["verdict"] == "not enough data" and out["years"] is None


def test_the_bar_depends_on_the_sector():
    # A 46% gross margin is wide for a grocer and narrow for software.
    assert {c["name"]: c["passed"] for c in moat(_strong(), "Consumer Defensive")["checks"]}["Pricing power"]
    tech = moat(_strong(), "Technology")
    assert not {c["name"]: c["passed"] for c in tech["checks"]}["Pricing power"]
    assert tech["basis"] == "bars for Technology companies"
    assert moat(_strong(), "Technology", "Consumer Electronics")["basis"] == "bars for Consumer Electronics companies"
    assert moat(_strong(), "Financial Services")["basis"] == "general bars (Financial Services has none of its own)"


def _roic(out):
    return next(c for c in out["checks"] if c["name"] == "Returns on capital")


def test_roic_uses_the_years_own_tax_rate():
    assert "tax rate 21%" in _roic(moat(_strong()))["value"]
    taxed = moat(_strong(pretax=[100] * 4, tax=[10] * 4))
    assert "tax rate 10%" in _roic(taxed)["value"]


def test_only_cash_beyond_what_the_business_needs_comes_off_capital():
    # Two years. Capital is equity + debt − (cash − 2% of revenue kept to run the business),
    # averaged over a year's start and end; the earliest year has only its end.
    two = ["2025-09-30", "2024-09-30"]
    s = {"income": _table({"Total Revenue": [1000, 1000], "Operating Income": [100, 100]}, two),
         "balance": _table({"Stockholders Equity": [300, 200], "Total Debt": [100, 100],
                            "Cash And Cash Equivalents": [120, 20]}, two),
         "cashflow": _table({"Free Cash Flow": [1, 1]}, two)}
    later = (300 + 100 - (120 - 20)) / 2 + (200 + 100 - 0) / 2  # 2024's cash is all needed
    earlier = 200 + 100
    low = 100 * 0.79 / max(later, earlier)
    mean = (100 * 0.79 / later + 100 * 0.79 / earlier) / 2
    assert _roic(moat(s))["value"] == f"ROIC {mean:.0%} on average, lowest {low:.0%} (tax rate 21%)"


def test_years_line_up_by_date_when_a_statement_has_more_of_them():
    s = _strong()
    s["cashflow"] = _table({"Free Cash Flow": [100, 95, 90, 80, -50]}, PERIODS + ["2021-09-30"])
    cash = next(c for c in moat(s)["checks"] if c["name"] == "Cash generation")
    assert cash["passed"] and cash["value"] == "free cash flow positive in 4 of 4 years"


def _bank(net_income, equity, assets, shares, revenue):
    return {"income": _table({"Total Revenue": revenue, "Net Income": net_income}),
            "balance": _table({"Stockholders Equity": equity, "Total Assets": assets,
                               "Ordinary Shares Number": shares, "Total Debt": [900] * 4}),
            "cashflow": _table({"Free Cash Flow": [-30, 40, -20, 10]})}


def test_a_bank_gets_bank_checks_instead_of_margins_and_roic():
    s = _bank(net_income=[30, 28, 25, 22], equity=[200, 185, 170, 160], assets=[2400, 2300, 2200, 2100],
              shares=[10, 10, 10, 10], revenue=[100, 95, 90, 85])
    out = moat(s, "Financial Services", "Banks - Diversified")
    names = [c["name"] for c in out["checks"]]
    assert names == ["Returns on equity", "Returns on assets", "Profitability", "Book value growth", "Steady growth"]
    assert out["verdict"] == "strong indicators" and out["passed"] == 5
    assert out["basis"].startswith("bank and insurer checks")


def test_a_low_return_bank_does_not_pass_on_size_alone():
    s = _bank(net_income=[10, 9, 9, 8], equity=[200, 195, 190, 185], assets=[2400, 2300, 2200, 2100],
              shares=[10, 10, 10, 10], revenue=[100, 95, 90, 85])
    by = {c["name"]: c["passed"] for c in moat(s, "Financial Services")["checks"]}
    assert by["Returns on equity"] is False and by["Returns on assets"] is False
