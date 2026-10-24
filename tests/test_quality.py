"""The checklist beyond the moat: strength, profitability, valuation, on hand-made statements."""

import pandas as pd
import pytest

from tradingagents.research.quality import checklist, implied_growth, raw_closes

pytestmark = pytest.mark.unit

PERIODS = ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"]


def _statements(rows):
    def table(kind):
        return {"annual": {"periods": PERIODS, "rows": {k: v for (kd, k), v in rows.items() if kd == kind}}}
    return {kind: table(kind) for kind in ("income", "balance", "cashflow")}


def _base(over=None):
    rows = {("income", "Total Revenue"): [100, 95, 90, 85], ("income", "Gross Profit"): [50, 47, 44, 41],
            ("income", "Operating Income"): [30, 28, 26, 24], ("income", "Net Income"): [22, 20, 19, 17],
            ("income", "Interest Expense"): [2, 2, 2, 2], ("income", "EBITDA"): [36, 34, 32, 30],
            ("income", "Diluted EPS"): [2.2, 2.0, 1.9, 1.7],
            ("balance", "Total Debt"): [40, 40, 40, 40], ("balance", "Stockholders Equity"): [60, 55, 50, 45],
            ("balance", "Cash And Cash Equivalents"): [10, 10, 10, 10],
            ("balance", "Current Assets"): [50, 48, 46, 44], ("balance", "Current Liabilities"): [30, 30, 30, 30],
            ("cashflow", "Free Cash Flow"): [20, 18, 17, 15]}
    rows.update(over or {})
    return _statements(rows)


def _items(out, key):
    return {i["name"]: i for i in next(s for s in out["sections"] if s["key"] == key)["items"]}


def test_a_sound_company_passes_the_strength_tests():
    out = checklist(_base())
    st = _items(out, "strength")
    assert st["Debt to equity"]["status"] == "ok" and st["Interest coverage"]["value"].startswith("15.0×")
    assert st["Current ratio"]["status"] == "ok"
    assert st["High-rate downturn"]["value"] == "coverage would be 7.5×"   # 0.7 × 30 / (2 + 0.02 × 40)
    assert out["sections"][0]["answer"].startswith("Likely")


def test_the_flags_follow_the_rules():
    out = checklist(_base({("income", "Interest Expense"): [12, 12, 12, 12],
                             ("balance", "Current Assets"): [20, 20, 20, 20],
                             ("balance", "Total Debt"): [150, 150, 150, 150]}))
    st = _items(out, "strength")
    assert st["Debt to equity"]["status"] == "flag"          # 150 / 60 = 2.5
    assert st["Interest coverage"]["status"] == "flag"       # 30 / 12 = 2.5, under 3
    assert st["Current ratio"]["status"] == "flag"           # 20 / 30
    assert st["High-rate downturn"]["status"] == "flag"      # 21 / (12 + 3) = 1.4
    assert out["sections"][0]["answer"].startswith("At risk")


def test_margin_trends_are_called_expanding_stable_or_shrinking():
    out = checklist(_base({("income", "Net Income"): [15, 18, 19, 20]}))
    pr = _items(out, "profitability")
    assert pr["Gross margin trend"]["value"].startswith("expanding")      # 48.2% to 50%
    assert pr["Net margin trend"]["value"].startswith("shrinking") and pr["Net margin trend"]["status"] == "warn"


def test_implied_growth_recovers_the_rate_that_priced_the_cash_flows():
    fcf, g = 10.0, 0.07
    value, f = 0.0, fcf
    for t in range(1, 11):
        f *= 1 + g
        value += f / 1.09 ** t
    value += f * 1.025 / (0.09 - 0.025) / 1.09 ** 10
    assert implied_growth(value, fcf) == pytest.approx(g, abs=1e-6)
    assert implied_growth(value, -1.0) is None


def test_raw_closes_undo_the_dividend_adjustment():
    # Raw: 100 before the ex-date, 98 after a $2 dividend. Adjusted: earlier prices × (1 − 2/100).
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    adjusted = pd.Series([100 * 0.98, 100 * 0.98, 98.0], index=idx)
    raw = raw_closes(adjusted, [["2024-01-04", 2.0]])
    assert list(raw.round(6)) == [100.0, 100.0, 98.0]


def test_a_bank_skips_the_industrial_tests_and_valuation_uses_book():
    s = _statements({("income", "Total Revenue"): [100, 90, 80, 70], ("income", "Net Income"): [30, 27, 24, 21],
                       ("balance", "Stockholders Equity"): [200, 190, 180, 170],
                       ("balance", "Total Assets"): [2400, 2300, 2200, 2100]})
    out = checklist(s, {"marketCap": 400.0}, financial=True)
    assert "Not the usual tests" in _items(out, "strength")
    assert _items(out, "valuation")["Price to book"]["value"].startswith("2.00×")
