"""Dividends, splits and share-count changes on hand-made Yahoo-shaped histories."""

from datetime import date

import pytest

from tradingagents.research.corporate import summarize

pytestmark = pytest.mark.unit

AS_OF = date(2026, 10, 24)


def _quarterly(start_year, end_year, amounts):
    """Four payments a year; ``amounts[year]`` is each payment that year."""
    return [[f"{y}-{m:02d}-15", amounts[y]] for y in range(start_year, end_year + 1) for m in (2, 5, 8, 11)
            if f"{y}-{m:02d}-15" < AS_OF.isoformat()]


def test_a_steady_raiser_counts_its_streak_yield_and_growth():
    amounts = {y: 0.10 * 1.05 ** (y - 2010) for y in range(2010, 2027)}
    out = summarize({"dividends": _quarterly(2010, 2026, amounts)}, AS_OF, price=10.0)
    dv = out["dividends"]
    assert dv["raised_years"] == 15                     # 2011 through 2025, each above the year before
    assert dv["growth_5y"] == pytest.approx(0.05)
    assert dv["ttm"] == pytest.approx(amounts[2026] * 3 + amounts[2025])
    assert dv["yield"] == pytest.approx(dv["ttm"] / 10.0)
    assert dv["by_year"][-1] == [2026, pytest.approx(amounts[2026] * 3), True]
    assert not [e for e in out["events"] if e["kind"] == "cut"]


def test_a_fifth_payment_in_a_calendar_year_is_not_a_raise_or_a_cut():
    # Same $0.50 every quarter, but one payment slipped from late 2019 into early 2020.
    pays = [[d, 0.5] for d in ("2019-02-01", "2019-05-01", "2019-08-01", "2020-01-02", "2020-02-01", "2020-05-01",
                               "2020-08-01", "2020-11-01", "2021-02-01", "2021-05-01", "2021-08-01", "2021-11-01")]
    out = summarize({"dividends": pays}, AS_OF)
    assert out["dividends"]["raised_years"] == 0
    assert not [e for e in out["events"] if e["kind"] == "cut"]


def test_a_cut_and_a_suspension_are_events():
    amounts = {2021: 0.36, 2022: 0.36, 2023: 0.125, 2024: 0.125}
    pays = [p for p in _quarterly(2021, 2024, amounts) if p[0] <= "2024-08-15"]
    out = summarize({"dividends": pays}, AS_OF)
    kinds = [e["kind"] for e in out["events"]]
    assert "cut" in kinds and "suspended" in kinds
    assert out["dividends"]["ttm"] == 0 and out["dividends"]["yield"] is None


def test_share_counts_are_restated_for_splits_before_calling_anything_dilution():
    # 100 shares, then a 4-for-1 split: 400 shares is the same company, not 300% dilution.
    shares = [["2019-12-01", 100.0], ["2020-12-01", 380.0], ["2021-12-01", 400.0]]
    out = summarize({"shares": shares, "splits": [["2020-08-31", 4.0]]}, AS_OF)
    by_year = {r["year"]: r for r in out["shares"]}
    assert by_year[2019]["shares"] == 400.0
    assert by_year[2020]["change"] == pytest.approx(-0.05)
    assert by_year[2021]["change"] == pytest.approx(400 / 380 - 1)
    kinds = {e["kind"] for e in out["events"]}
    assert kinds == {"split", "buyback", "dilution"}
    assert out["splits"][0]["label"] == "4-for-1 split"
    assert out["shares_net"]["change"] == pytest.approx(0.0)


def test_a_one_day_spike_in_reported_shares_is_ignored():
    shares = [["2024-12-01", 100.0], ["2025-10-01", 100.0], ["2025-11-01", 160.0], ["2025-12-01", 100.0]]
    out = summarize({"shares": shares}, AS_OF)
    assert out["shares"][0]["change"] == pytest.approx(0.0) and not out["events"]


def test_reverse_split_label_and_funds_skip_share_counts_and_cuts():
    out = summarize({"splits": [["2023-01-05", 0.1]], "shares": [["2022-12-01", 1.0], ["2023-12-01", 9.0]],
                     "dividends": [["2022-03-01", 1.0], ["2023-03-01", 0.5]]}, AS_OF, fund=True)
    assert out["splits"][0]["label"] == "1-for-10 reverse split"
    assert out["shares"] == [] and out["fund"]
    assert {e["kind"] for e in out["events"]} == {"split", "dividend", "suspended"}   # no cut, though 2023 halved


def test_nothing_on_record():
    out = summarize({}, AS_OF)
    assert out["dividends"] is None and out["events"] == [] and out["shares_net"] is None
