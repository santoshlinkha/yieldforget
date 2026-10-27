"""Triple-barrier policy labels: timing after the cutoff, gaps, no-fills, touches, delistings, incomplete windows."""

import math

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks.policies import POLICY_SET, PolicySpec, label, simulate

pytestmark = pytest.mark.unit

SIG = 0.01                       # daily volatility: sigma_H = 0.01 * sqrt(20) = 4.47%
SH = SIG * math.sqrt(20)


def _run(path, spec, low=None, t=0, cost=0.0, cash=0.0, div=None):
    px = np.asarray(path, dtype=float)[:, None]
    lo = px.copy() if low is None else np.asarray(low, dtype=float)[:, None]
    dv = np.zeros_like(px) if div is None else np.asarray(div, dtype=float)[:, None]
    cash_ret = np.full(len(px), cash)
    return simulate(px, lo, dv, cash_ret, np.array([t]), np.array([0]), np.array([SIG]), np.array([cost]), spec).iloc[0]


def test_entry_is_after_the_cutoff_and_the_stop_sells_at_the_next_close_with_the_gap():
    # Decision at t=0 (close 100). Entry at the close of 1 (100). Close 3 falls through the stop; the sale is the close of 4,
    # which gapped further down: the loss is the gap's, not the stop level's.
    path = [100, 100, 101, 100 * math.exp(-SH) - 0.5, 90, 90] + [90] * 20
    r = _run(path, POLICY_SET["next_close"])
    assert r.entry_idx == 1 and r.entry_px == 100
    assert r.outcome == "stop" and r.held == 3 and r.net == pytest.approx(-0.10)
    assert r.label_end == 4


def test_a_rise_on_the_entry_session_is_not_traded_and_expiry_sells_at_the_deadline():
    path = [100, 100] + [100.5] * 25
    r = _run(path, POLICY_SET["next_close"], cost=0.002, cash=0.0001)
    assert r.outcome == "expiry" and r.held == 20
    assert r.net == pytest.approx(0.005 - 0.002)
    assert r.excess == pytest.approx(r.net - ((1.0001) ** 20 - 1))


def test_the_exit_is_measured_from_the_actual_fill_not_the_signal_price():
    spec = POLICY_SET["limit_dip"]
    lim = 100 * math.exp(-0.5 * SIG * math.sqrt(5))
    low = [100, 100, lim * 0.99] + [lim] * 30
    path = [100, 100, lim] + [lim * math.exp(SH) + 0.01] * 30
    r = _run(path, spec, low=low)
    assert r.entry_idx == 2 and r.entry_px == pytest.approx(lim)
    assert r.outcome == "target" and r.net == pytest.approx((lim * math.exp(SH) + 0.01) / lim - 1)


def test_a_touch_is_not_a_fill_and_no_fill_stays_in_cash():
    spec = POLICY_SET["limit_dip"]
    lim = 100 * math.exp(-0.5 * SIG * math.sqrt(5))
    low = [100] + [lim] * 40                                  # touches the limit every day, never goes through
    r = _run([100] * 41, spec, low=low, cash=0.001)
    assert r.outcome == "nofill" and r.net == 0 and r.excess == 0 and r.label_end == 5


def test_a_window_that_has_not_finished_has_no_label():
    r = _run([100] * 10, POLICY_SET["next_close"])
    assert r.outcome == "incomplete" and np.isnan(r.net)
    lim_open = _run([100] * 3, POLICY_SET["limit_dip"])        # the limit's own window is not over
    assert lim_open.outcome == "incomplete"


def test_a_delisting_is_kept_and_sold_at_the_last_price_not_dropped():
    path = [100, 100, 95, 80, np.nan, np.nan] + [np.nan] * 20
    r = _run(path, PolicySpec("x", "next_close", profit_k=10, risk_k=10))
    assert r.outcome == "delisted" and r.net == pytest.approx(-0.20)


def test_holding_from_now_has_no_entry_cost_and_starts_at_todays_close():
    path = [100] + [101] * 25
    r = _run(path, POLICY_SET["hold_now"], cost=0.004)
    assert r.entry_idx == 0 and r.entry_px == 100
    assert r.net == pytest.approx(0.01 - 0.002)               # only the exit half of the cost


def test_dividends_count_and_label_uses_executable_lows():
    idx = pd.bdate_range("2026-01-01", periods=30)
    px = pd.DataFrame({"AAA": 100.0}, index=idx)
    div = pd.DataFrame({"AAA": 0.0}, index=idx)
    div.iloc[5] = 1.0
    rows = pd.DataFrame({"ewma_vol_daily": [SIG], "cost_rt": [0.0]},
                        index=pd.MultiIndex.from_tuples([(idx[0], "AAA")], names=["date", "ticker"]))
    out = label(rows, px, px, px, div, pd.Series(100.0, index=idx), POLICY_SET["next_close"])
    assert out.iloc[0].outcome == "expiry" and out.iloc[0].net == pytest.approx(0.01)
    assert out.iloc[0].entry_date == idx[1] and out.iloc[0].label_end_date == idx[21]
