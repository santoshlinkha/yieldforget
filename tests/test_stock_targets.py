"""The stock targets mean what their definitions say, on hand-made price paths."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.stocks.targets import Thresholds, current_trend, labels

pytestmark = pytest.mark.unit

H = 10
DAYS = pd.bdate_range("2024-01-01", periods=140)
T = DAYS[119]                      # decision date: 120 sessions of history before it


def _frames(after, before_noise=0.01, high_pad=0.002):
    """A stock with a calm, known history, then ``after`` (closes for the sessions after T)."""
    rng = np.random.default_rng(0)
    hist = 100 * np.exp(np.cumsum(rng.normal(0, before_noise, 119)))
    hist = hist / hist[-1] * 100                       # close at T = 100
    path = np.concatenate([hist, [100.0], after])
    close = pd.DataFrame({"S": path[:len(DAYS)]}, index=DAYS)
    high, low = close * (1 + high_pad), close * (1 - high_pad)
    return close, high, low


def _run(close, high, low, th=Thresholds()):
    flat = pd.Series(1.0, index=DAYS)
    out = labels(close, high, low, cash=flat, spy=flat, dates=pd.DatetimeIndex([T]), horizon=H, th=th)
    return out.loc[(T, "S")]


def test_a_steady_climb_is_an_up_trend():
    close, high, low = _frames(100 * np.exp(np.linspace(0.01, 0.12, 20)))
    row = _run(close, high, low)
    assert row["trend_state"] == 0 and row["ret_raw"] == pytest.approx(np.log(close.iloc[119 + H, 0] / 100))
    assert row["up_break"] == 1 and row["large_loss"] == 0


def test_a_volatile_round_trip_is_unclear_not_sideways():
    swing = [110, 92, 112, 90, 111, 91, 109, 93, 105, 100] + [100] * 10
    close, high, low = _frames(np.array(swing, float))
    row = _run(close, high, low)
    assert abs(row["ret_raw"]) < 1e-9
    assert row["trend_state"] == 3                     # unclear: ended flat, but the range was huge


def test_a_quiet_flat_period_is_sideways():
    close, high, low = _frames(np.full(20, 100.0) + np.tile([0.05, -0.05], 10), before_noise=0.02)
    assert _run(close, high, low)["trend_state"] == 2


def test_a_dip_that_recovers_is_a_drawdown_but_not_a_terminal_loss():
    path = np.array([97, 90, 84, 82, 86, 92, 96, 99, 100, 101] + [101] * 10, float)
    close, high, low = _frames(path)
    row = _run(close, high, low)
    assert row["drawdown"] == pytest.approx(82 * (1 - 0.002) / 100 - 1)
    assert row["deep_drawdown"] == 1 and row["large_loss"] == 0


def test_one_bar_through_both_barriers_is_recorded_without_inventing_an_order():
    close, high, low = _frames(np.full(20, 100.0))
    high.iloc[121, 0], low.iloc[121, 0] = 130.0, 70.0  # one wild bar two sessions after T
    row = _run(close, high, low)
    assert row["up_break"] == 1 and row["down_break"] == 1 and row["both_same_bar"] == 1
    assert row["failed_up"] == 1 and row["failed_down"] == 1   # both closed back inside the range


def test_the_tradable_return_enters_at_the_next_close():
    after = np.array([110.0] + [121.0] * 19)            # the jump happens on the first session after T
    close, high, low = _frames(after)
    row = _run(close, high, low)
    assert row["label"] == pytest.approx(np.log(121 / 110))     # bought at 110, not at T's 100
    assert row["ret_raw"] == pytest.approx(np.log(121 / 100))   # the path event still sees the jump


def test_nothing_after_the_cutoff_shapes_the_barriers_or_the_scale():
    close, high, low = _frames(np.full(20, 100.0))
    base = _run(close, high, low)
    close2, high2, low2 = close.copy(), high.copy(), low.copy()
    close2.iloc[120:] *= 3; high2.iloc[120:] *= 3; low2.iloc[120:] *= 3
    moved = _run(close2, high2, low2)
    for k in ("upper", "lower", "sigma_h"):
        assert moved[k] == pytest.approx(base[k])


def test_labels_past_the_data_are_missing_not_zero():
    close, high, low = _frames(np.full(20, 100.0))
    # only 5 sessions after T, against a horizon of 10
    close = close.iloc[:125]; high = high.iloc[:125]; low = low.iloc[:125]
    flat = pd.Series(1.0, index=close.index)
    out = labels(close, high, low, flat, flat, pd.DatetimeIndex([T]), H).loc[(T, "S")]
    assert np.isnan(out["label"]) and np.isnan(out["up_break"]) and np.isnan(out["trend_state"])


def test_thresholds_have_a_stable_version():
    assert Thresholds().version() == Thresholds().version() != Thresholds(trend_k=0.6).version()


def test_current_trend_is_detection_from_the_past():
    up = pd.DataFrame({"S": np.linspace(50, 150, 300)}, index=pd.bdate_range("2023-01-02", periods=300))
    assert current_trend(up).iloc[-1, 0] == 1 and current_trend(-up + 300).iloc[-1, 0] == -1
