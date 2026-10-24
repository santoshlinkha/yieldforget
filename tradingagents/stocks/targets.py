"""What the stock models forecast, defined before any training, with frozen thresholds.

Timing. A forecast is made after the close of its *decision date* t, from data
known by then. It cannot trade at that close: the **tradable return** enters at
the next session's close (t+1) and exits ``h`` sessions later (t+1+h), log, above
cash, and separately above SPY (benchmark-relative). The **path events** describe
the ``h`` sessions after the cutoff, t+1 to t+h, measured from the close at t:

- **Scale.** sigma_h = the stock's daily EWMA volatility at t (RiskMetrics, lambda
  0.94) times sqrt(h). Every threshold is a multiple of it, so "a big move" means
  the same for a utility and for a chip maker, and it is known at t.
- **Terminal return** r = log(close[t+h] / close[t]), raw (not above cash).
- **Drawdown from entry** = min(low[t+1..t+h]) / close[t] - 1: the worst point a
  holder from t saw. It is a different thing from the terminal loss: a stock can
  fall 15% and recover by the end.
- **Large loss** (terminal): r <= -loss_k * sigma_h.
- **Deep drawdown**: drawdown from entry <= expm1(-drawdown_k * sigma_h).
- **Breakout barriers** come from the previous trading range, known at t: the
  highest high and lowest low of the last ``range_window`` sessions through t.
  An upward breakout is any high in t+1..t+h above the upper barrier; a downward
  one any low below the lower. Both can happen; they are separate events, not one
  classification. A **failed** breakout crossed its barrier but closed the period
  back inside it (close[t+h] at or below the upper / at or above the lower).
  Daily bars do not say which came first when one bar crosses both barriers:
  ``both_same_bar`` records that, and no ordering is invented.
- **Trend state** (exactly one of four, covering every path):
  *up*: r >= trend_k * sigma_h and path efficiency >= trend_efficiency, where
  efficiency = |r| / sum of |daily log returns| (1 = a straight line);
  *down*: the mirror image;
  *sideways*: |r| <= sideways_k * sigma_h and the whole high-low range of the
  period within sideways_range_k * sigma_h of the start (so a volatile round trip
  that ends where it began is not sideways);
  *unclear*: everything else.

Large loss, deep drawdown and breakouts overlap the trend states freely.

Detecting a trend that already exists (``current_trend``, from moving averages at t)
is a feature and a display, never a forecast.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

EWMA_LAMBDA = 0.94
TREND_STATES = ("up", "down", "sideways", "unclear")
EVENTS = ("up_break", "down_break", "large_loss", "deep_drawdown")
# Risks during the actual position (entry at the next close): what risk gates use.
POSITION_EVENTS = ("pos_large_loss", "pos_deep_drawdown")


@dataclass(frozen=True)
class Thresholds:
    """Every event threshold, as multiples of sigma_h. Frozen before a test period is scored:
    a study records ``version()``, and a forecast run says which version it used."""
    range_window: int = 20
    trend_k: float = 0.5
    trend_efficiency: float = 0.30
    sideways_k: float = 0.35
    sideways_range_k: float = 1.25
    loss_k: float = 1.5
    drawdown_k: float = 1.5
    contract: str = "v2"      # v2: position risks from the next-close entry; arithmetic barriers via expm1

    def version(self) -> str:
        return "thr-" + hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def ewma_vol(close: pd.DataFrame) -> pd.DataFrame:
    """Daily EWMA volatility of log returns, known at each close."""
    r = np.log(close).diff()
    return r.pow(2).ewm(alpha=1 - EWMA_LAMBDA, min_periods=60).mean().pow(0.5)


def _forward(frame: np.ndarray, rows: np.ndarray, h: int) -> np.ndarray:
    """``(len(rows), h, columns)``: the values on the h sessions after each row; NaN past the end."""
    n = frame.shape[0]
    idx = rows[:, None] + np.arange(1, h + 1)[None, :]
    out = frame[np.minimum(idx, n - 1)]
    out[idx >= n] = np.nan
    return out


def labels(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, cash: pd.Series, spy: pd.Series,
           dates: pd.DatetimeIndex, horizon: int, th: Thresholds = Thresholds()) -> pd.DataFrame:
    """Every target for each (date, ticker) in ``dates`` x ``close.columns``; NaN where the future
    has not happened yet. ``label`` (the tradable return above cash) is what return models fit."""
    sessions = close.index
    rows = sessions.get_indexer(dates)
    if (rows < 0).any():
        raise ValueError("decision dates must be sessions of the price frame")
    tickers = list(close.columns)
    c, hi, lo = (f.to_numpy(dtype=float) for f in (close, high.reindex_like(close), low.reindex_like(close)))
    logc = np.log(c)
    n = len(sessions)
    sigma_h = ewma_vol(close).to_numpy() [rows] * np.sqrt(horizon)

    start = c[rows]
    fwd_c = _forward(c, rows, horizon)
    fwd_h = _forward(hi, rows, horizon)
    fwd_l = _forward(lo, rows, horizon)
    complete = (rows + horizon < n)[:, None]
    ended = np.where(complete, 1.0, np.nan)

    r = (np.log(fwd_c[:, -1, :]) - np.log(start)) * ended
    daily = np.diff(np.concatenate([np.log(start)[:, None, :], np.log(fwd_c)], axis=1), axis=1)
    efficiency = np.abs(r) / np.nansum(np.abs(daily), axis=1)
    top, bottom = np.nanmax(fwd_h, axis=1) * ended, np.nanmin(fwd_l, axis=1) * ended
    drawdown = bottom / start - 1
    path_range = (top - bottom) / start

    w = th.range_window
    upper = high.rolling(w, min_periods=w).max().to_numpy()[rows]
    lower = low.rolling(w, min_periods=w).min().to_numpy()[rows]
    above = fwd_h > upper[:, None, :]
    below = fwd_l < lower[:, None, :]
    up_break = above.any(axis=1)
    down_break = below.any(axis=1)
    first_up = np.where(up_break, above.argmax(axis=1), horizon)
    first_down = np.where(down_break, below.argmax(axis=1), horizon)
    both_same_bar = up_break & down_break & (first_up == first_down)
    end_close = fwd_c[:, -1, :]

    # The session (1 to h) on which the drawdown first went past its threshold: an alarm's lead time.
    # A fall of k sigma in log terms is an arithmetic return of expm1(-k sigma).
    dd_line = np.expm1(-th.drawdown_k * sigma_h)
    breached = (fwd_l / start[:, None, :] - 1) <= dd_line[:, None, :]
    dd_day = np.where(breached.any(axis=1), breached.argmax(axis=1) + 1.0, np.nan)

    big = th.trend_k * sigma_h
    up = (r >= big) & (efficiency >= th.trend_efficiency)
    down = (r <= -big) & (efficiency >= th.trend_efficiency)
    sideways = (np.abs(r) <= th.sideways_k * sigma_h) & (path_range <= th.sideways_range_k * sigma_h)
    state = np.full(r.shape, 3.0)                         # unclear
    state[sideways] = 2.0
    state[down] = 1.0
    state[up] = 0.0

    # Tradable: enter at the next close, hold h sessions; above cash and above SPY.
    entry = np.minimum(rows + 1, n - 1)
    exit_ = np.minimum(rows + 1 + horizon, n - 1)
    tradable_ok = (rows + 1 + horizon < n)[:, None]
    stock_ret = (logc[exit_] - logc[entry]) * np.where(tradable_ok, 1.0, np.nan)
    lc = np.log(cash.reindex(sessions).ffill().to_numpy())
    ls = np.log(spy.reindex(sessions).ffill().to_numpy())
    cash_ret = lc[exit_] - lc[entry]          # missing before the cash fund existed: no label, not zero
    spy_ret = ls[exit_] - ls[entry]

    # Position risks, on the actual holding: entry at close t+1, adverse excursion over the lows of
    # t+2..t+1+h relative to the entry close, never above zero; terminal on the exit close.
    entry_px = c[entry]
    later_lows = _forward(lo, entry, horizon)              # lows of entry+1 .. entry+h
    mae = np.minimum(np.nanmin(later_lows, axis=1) / entry_px - 1, 0.0) * np.where(tradable_ok, 1.0, np.nan)
    pos_ret = np.expm1(stock_ret)

    def frame(values, mask_to=None):
        v = np.asarray(values, dtype=float)
        if mask_to is not None:
            v = np.where(np.isnan(mask_to), np.nan, v)
        return pd.DataFrame(v, index=dates, columns=tickers).stack(future_stack=True)

    out = pd.DataFrame({
        "label": frame(stock_ret - cash_ret[:, None]),
        "label_rel": frame(stock_ret - spy_ret[:, None]),
        "ret_raw": frame(r),
        "drawdown": frame(drawdown),
        "sigma_h": frame(sigma_h),
        "upper": frame(upper), "lower": frame(lower),
        "up_break": frame(up_break, r), "down_break": frame(down_break, r),
        "failed_up": frame(up_break & (end_close <= upper), r),
        "failed_down": frame(down_break & (end_close >= lower), r),
        "both_same_bar": frame(both_same_bar, r),
        "large_loss": frame(r <= -th.loss_k * sigma_h, r),
        "deep_drawdown": frame(drawdown <= dd_line, r),
        "mae_position": frame(mae),
        "pos_large_loss": frame(pos_ret <= np.expm1(-th.loss_k * sigma_h), stock_ret),
        "pos_deep_drawdown": frame(mae <= dd_line, mae),
        "dd_breach_day": frame(dd_day, r),
        "trend_state": frame(state, r),
    })
    # Where sigma_h is unknown (too little history), no threshold applies: no event label.
    unknown = out["sigma_h"].isna()
    out.loc[unknown, ["large_loss", "deep_drawdown", "trend_state", "pos_large_loss", "pos_deep_drawdown"]] = np.nan
    label_end = pd.Series(sessions[np.minimum(rows + 1 + horizon, n - 1)], index=dates)
    label_end[~tradable_ok[:, 0]] = pd.NaT
    out["label_end"] = label_end.reindex(out.index.get_level_values(0)).to_numpy()
    out.index.names = ["date", "ticker"]
    return out


def current_trend(close: pd.DataFrame) -> pd.DataFrame:
    """The trend already in place at each close (detection, not forecast): +1 when the price is
    above a rising 50-day average that is above the 200-day, -1 for the mirror image, else 0."""
    ma50, ma200 = close.rolling(50, min_periods=40).mean(), close.rolling(200, min_periods=150).mean()
    rising = ma50.diff(10) > 0
    up = (close > ma50) & rising & (ma50 > ma200)
    down = (close < ma50) & ~rising & (ma50 < ma200)
    return up.astype(float) - down.astype(float)
