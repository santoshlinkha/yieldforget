"""One row per (decision date, stock): features known at the close, and every target after it.

Feature groups (for ablation, each switched on or off as a whole):

- **technical**: the fund models' price and volume features (momentum, reversal,
  volatility, beta, distance from highs, drawdown, range, liquidity), plus the
  slopes of the 20- and 50-day averages, where the close sits in its 20-day and
  52-week ranges, strength against SPY and against the stock's own sector fund
  over 1 and 3 months, the week's volume against its 3-month average, and the
  trend already in place (detected from moving averages: a feature, not a forecast);
- **market**: VIX and its slope, the 10-year yield's change, the bill rate, SPY's
  return and volatility (levels, the same for every stock on a date);
- **fundamental**: SEC filings as known on the date (``fundamentals``), with
  valuation yields on the price as it then traded;
- **macro**: economic series as published by the date (``macro``).

Cross-sectional features are ranked within each date into [-0.5, 0.5]; market and
economic features stay as levels. Every feature uses data through the decision
date's close only. Decision dates are every 20 sessions from 2006 (every 10 for
the 5-session horizon), the same dates for every stock, plus the latest close.

``cost_rt`` is the estimated round-trip trading cost: the bid-ask spread from daily
highs, lows and closes (Abdi and Ranaldo, 2017, over the 21 sessions before the
date, so it needs no future bar), plus 2 bp of slippage a side; Robinhood charges
no commission. It is reported beside the forecasts, never folded into them silently.

``in_index``: whether a member of today's S&P 500 had already been added to the
index on the date. Training uses only such rows (and the requested stocks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.features import (MACRO_FEATURES, MARKET_FEATURES, PRICE_FEATURES, _rank,
                                             decision_dates, market_features, price_features)

from . import fundamentals as FU
from .data import CASH, StockData, raw_close
from .targets import Thresholds, current_trend, ewma_vol, labels

TECHNICAL_EXTRA = ["ma20_slope", "ma50_slope", "range_pos_20", "range_pos_52w", "rel_spy_1m", "rel_spy_3m",
                   "rel_sector_1m", "rel_sector_3m", "volume_chg_1w", "current_trend"]
GROUPS = {
    "technical": list(PRICE_FEATURES) + TECHNICAL_EXTRA,
    "market": list(MARKET_FEATURES),
    "fundamental": list(FU.FEATURES) + list(FU.VALUATION) + ["filing_age_days"],
    "macro": list(MACRO_FEATURES),
}
DEFAULT_GROUPS = ("technical", "market", "fundamental", "macro")
UNRANKED = {"current_trend"}
SLIPPAGE = 0.0002


def every_for(horizon: int) -> int:
    return 10 if horizon <= 5 else 20


def features_for(groups=DEFAULT_GROUPS) -> list[str]:
    return [f for g in groups for f in GROUPS[g]]


def spread_cost(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Round-trip cost estimate at each close: the Abdi-Ranaldo spread over the previous 21 sessions
    (each term uses the next day's mid, so the window stops a day before the date), plus slippage."""
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2
    term = 4 * (c - eta) * (c - eta.shift(-1))
    s2 = term.shift(1).rolling(21, min_periods=15).mean().clip(lower=0)
    return (np.sqrt(s2) + 2 * SLIPPAGE).clip(lower=2 * SLIPPAGE, upper=0.05)


def _technical(data: StockData, tickers: list[str], dates) -> dict[str, pd.DataFrame]:
    close, high, low, volume = (f[tickers] for f in (data.close, data.high, data.low, data.volume))
    spy = data.close["SPY"]
    f = price_features(close, high, low, volume, spy)
    ma20, ma50 = close.rolling(20, min_periods=15).mean(), close.rolling(50, min_periods=40).mean()
    f["ma20_slope"] = np.log(ma20 / ma20.shift(5))
    f["ma50_slope"] = np.log(ma50 / ma50.shift(10))
    h20, l20 = high.rolling(20, min_periods=15).max(), low.rolling(20, min_periods=15).min()
    f["range_pos_20"] = (close - l20) / (h20 - l20).replace(0, np.nan)
    h52, l52 = high.rolling(252, min_periods=126).max(), low.rolling(252, min_periods=126).min()
    f["range_pos_52w"] = (close - l52) / (h52 - l52).replace(0, np.nan)
    lc = np.log(close)
    for n, tag in ((21, "1m"), (63, "3m")):
        own = lc - lc.shift(n)
        f[f"rel_spy_{tag}"] = own.sub(np.log(spy / spy.shift(n)), axis=0)
        sector = pd.DataFrame(index=close.index, columns=tickers, dtype=float)
        for t in tickers:
            etf = data.info.get(t, {}).get("sector_etf")
            if etf and etf in data.close.columns:
                s = data.close[etf]
                sector[t] = np.log(s / s.shift(n))
        f[f"rel_sector_{tag}"] = own - sector
    f["volume_chg_1w"] = np.log(volume.rolling(5, min_periods=4).mean()
                                / volume.rolling(63, min_periods=40).mean().replace(0, np.nan))
    f["current_trend"] = current_trend(close)
    return f


def _fundamental(data: StockData, tickers: list[str], dates) -> dict[str, pd.DataFrame]:
    cols = {k: {} for k in FU.FEATURES + FU.VALUATION + ["filing_age_days"]}
    for t in tickers:
        facts = data.facts.get(t)
        if not facts:
            continue
        acts = data.actions.get(t) or {}
        h = FU.history(facts, dates, acts.get("splits"))
        v = FU.valuation(h, raw_close(data.close[t], acts).reindex(dates))
        for k in FU.FEATURES + ["filing_age_days"]:
            cols[k][t] = h[k]
        for k in FU.VALUATION:
            cols[k][t] = v[k]
    return {k: pd.DataFrame(v, index=dates).reindex(columns=tickers) for k, v in cols.items()}


def build(data: StockData, horizon: int, dates: pd.DatetimeIndex | None = None, th: Thresholds = Thresholds(),
          groups=DEFAULT_GROUPS, start: str = "2006-01-01", macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """The panel for one horizon. ``macro``: precomputed economic features (shared across horizons)."""
    tickers = data.stocks
    sessions = data.close.index
    if dates is None:
        dates = decision_dates(sessions, start=start, every=every_for(horizon)).union(pd.DatetimeIndex([sessions[-1]]))
    frames = []
    if "technical" in groups:
        for name, frame in _technical(data, tickers, dates).items():
            if name not in GROUPS["technical"] and name != "vol_3m":
                continue
            sampled = frame.reindex(dates)
            frames.append((sampled if name in UNRANKED else _rank(sampled)).stack(future_stack=True).rename(name))
    if "fundamental" in groups:
        for name, frame in _fundamental(data, tickers, dates).items():
            frames.append(_rank(frame).stack(future_stack=True).rename(name))
    panel = pd.concat(frames, axis=1) if frames else pd.DataFrame(
        index=pd.MultiIndex.from_product([dates, tickers]))
    panel.index.names = ["date", "ticker"]
    idx = panel.index
    level = lambda frame: frame.reindex(dates).reindex(idx.get_level_values(0)).set_index(idx)  # noqa: E731
    if "market" in groups:
        market = {m: data.close[m].dropna() for m in ("SPY", "^VIX", "^VIX3M", "^IRX", "^TNX") if m in data.close}
        panel = panel.join(level(market_features(market).reindex(sessions).ffill()))
    if "macro" in groups and macro is not None:
        panel = panel.join(level(macro.reindex(dates)))

    close, high, low = data.close[tickers], data.high[tickers], data.low[tickers]
    raw_vol = np.log(close).diff().rolling(63, min_periods=40).std() * np.sqrt(252)
    panel["raw_vol_3m"] = raw_vol.reindex(dates).stack(future_stack=True).reindex(idx).to_numpy()
    panel["ewma_vol_daily"] = ewma_vol(close).reindex(dates).stack(future_stack=True).reindex(idx).to_numpy()
    panel["cost_rt"] = spread_cost(close, high, low).reindex(dates).stack(future_stack=True).reindex(idx).to_numpy()
    cash = data.close[CASH]
    cash_log = np.log(cash).diff()
    panel["cash_h"] = (cash_log.rolling(21, min_periods=10).mean() * horizon).reindex(dates).fillna(0.0) \
        .reindex(idx.get_level_values(0)).to_numpy()
    targets = labels(close, high, low, cash, data.close["SPY"], dates, horizon, th)
    panel = panel.join(targets)
    added = pd.Series({t: pd.Timestamp(i["added"]) for t, i in data.info.items() if i.get("added")})
    when = added.reindex(idx.get_level_values(1)).to_numpy()
    in_index = pd.isna(when) | (idx.get_level_values(0).to_numpy() >= when)
    requested = idx.get_level_values(1).isin(data.requested)
    panel["in_index"] = in_index | requested
    # A row needs a price on its date and a year of history.
    have = close.reindex(dates).stack(future_stack=True).reindex(idx).notna().to_numpy()
    year = raw_vol.reindex(dates).stack(future_stack=True).reindex(idx).notna().to_numpy() & \
        close.shift(252).reindex(dates).stack(future_stack=True).reindex(idx).notna().to_numpy()
    return panel[have & year]
