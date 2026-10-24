"""Forecasting features and labels, each computed from data known on its date only.

A sample is one asset on one *decision date* (by default the last session of
each month). Every feature uses bars up to and including that day's close, and
analyst and earnings records stamped strictly before that day. The label is the
log return over the next ``horizon`` sessions above cash (the T-bill fund), from
that close to the close ``horizon`` sessions later. Execution timing is the
backtest's business; this module only guarantees that no feature looks past its date.

Price and volume: momentum over 1, 3, 6 and 12 months (the last skipping its
latest month), one-week reversal, volatility (all and downside), beta to SPY,
distance from the 52-week high and from the 200-day average, the latest
1-year drawdown, high-low range, dollar volume, its trend, and Amihud
illiquidity.

Market (the same for every asset on a date): VIX, its slope to 3-month VIX,
the 10-year yield's 3-month change, the T-bill rate, SPY's 3-month return and
1-month volatility.

Analysts (stocks; from Yahoo's dated upgrades and downgrades): the upside to the
median of each firm's latest price target in the past year, how many firms that
is, net upgrades and net target raises over 90 days.

Earnings (stocks; from Yahoo's earnings dates): the last reported surprise,
the average of the last four, and sessions since the last report.

Cross-sectional features are ranked within each date into [-0.5, 0.5], so a
model learns relative standing, not levels that drift over decades. Market
features stay as levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EWMA_LAMBDA = 0.94

PRICE_FEATURES = ["mom_1m", "mom_3m", "mom_6m", "mom_12_1", "rev_1w", "vol_1m", "vol_3m", "downvol_3m",
                  "beta_1y", "dist_52w_high", "gap_ma200", "dd_1y", "range_1m", "log_dollar_vol",
                  "volume_trend", "amihud"]
MARKET_FEATURES = ["vix", "vix_slope", "tnx_chg_3m", "irx", "spy_mom_3m", "spy_vol_1m"]
ANALYST_FEATURES = ["target_upside", "n_target_firms", "net_upgrades_90d", "net_target_raises_90d"]
EARNINGS_FEATURES = ["last_surprise", "surprise_4q", "sessions_since_report"]


def decision_dates(index: pd.DatetimeIndex, start=None, every: str | int = "month") -> pd.DatetimeIndex:
    """The last session of each month, or every ``every`` sessions when an int, in ``index``."""
    index = index[index >= pd.Timestamp(start)] if start is not None else index
    if isinstance(every, int):
        return index[::every]
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Each row ranked into [-0.5, 0.5]; missing stays missing."""
    ranks = frame.rank(axis=1, pct=True) - 0.5
    ranks[frame.notna().sum(axis=1) <= 1] = np.nan   # one asset alone has no rank
    return ranks


def price_features(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame,
                   market: pd.Series) -> dict[str, pd.DataFrame]:
    """Every price feature on every day (the caller samples the decision dates)."""
    r = np.log(close).diff()
    mkt = np.log(market).diff()
    f = {
        "mom_1m": np.log(close / close.shift(21)),
        "mom_3m": np.log(close / close.shift(63)),
        "mom_6m": np.log(close / close.shift(126)),
        "mom_12_1": np.log(close.shift(21) / close.shift(252)),
        "rev_1w": np.log(close / close.shift(5)),
        "vol_1m": r.rolling(21, min_periods=15).std() * np.sqrt(TRADING_DAYS),
        "vol_3m": r.rolling(63, min_periods=40).std() * np.sqrt(TRADING_DAYS),
        "downvol_3m": r.clip(upper=0).pow(2).rolling(63, min_periods=40).mean().pow(0.5) * np.sqrt(TRADING_DAYS),
        "dist_52w_high": close / close.rolling(252, min_periods=126).max() - 1,
        "gap_ma200": close / close.rolling(200, min_periods=150).mean() - 1,
        "range_1m": ((high - low) / close).rolling(21, min_periods=15).mean(),
    }
    # The deepest point of the past year, measured from the running one-year peak.
    peak = close.rolling(252, min_periods=126).max()
    f["dd_1y"] = (close / peak - 1).rolling(252, min_periods=126).min()
    cov = r.rolling(252, min_periods=126).cov(mkt)
    f["beta_1y"] = cov.div(mkt.rolling(252, min_periods=126).var(), axis=0)
    dollar = close * volume
    f["log_dollar_vol"] = np.log(dollar.rolling(21, min_periods=15).mean().replace(0, np.nan))
    f["volume_trend"] = np.log(volume.rolling(21, min_periods=15).mean()
                               / volume.rolling(126, min_periods=90).mean().replace(0, np.nan))
    f["amihud"] = (r.abs() / dollar.replace(0, np.nan)).rolling(21, min_periods=15).mean() * 1e6
    return f


def market_features(market: dict[str, pd.Series]) -> pd.DataFrame:
    spy = market["SPY"]
    out = pd.DataFrame(index=spy.index)
    out["vix"] = market.get("^VIX")
    if "^VIX3M" in market:
        out["vix_slope"] = market["^VIX"] / market["^VIX3M"] - 1
    out["tnx_chg_3m"] = market["^TNX"] - market["^TNX"].shift(63) if "^TNX" in market else np.nan
    out["irx"] = market.get("^IRX")
    out["spy_mom_3m"] = np.log(spy / spy.shift(63))
    out["spy_vol_1m"] = np.log(spy).diff().rolling(21, min_periods=15).std() * np.sqrt(TRADING_DAYS)
    return out.ffill()


def analyst_features(actions: pd.DataFrame, close: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """From one ticker's dated upgrades/downgrades: known strictly before each date."""
    out = pd.DataFrame(index=dates, columns=ANALYST_FEATURES, dtype=float)
    if actions is None or actions.empty:
        return out
    a = actions.copy()
    a.index = pd.to_datetime(a.index).tz_localize(None) if getattr(a.index, "tz", None) else pd.to_datetime(a.index)
    a = a.sort_index()
    for d in dates:
        seen = a[a.index < d]
        year = seen[seen.index >= d - pd.Timedelta(days=365)]
        targets = year[year["currentPriceTarget"].fillna(0) > 0]
        latest = targets.groupby("Firm")["currentPriceTarget"].last()
        px = close.loc[:d].iloc[-1] if len(close.loc[:d].dropna()) else np.nan
        if len(latest) and px and np.isfinite(px):
            out.at[d, "target_upside"] = float(np.log(latest.median() / px))
            out.at[d, "n_target_firms"] = float(len(latest))
        q = seen[seen.index >= d - pd.Timedelta(days=90)]
        out.at[d, "net_upgrades_90d"] = float((q["Action"] == "up").sum() - (q["Action"] == "down").sum())
        if "priceTargetAction" in q:
            out.at[d, "net_target_raises_90d"] = float((q["priceTargetAction"] == "Raises").sum()
                                                       - (q["priceTargetAction"] == "Lowers").sum())
    return out


def earnings_features(reports: pd.DataFrame, sessions: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """From one ticker's earnings dates: reports made strictly before each date."""
    out = pd.DataFrame(index=dates, columns=EARNINGS_FEATURES, dtype=float)
    if reports is None or reports.empty:
        return out
    e = reports.dropna(subset=["Reported EPS"]).copy()
    idx = pd.to_datetime(e.index)
    e.index = (idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx).normalize()
    e = e.sort_index()
    for d in dates:
        past = e[e.index < d]
        if past.empty:
            continue
        surprise = past["Surprise(%)"].astype(float) / 100
        out.at[d, "last_surprise"] = float(np.clip(surprise.iloc[-1], -2, 2))
        out.at[d, "surprise_4q"] = float(np.clip(surprise.iloc[-4:].mean(), -2, 2))
        out.at[d, "sessions_since_report"] = float(((sessions > past.index[-1]) & (sessions <= d)).sum())
    return out


def build_panel(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame,
                market: dict[str, pd.Series], cash: str, dates: pd.DatetimeIndex, horizon: int = 21,
                analyst: dict[str, pd.DataFrame] | None = None, earnings: dict[str, pd.DataFrame] | None = None
                ) -> pd.DataFrame:
    """One row per (date, ticker): ranked features, market levels, and the forward label.

    ``label`` is the log return above cash over the next ``horizon`` sessions
    (NaN where it has not happened yet); ``label_end`` is the session it ends on,
    which the walk-forward split uses to keep training labels out of the test period.
    """
    tickers = [t for t in close.columns if t != cash]
    sessions = close.index
    pf = price_features(close[tickers], high[tickers], low[tickers], volume[tickers], market["SPY"])
    mf = market_features(market).reindex(sessions).ffill()
    logc = np.log(close)
    fwd = logc.shift(-horizon) - logc
    label = fwd[tickers].sub(fwd[cash].fillna(0.0), axis=0)
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    label_end = pd.Series(sessions[np.minimum(pos.to_numpy() + horizon, len(sessions) - 1)], index=sessions)
    label_end[pos.to_numpy() + horizon >= len(sessions)] = pd.NaT

    frames = []
    for name, frame in pf.items():
        frames.append(_rank(frame.loc[dates]).stack(future_stack=True).rename(name))
    panel = pd.concat(frames, axis=1)
    raw_vol = pf["vol_3m"].loc[dates].stack(future_stack=True).rename("raw_vol_3m")
    panel = panel.join(raw_vol)
    # Daily EWMA volatility (RiskMetrics, lambda 0.94) known at each date: the scale of the
    # forecast distribution. Close-to-close returns, not intraday realized volatility.
    daily = np.log(close[tickers]).diff()
    ewma = daily.pow(2).ewm(alpha=1 - EWMA_LAMBDA, min_periods=60).mean().pow(0.5)
    panel = panel.join(ewma.loc[dates].stack(future_stack=True).rename("ewma_vol_daily"))
    # What cash earned over the last month, scaled to the horizon: the hurdle for "a gain".
    cash_log = np.log(close[cash]).diff()
    cash_h = (cash_log.rolling(21, min_periods=10).mean() * horizon).reindex(dates).fillna(0.0)
    panel["cash_h"] = cash_h.reindex(panel.index.get_level_values(0)).to_numpy()
    for extra, names in ((analyst, ANALYST_FEATURES), (earnings, EARNINGS_FEATURES)):
        if not extra:
            continue
        rows = []
        for t, frame in extra.items():
            if t in tickers and frame is not None:
                f = frame.reindex(dates)
                f.index = pd.MultiIndex.from_product([dates, [t]])
                rows.append(f)
        if rows:
            wide = pd.concat(rows).reindex(panel.index)
            for name in names:
                # Ranked within each date across the tickers that have the feature.
                panel[name] = wide[name].groupby(level=0).rank(pct=True) - 0.5
    m = mf.loc[dates]
    panel = panel.join(m.reindex(panel.index.get_level_values(0)).set_index(panel.index))
    panel["label"] = label.loc[dates].stack(future_stack=True).reindex(panel.index)
    panel["label_end"] = label_end.loc[dates].reindex(panel.index.get_level_values(0)).to_numpy()
    panel.index.names = ["date", "ticker"]
    # A ticker needs prices on the date and a year of history to be in the sample.
    have = close[tickers].loc[dates].stack(future_stack=True).reindex(panel.index).notna()
    return panel[have & panel["mom_12_1"].notna()]
