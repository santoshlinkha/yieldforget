"""Price history and the risk figures the optimizer works from.

Everything here is computed from adjusted daily closes:

- volatility, historical Sharpe and covariance over a recent lookback;
- a stress loss per asset: the deepest peak-to-trough drop in its whole history.
  An asset with a short history has not lived through a full crash, so its stress
  loss is taken from a longer-lived fund tracking the same market when one is
  named (QQQM from QQQ), and otherwise floored rather than trusted (a fund listed
  in 2024 has never seen 2008, 2020 or 2022).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows import local_store
from tradingagents.dataflows.vendors.yahoo.market import get_full_history_closes_many, get_history_bars_many

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def fetch_closes(tickers, required=(), cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Adjusted daily closes over each ticker's whole history, one column each.

    A ticker in ``required`` without prices is an error; any other one is left
    out, so one delisted fund in a scan list does not stop the scan. With
    ``cache_dir`` the prices are kept for good in the local market-data store
    (``tradingagents.dataflows.local_store``): each ticker's history is
    downloaded once, and afterwards only the sessions since its last stored
    close, through the last *final* close. Without it, the whole history is
    downloaded each time.
    """
    tickers = list(dict.fromkeys(tickers))
    market_store = local_store.store(cache_dir) if cache_dir else None
    if market_store is not None:
        cached = market_store.bars(tickers, get_history_bars_many)["Close"]
    else:
        end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        cached = get_full_history_closes_many(tickers, end)
    have = [t for t in tickers if t in cached.columns and cached[t].notna().any()]
    missing = [t for t in required if t not in have]
    if missing:
        raise ValueError(f"no price history for {', '.join(missing)}")
    if len(have) < len(tickers):
        logger.warning("no prices for %s; left out of the scan", ", ".join(set(tickers) - set(have)))
    return cached[have]


def latest_prices(cache_dir: str | Path, tickers) -> dict[str, float]:
    """Last stored close for each ticker, without downloading; empty for tickers never fetched."""
    market_store = local_store.store(cache_dir)
    if market_store is not None:
        return market_store.last_close(tickers)
    files = sorted(Path(cache_dir).glob("closes_*.pkl"))
    if not files:
        return {}
    frame = pd.read_pickle(files[-1])
    return {t: float(frame[t].dropna().iloc[-1]) for t in tickers if t in frame and frame[t].notna().any()}


def max_drawdown(prices: pd.Series) -> float:
    """Deepest peak-to-trough drop, as a positive fraction."""
    prices = prices.dropna()
    if prices.empty:
        return 0.0
    return float(1 - (prices / prices.cummax()).min())


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Pairwise covariance over unequal histories need not be PSD; clip its eigenvalues."""
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    return vectors @ np.diag(np.clip(values, 1e-10, None)) @ vectors.T


@dataclass
class AssetStats:
    tickers: list[str]          # the risky assets, in the order of every array below
    prices: dict[str, float]    # last close, cash ticker included
    vol: np.ndarray             # annualized volatility
    sharpe: np.ndarray          # historical annualized Sharpe over the lookback
    cov: np.ndarray | None      # annualized covariance; None when skipped
    worst_loss: np.ndarray      # stress loss, a positive fraction
    years: np.ndarray           # years of price history
    floored: np.ndarray         # whether the stress loss was floored for a short history
    cash_return: float          # the cash asset's trailing one-year return


def compute_stats(closes: pd.DataFrame, holdings: list[str], cash_ticker: str, lookback_years: float = 10,
                  short_history_years: float = 10, short_history_floor: float = 0.5,
                  proxies: dict[str, str] | None = None, cash_proxy: str | None = None,
                  with_cov: bool = True) -> AssetStats:
    """Risk and return figures for every holding but the cash one.

    Sharpe ratios are measured against what cash earned on the same days, not
    today's yield: a decade of near-zero rates would otherwise make every bond
    fund look like a loser against today's 4%. Before the cash fund existed,
    ``cash_proxy`` (BIL for SGOV) stands in for it.

    ``with_cov=False`` skips the covariance, the one part that grows with the
    square of the list, for screening hundreds of stocks before optimizing.
    """
    if cash_ticker not in closes:
        raise ValueError(f"cash ticker {cash_ticker} has no price history")
    returns = closes.pct_change(fill_method=None)
    last = closes.index[-1]

    cash = returns[cash_ticker]
    if cash_proxy and cash_proxy in returns:
        cash = cash.fillna(returns[cash_proxy])
    cash_return = float((1 + cash.dropna().iloc[-TRADING_DAYS:]).prod() - 1)

    recent = returns[returns.index > last - pd.DateOffset(years=lookback_years)]
    tickers = [t for t in holdings if t != cash_ticker]
    if not tickers:
        raise ValueError("the portfolio holds nothing but cash")
    vol = recent[tickers].std().to_numpy() * np.sqrt(TRADING_DAYS)
    if np.isnan(vol).any():
        missing = [t for t, v in zip(tickers, vol, strict=True) if np.isnan(v)]
        raise ValueError(f"not enough recent prices for {', '.join(missing)}")
    excess = recent[tickers].sub(cash.loc[recent.index].fillna(0.0), axis=0)
    sharpe = excess.mean().to_numpy() * TRADING_DAYS / vol
    cov = (_nearest_psd(recent[tickers].cov(min_periods=60).fillna(0.0).to_numpy() * TRADING_DAYS)
           if with_cov else None)

    def crash_history(t):
        names = [t, *([proxies[t]] if proxies and proxies.get(t) in closes else [])]
        years = max((last - closes[n].first_valid_index()).days / 365.25 for n in names)
        return years, max(max_drawdown(closes[n]) for n in names)

    years, observed = map(np.array, zip(*(crash_history(t) for t in tickers), strict=True))
    floored = (years < short_history_years) & (observed < short_history_floor) & (vol > 0.05)
    worst_loss = np.where(floored, short_history_floor, observed)

    prices = {t: float(closes[t].dropna().iloc[-1]) for t in [*tickers, cash_ticker]}
    return AssetStats(tickers, prices, vol, sharpe, cov, worst_loss, years, floored, cash_return)
