"""Daily bars from Yahoo Finance for research: adjusted close, high, low and volume.

They come from the local market-data store (``tradingagents.dataflows.local_store``):
downloaded once, then only the new sessions. Closes are dividend- and split-adjusted,
so returns are total returns; volume is as reported.

``MARKET`` lists the series the market features use: SPY, VIX (^VIX), 3-month VIX
(^VIX3M), the 13-week bill rate (^IRX) and the 10-year Treasury yield (^TNX).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.dataflows import local_store
from tradingagents.dataflows.vendors.yahoo.market import BAR_FIELDS as FIELDS
from tradingagents.dataflows.vendors.yahoo.market import get_history_bars_many

MARKET = ("SPY", "^VIX", "^VIX3M", "^IRX", "^TNX")


def bars(tickers, cache_dir: str | Path | None = None, download=get_history_bars_many) -> dict[str, pd.DataFrame]:
    """``{"Close": frame, "High": ..., "Low": ..., "Volume": ...}``, one column per ticker."""
    tickers = list(dict.fromkeys(tickers))
    market_store = local_store.store(cache_dir) if cache_dir else None
    if market_store is not None:
        return market_store.bars(tickers, download)
    return download(tickers, None, (date.today() + timedelta(days=1)).isoformat())


__all__ = ["FIELDS", "MARKET", "bars"]
